"""Remove unused prefill KV inputs without changing NPU code or tensor values.

LiteRT-LM 0.17 allocates shared caches from prefill input requirements first.
Unused final-layer prefill inputs receive HostMemory, which is then incorrectly
reused for NPU decode. Omitting those inputs makes the runtime allocate them
from the decode graph's real NPU requirements.
"""
import argparse,gc,hashlib,json,mmap,pathlib,re,shutil
import flatbuffers
from flatbuffers import flexbuffers
from ai_edge_litert import schema_py_generated as s
class SharedStringBuilder(flatbuffers.Builder):
    def __init__(self,size):
        super().__init__(size);self.string_offsets={}
    def CreateString(self,value,encoding='utf-8',errors='strict'):
        key=value.encode(encoding,errors) if isinstance(value,str) else bytes(value)
        if key not in self.string_offsets:
            self.string_offsets[key]=super().CreateString(value,encoding,errors)
        return self.string_offsets[key]

def buffer_inventory(model):
    return [{'offset':int(b.offset),'size':int(b.size),
             'inline_sha256':hashlib.sha256(b.data).hexdigest() if b.data is not None else None}
            for b in model.buffers]


def rewrite(model):
    changes=[]
    decode=next(sig for sig in model.signatureDefs if sig.signatureKey==b'decode')
    decode_names={item.name for item in decode.inputs}
    for sig in model.signatureDefs:
        if not re.fullmatch(rb'prefill_\d+',sig.signatureKey):continue
        if sum(other.subgraphIndex==sig.subgraphIndex for other in model.signatureDefs)!=1:
            raise ValueError("Cannot prune a graph shared by multiple signatures")
        graph=model.subgraphs[sig.subgraphIndex]
        used={int(i) for op in graph.operators for i in op.inputs if int(i)>=0}
        used.update(map(int,graph.outputs))
        removable=[item for item in sig.inputs if re.fullmatch(rb'kv_cache_[kv]_\d+',item.name)
                   and item.tensorIndex not in used]
        for item in removable:
            if item.name not in decode_names:raise ValueError('Unused prefill cache has no decode input')
            if item.tensorIndex not in graph.inputs:raise ValueError('Signature and graph input mismatch')
        removed={item.tensorIndex for item in removable}
        sig.inputs=[item for item in sig.inputs if item.tensorIndex not in removed]
        graph.inputs=[int(i) for i in graph.inputs if int(i) not in removed]
        changes.extend({'signature':sig.signatureKey.decode(),'name':item.name.decode(),
                        'tensor_index':item.tensorIndex} for item in removable)
    if not changes:raise ValueError('No unused prefill KV inputs found')
    return changes

def repair(source,output):
    if output.exists():raise FileExistsError(output)
    with source.open('rb') as f:
        data=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        model=s.ModelT.InitFromObj(s.Model.GetRootAsModel(data,0))
        if model.externalBuffers or model.externalBufferGroups:raise ValueError('Unsupported external groups')
        buffers=buffer_inventory(model)
        external=[int(b.offset) for b in model.buffers if b.offset and b.size]
        bytecode=set()
        for graph in model.subgraphs:
            for op in graph.operators:
                if op.largeCustomOptionsOffset:external.append(int(op.largeCustomOptionsOffset))
                code=model.operatorCodes[op.opcodeIndex]
                if code.customCode==b'DISPATCH_OP':
                    opt=flexbuffers.Loads(op.customOptions.tobytes())
                    start,size=int(opt['bytecode_offset']),int(opt['bytecode_size'])
                    if not 0<start<start+size<=len(data):raise ValueError('Invalid bytecode range')
                    external.append(start);bytecode.add((start,size))
        if not bytecode:raise ValueError('Expected compiled NPU model')
        limit=min(external)
        changes=rewrite(model)
        builder=SharedStringBuilder(1024*1024);packed=model.Pack(builder);builder.Finish(packed,file_identifier=b'TFL3')
        blob=bytes(builder.Output())
        if len(blob)>limit:raise ValueError('Metadata exceeded reserved space')
        rebuilt=s.ModelT.InitFromObj(s.Model.GetRootAsModel(blob,0))
        if buffer_inventory(rebuilt)!=buffers:raise ValueError('Changed constant buffer')
        payloads=[{'offset':o,'bytes':n,'sha256':hashlib.sha256(memoryview(data)[o:o+n]).hexdigest()} for o,n in sorted(bytecode)]
        del model,graph,op,code;gc.collect();data.close()
    shutil.copyfile(source,output)
    with output.open('r+b') as f:f.write(blob);f.write(b'\0'*(limit-len(blob)))
    with source.open('rb') as f,output.open('rb') as g:
        f.seek(limit);g.seek(limit)
        while chunk:=f.read(8*1024*1024):
            if chunk!=g.read(len(chunk)):raise ValueError('Changed bytes outside metadata')
    def sha(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    report={'source_sha256':sha(source),'sha256':sha(output),'bytes':output.stat().st_size,
            'changes':changes,'all_bytes_after_metadata_identical':True,'buffers_identical':True,
            'bytecode_regions':payloads,'metadata_bytes':len(blob),'reserved_bytes':limit}
    output.with_suffix('.repair.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    print(json.dumps(repair(a.source,a.output)),flush=True)
