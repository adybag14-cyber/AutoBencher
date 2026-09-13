#!/usr/bin/env python3
"""Compare stored FC/embedding weights and quantization parameters across contexts."""
import argparse, gc, hashlib, json, mmap, pathlib
from ai_edge_litert import schema_py_generated as schema
from prepare_litert_mobile import tflite_sections

def inventory(path):
    records={}
    with path.open('rb') as f:
        data=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        for sid,base,end in tflite_sections(path):
            model=schema.Model.GetRootAsModel(memoryview(data)[base:end],0)
            seen=set()
            def buffer_hash(index):
                buffer=model.Buffers(index)
                offset=buffer._tab.Vector(buffer._tab.Offset(4)) if buffer.DataLength() else buffer.Offset()
                size=buffer.DataLength() or buffer.Size()
                return hashlib.sha256(memoryview(data)[base+offset:base+offset+size]).hexdigest()
            for i in range(model.SubgraphsLength()):
                sg=model.Subgraphs(i)
                for j in range(sg.OperatorsLength()):
                    op=sg.Operators(j);code=model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
                    if code not in (schema.BuiltinOperator.FULLY_CONNECTED,schema.BuiltinOperator.EMBEDDING_LOOKUP):continue
                    weight=sg.Tensors(op.Inputs(1))
                    if weight.Buffer() in seen:continue
                    seen.add(weight.Buffer())
                    name=f'{sid}:{weight.Name().decode()}'
                    q=weight.Quantization()
                    quant={}
                    if q is not None:
                        quant={'scales':q.ScaleAsNumpy().tolist() if q.ScaleLength() else [],
                               'zero_points':q.ZeroPointAsNumpy().tolist() if q.ZeroPointLength() else [],
                               'dimension':q.QuantizedDimension(),'details_type':q.DetailsType()}
                        if q.DetailsType()==schema.QuantizationDetails.BlockwiseQuantization:
                            bq=schema.BlockwiseQuantization();table=q.Details();bq.Init(table.Bytes,table.Pos)
                            quant['block_size']=bq.BlockSize()
                            quant['scale_hash']=buffer_hash(sg.Tensors(bq.Scales()).Buffer())
                            if bq.ZeroPoints()>=0:quant['zero_point_hash']=buffer_hash(sg.Tensors(bq.ZeroPoints()).Buffer())
                    record={'shape':[weight.Shape(k) for k in range(weight.ShapeLength())],
                            'type':weight.Type(),'weight_sha256':buffer_hash(weight.Buffer()),
                            'quantization_sha256':hashlib.sha256(json.dumps(quant,sort_keys=True).encode()).hexdigest()}
                    if name in records and records[name]!=record:raise ValueError('Conflicting weight name')
                    records[name]=record
        del model,sg,op,weight,q
        if 'bq' in locals():del bq,table
        gc.collect();data.close()
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True)
    p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();reference=inventory(a.reference);candidate=inventory(a.candidate)
    mismatch=[name for name in sorted(set(reference)|set(candidate)) if reference.get(name)!=candidate.get(name)]
    report={'reference':str(a.reference),'candidate':str(a.candidate),
            'reference_weight_buffers':len(reference),'candidate_weight_buffers':len(candidate),
            'mismatches':mismatch,'passed':not mismatch and len(reference)==296,
            'scope':'Stored fully connected and embedding weight bytes, shapes, types and quantization parameters. Context-dependent graph execution and accuracy require separate tests.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
