#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect stored LiteRT weight types and scales; never infer precision from a filename."""
from __future__ import annotations
import argparse, gc, hashlib, io, json, mmap, re, tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import peek_litertlm_file


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def inspect_section(data,base,end,section_id,selection):
    view=memoryview(data)[base:end]
    model=schema.Model.GetRootAsModel(view,0)
    names={v:k for k,v in vars(schema.TensorType).items() if isinstance(v,int)}
    opnames={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    counts=Counter(); unique={}; groups={}; errors=[]; seen_scales=set(); invalid_scales=0
    for index in range(model.SubgraphsLength()):
        sg=model.Subgraphs(index)
        for i in range(sg.OperatorsLength()):
            op=sg.Operators(i); code=model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if code not in (schema.BuiltinOperator.FULLY_CONNECTED,schema.BuiltinOperator.EMBEDDING_LOOKUP): continue
            weight=sg.Tensors(op.Inputs(1)); buffer=model.Buffers(weight.Buffer())
            output=sg.Tensors(op.Outputs(0)).Name().decode(errors='replace') if op.OutputsLength() else ''
            name=weight.Name().decode(errors='replace'); dtype=names.get(weight.Type(),str(weight.Type()))
            key=f'{section_id}:{weight.Buffer()}'
            unique.setdefault(key,{'name':name,'type':dtype,'shape':[weight.Shape(k) for k in range(weight.ShapeLength())],
                'stored_bytes':int(buffer.DataLength() or buffer.Size()),'operation':opnames.get(code,str(code))})
            counts[dtype]+=1
            text=output+' '+name
            layer=re.search(r'LlamaDecoderLayer_(\d+)/',text)
            group='attention' if 'LlamaAttention_self_attn' in text else ('mlp' if 'LlamaMLP_mlp' in text else None)
            if layer and group:
                label=f'{group}.{int(layer.group(1))}'; groups.setdefault(label,set()).add(dtype)
                if selection:
                    protected=selection['attention_int8_layers' if group=='attention' else 'mlp_int8_layers']
                    expected='INT8' if int(layer.group(1)) in protected else 'INT4'
                    if dtype!=expected: errors.append(f'{label}: expected {expected}, stored {dtype}')
            q=weight.Quantization()
            if q is None: continue
            if q.DetailsType()==schema.QuantizationDetails.BlockwiseQuantization:
                bq=schema.BlockwiseQuantization();tab=q.Details();bq.Init(tab.Bytes,tab.Pos)
                st=sg.Tensors(bq.Scales()); bidx=st.Buffer()
                if bidx in seen_scales: continue
                seen_scales.add(bidx); b=model.Buffers(bidx)
                off=b._tab.Vector(b._tab.Offset(4)) if b.DataLength() else b.Offset()
                size=b.DataLength() or b.Size()
                if st.Type() not in (schema.TensorType.FLOAT16,schema.TensorType.FLOAT32):
                    errors.append('Unexpected scale tensor type');continue
                typ=np.float16 if st.Type()==schema.TensorType.FLOAT16 else np.float32
                values=np.ndarray(size//np.dtype(typ).itemsize,dtype=typ,buffer=data,offset=base+off)
                invalid_scales+=int(np.count_nonzero(~np.isfinite(values)|(values<=0)))
            elif q.ScaleLength():
                values=q.ScaleAsNumpy()
                invalid_scales+=int(np.count_nonzero(~np.isfinite(values)|(values<=0)))
    return {'section':section_id,'fc_embedding_operator_weight_types':dict(counts),
        'unique_weight_buffers':list(unique.values()),'layer_weight_types':{k:sorted(v) for k,v in groups.items()},
        'invalid_scale_entries':invalid_scales,'precision_mismatches':sorted(set(errors))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True)
    ap.add_argument('--selection',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();selection=json.loads(args.selection.read_text())['selection'] if args.selection else None
    with tempfile.TemporaryDirectory(prefix='litert_precision_audit_') as temp:
        output=io.StringIO();peek_litertlm_file(str(args.artifact),temp,output);peek=output.getvalue()
        sections=[]
        pattern=r'Section (\d+):\n  Items:\n(.*?)\n  Begin Offset: (\d+)\n  End Offset:\s+(\d+)\n  Data Type:\s+(\S+)'
        for match in re.finditer(pattern,peek,re.S):
            if match.group(5)=='TFLiteModel': sections.append((int(match.group(1)),int(match.group(3)),int(match.group(4))))
        if not sections: raise RuntimeError('No TFLite sections found')
        with args.artifact.open('rb') as f:
            data=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
            results=[inspect_section(data,b,e,sid,selection) for sid,b,e in sections]
            gc.collect();data.close()
    groups={}
    for result in results: groups.update(result['layer_weight_types'])
    errors=[e for r in results for e in r['precision_mismatches']]
    invalid=sum(r['invalid_scale_entries'] for r in results)
    if selection and (len(groups)!=84): errors.append(f'Expected 84 identifiable attention/MLP groups, found {len(groups)}')
    report={'artifact':str(args.artifact),'sha256':digest(args.artifact),'bytes':args.artifact.stat().st_size,
        'audit_kind':'stored_fc_embedding_weight_types_and_scales','sections':results,
        'identifiable_layer_groups':len(groups),'invalid_scale_entries':invalid,
        'errors':errors,'passed':not errors and invalid==0,
        'scope':'Stored tensor types and scale validity only. No runtime, accuracy or context-retention claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='sections'}),flush=True)
    if not report['passed']: raise SystemExit(2)

if __name__=='__main__': main()
