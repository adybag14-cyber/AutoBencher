#!/usr/bin/env python3
"""Choose precision from independent calibration error, before evaluation."""
import argparse, hashlib, json, math, pathlib

def select(calibration, maximum_ratio=1.25):
    if not math.isfinite(maximum_ratio) or maximum_ratio<1:
        raise ValueError('The error ratio must be finite and at least one')
    groups=calibration['groups']
    if len(groups)!=84 or {(g['group'],g['layer']) for g in groups} != {(kind,i) for kind in ['attn','mlp'] for i in range(42)}:
        raise ValueError('Expected all 84 MiniCPM5 attention and MLP groups')
    if calibration['calibration'].get('no_benchmark_data') is not True:
        raise ValueError('Calibration must declare an independent corpus with no benchmark data')
    for g in groups:
        if any(not math.isfinite(g[k]) or g[k]<0 for k in ['int4_err','int8_err','signal']) or not math.isfinite(g['relative_gain']):
            raise ValueError('Invalid calibration error or signal')
    baseline=sum(g['int8_err']/max(g['signal'],1e-30) for g in groups)
    budget=(maximum_ratio-1)*baseline
    extra=0.0; int4=[]
    for g in sorted((g for g in groups if g['group']=='mlp'),key=lambda g:(g['relative_gain'],g['layer'])):
        cost=max(0,g['int4_err']-g['int8_err'])/max(g['signal'],1e-30)
        if extra+cost<=budget:int4.append(g['layer']);extra+=cost
    return {'schema_version':1,'selection':{'attention_int8_layers':list(range(42)),
        'mlp_int8_layers':[i for i in range(42) if i not in int4],'embedding_int8':True,'lm_head_int8':True},
        'maximum_proxy_error_ratio':maximum_ratio,'all_int8_proxy_error':baseline,
        'selected_extra_proxy_error':extra,'int4_mlp_layers':int4,
        'calibration_provenance':calibration['calibration'],
        'benchmark_inputs_used_for_selection':False,
        'scope':'Activation-weighted calibration proxy only; real exported-device accuracy must be evaluated separately.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--sensitivity',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--maximum-ratio',type=float,default=1.25)
    a=p.parse_args();data=a.sensitivity.read_bytes();result=select(json.loads(data),a.maximum_ratio)
    result['calibration_sha256']=hashlib.sha256(data).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
