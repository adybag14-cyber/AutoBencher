#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge a calibrated precision map with an explicit no-demotion floor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def merge(calibration: dict, floor: dict) -> dict:
    original=calibration['selection']
    retained=floor['selection']
    merged=dict(original)
    for key in ('attention_int8_layers','mlp_int8_layers'):
        values=list(original[key])+list(retained[key])
        if any(type(x) is not int or x<0 or x>=42 for x in values):
            raise ValueError('MiniCPM5-2B layer indices must be integers in [0,41]')
        merged[key]=sorted(set(values))
        if not set(retained[key]).issubset(merged[key]): raise AssertionError('Precision floor was not preserved')
    for key in ('embedding_int8','lm_head_int8'):
        merged[key]=bool(original.get(key) or retained.get(key))
    merged['profile']=f"v3.2-context-aware-conservative-{calibration['context_capacity']}"
    result=dict(calibration)
    result['calibration_only_selection']=original
    result['selection']=merged
    result['precision_policy']={
        'strategy':'Union of calibrated-sensitive blocks and the frozen DynV31A INT8 floor; no floor block is demoted.',
        'floor_family':floor['family'],
        'floor_reference_artifact_sha256':floor['reference_artifact_sha256'],
        'mlp_int8_count':len(merged['mlp_int8_layers']),
        'additional_mlp_int8_layers':sorted(set(merged['mlp_int8_layers'])-set(retained['mlp_int8_layers'])),
        'quality_status':'Unvalidated candidate. More precision is not a substitute for end-to-end evaluation.',
        'evaluation_note':'Previously inspected GPQA/IFEval cases remain development regression tests, not fresh held-out validation.'}
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--floor',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=merge(json.loads(args.calibration.read_text(encoding='utf-8')),
                 json.loads(args.floor.read_text(encoding='utf-8')))
    result['input_sha256']={
        'calibration':hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        'floor':hashlib.sha256(args.floor.read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'out':str(args.out),'selection':result['selection'],'policy':result['precision_policy']}),flush=True)

if __name__=='__main__': main()
