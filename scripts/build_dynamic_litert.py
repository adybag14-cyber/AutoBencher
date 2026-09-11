#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build independently named OCTAV-aware LiteRT refinement candidates.

Never lowers a previously frozen INT8 precision floor. Selection is based only
on calibration; benchmark outputs are not inputs. Does not overwrite exports.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.metadata, json, os, subprocess, sys, time
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def select_refinement(calibration,floor,mlp_count):
    result=copy.deepcopy(calibration)
    base=floor['selection']; current=result['selection']
    n=len(current['attention_int8_layers'])
    protected=set(base['mlp_int8_layers'])|{0,n-1}
    if mlp_count<len(protected) or mlp_count>n: raise ValueError('Quota conflicts with frozen precision floor')
    ranking=sorted(result['mlp_ranking'],key=lambda row:(-row['relative_gain'],row['layer']))
    for row in ranking:
        if len(protected)>=mlp_count: break
        protected.add(row['layer'])
    if len(protected)!=mlp_count: raise ValueError('Incomplete calibration ranking')
    current.update(attention_int8_layers=list(range(n)),mlp_int8_layers=sorted(protected),
        profile='v3.2-octav-response-longdoc-floor-preserving',embedding_int8=True,lm_head_int8=True)
    result['precision_floor']={'family':floor['family'],'selection':base,
        'additional_mlp_int8_layers':sorted(protected-set(base['mlp_int8_layers']))}
    result['selection_policy']='Frozen prior INT8 floor plus calibration-ranked additions; no evaluation-answer tuning'
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--sensitivity',type=Path,required=True)
    ap.add_argument('--floor',type=Path,required=True)
    ap.add_argument('--exporter',type=Path,required=True)
    ap.add_argument('--zero-scale-fixer',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--contexts',default='16384,32768,65536')
    ap.add_argument('--mlp-int8',type=int,default=28)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    contexts=[int(x) for x in args.contexts.split(',')]
    tags={16384:'16k',32768:'32k',65536:'64k'}
    if not contexts or any(c not in tags for c in contexts) or len(set(contexts))!=len(contexts):
        raise ValueError('Contexts must be unique members of 16384,32768,65536')
    for path in (args.model,args.sensitivity,args.floor,args.exporter,args.zero_scale_fixer,args.source_root):
        if not path.exists(): raise FileNotFoundError(path)
    selection=select_refinement(json.loads(args.sensitivity.read_text()),json.loads(args.floor.read_text()),args.mlp_int8)
    args.output.mkdir(parents=True,exist_ok=True)
    selection_path=args.output/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
        raise RuntimeError('Existing output belongs to a different selection')
    atomic_json(selection_path,selection)
    fingerprints={'exporter_sha256':sha(args.exporter),'zero_scale_fixer_sha256':sha(args.zero_scale_fixer),
        'sensitivity_sha256':sha(args.sensitivity),'selection_sha256':sha(selection_path),
        'model_config_sha256':sha(args.model/'config.json'),
        'model_weight_sha256':{p.name:sha(p) for p in sorted(args.model.glob('*.safetensors'))},
        'toolchain':{name:importlib.metadata.version(name) for name in
            ('torch','transformers','ai-edge-quantizer-nightly','litert-torch','litert-lm')}}
    if not fingerprints['model_weight_sha256']:
        raise FileNotFoundError('No safetensors weights in the supplied model directory')
    manifest_path=args.output/'build-manifest.json'
    manifest={'family':'DynV32O28','status':'building','started_unix':time.time(),
        'method':'Independent Unsloth Dynamic-v3-inspired LiteRT adaptation, not an official Unsloth quant',
        'validation_status':'not_validated','near_lossless_demonstrated':False,
        'fingerprints':fingerprints,'selection':selection['selection'],'artifacts':{}}
    if manifest_path.exists():
        if not args.resume: raise FileExistsError('Use --resume for an existing manifest')
        previous=json.loads(manifest_path.read_text())
        if previous['fingerprints']!=fingerprints: raise RuntimeError('Cannot resume changed inputs')
        manifest=previous;manifest['status']='building'
    atomic_json(manifest_path,manifest)
    for context in contexts:
        tag=tags[context];out=args.output/tag;raw=out/'raw'
        destination=out/f'MiniCPM5-2B-LiteRT-DynV32O28-{tag}.litertlm'
        previous=manifest['artifacts'].get(tag)
        if previous:
            if destination.stat().st_size!=previous['bytes'] or sha(destination)!=previous['sha256']:
                raise RuntimeError('Completed artifact identity changed')
            print('BUILD_RETAIN',tag,flush=True);continue
        if raw.exists() or destination.exists():
            raise FileExistsError(f'Untracked export already exists: {out}; choose a new output root')
        out.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env.update(CACHE=str(context),PREFILL='1024,256,64,16,4,1',EXTERNALIZE_EMBEDDER='1',USE_JINJA='1',
            DYNV3_SENSITIVITY=str(selection_path.resolve()),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        print('BUILD_START',tag,selection['selection'],flush=True)
        started=time.monotonic()
        with (out/'export.log').open('wb') as log:
            subprocess.run([sys.executable,str(args.exporter),str(args.model),str(raw),
                str(args.model/'chat_template.jinja'),'MINICPM_DYNV3'],cwd=args.source_root,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        with (out/'zero-scale-fix.log').open('wb') as log:
            subprocess.run([sys.executable,str(args.zero_scale_fixer),str(raw/'model.litertlm'),str(destination)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        entry={'path':str(destination),'context_capacity':context,'bytes':destination.stat().st_size,
            'sha256':sha(destination),'build_seconds':time.monotonic()-started,
            'runtime_validation':'pending','benchmark_validation':'pending'}
        manifest['artifacts'][tag]=entry;atomic_json(manifest_path,manifest)
        print('BUILD_DONE',tag,json.dumps(entry),flush=True)
    manifest.update(status='built',finished_unix=time.time());atomic_json(manifest_path,manifest)
    print('ALL_BUILDS_DONE',manifest_path,flush=True)

if __name__=='__main__': main()
