#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run real AutoBencher diagnostics against all completed DynV31A LiteRT exports.

The historical BF16 controls use identical item selection and token ceilings.
This diagnostic is NOT a complete benchmark suite or a losslessness certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def write_json(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def wait_ready(url, process, timeout=180):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Server exited with code {process.returncode}')
        try:
            with urllib.request.urlopen(url,timeout=3) as response:
                if response.status==200: return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f'Server readiness timed out: {url}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tags',default='16k,32k,64k')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--out',type=Path,default=Path('results/minicpm5-dynv31-2026-09-11'))
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=(repo/args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    tags=args.tags.split(',')
    if any(t not in ('16k','32k','64k') for t in tags): raise ValueError('Unsupported context tag')
    root=Path(r'\\wsl.localhost\Ubuntu\home\tdamre\minicpm5-litert-20260910')
    source=json.loads((repo/'results/minicpm5-dynv3-2026-09-11/autobencher_dynamic_v3_results.json').read_text())
    profile=(repo/'autobencher-litert-dynv3.toml').read_text()
    profile=re.sub(r'^workspace\s*=.*$', "workspace = '"+(repo/'.autobencher').as_posix()+"'", profile, flags=re.M)
    binary=repo/'target/release/autobencher.exe'
    if not binary.is_file(): raise FileNotFoundError(binary)
    summary={'schema_version':1,'status':'running','family':'DynV31A','scope':'two-item matched regression diagnostic',
        'near_lossless_demonstrated':False,'variants':{},'bf16_baseline':source['bf16_baseline'],
        'protocol':{'seed':42,'temperature':0,'top_p':1,'thinking':False,'batch_size':1,
                    'limit':1,'token_caps':{'ifeval':512,'gpqa-diamond':2048},
                    'backend':'LiteRT-LM CPU','cpu_threads':32},
        'limitations':['Not full benchmark scores; one item per selected benchmark.',
          'BF16 controls are retained earlier runs with identical evaluator/item/settings.',
          'BF16 GPU latency and LiteRT CPU latency are not comparable.',
          'End-to-end differences can include conversion, tokenizer and runtime effects as well as weight quantization.']}
    if args.resume:
        previous=json.loads((out/'comparison.json').read_text(encoding='utf-8'))
        if previous.get('family')!='DynV31A' or previous.get('near_lossless_demonstrated') is not False:
            raise RuntimeError('Cannot resume a different candidate family or an untrusted comparison')
        summary=previous
        summary['status']='running'; summary.pop('error',None)
    elif (out/'comparison.json').exists():
        raise FileExistsError('Use --resume or choose a new output directory; existing evidence is not overwritten')
    summary['protocol']['request_timeout_seconds']=3600
    summary['protocol']['transport_note']='Resumed unfinished checks after the default client timeout aborted a slow CPU response. Existing completed scores are retained; prompts and output allowances are unchanged.'
    write_json(out/'comparison.json',summary)
    children=[]; logs=[]
    try:
        for name,command in [
          ('server',['wsl.exe','-d','Ubuntu','--','bash','/mnt/c/Users/adyba/docker-chatgpt-devbox/workspace/AutoBencher/scripts/serve_minicpm5_dynv31_frozen.sh']),
          ('proxy',[sys.executable,str(repo/'scripts/litert_refinement_proxy.py'),'--listen-port','9392','--backend','http://127.0.0.1:9391'])]:
            log=(out/f'{name}.log').open('w',encoding='utf-8'); logs.append(log)
            process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            children.append(process)
            wait_ready(f'http://127.0.0.1:{9391 if name=="server" else 9392}/v1/models',process)
        with urllib.request.urlopen('http://127.0.0.1:9392/v1/models',timeout=10) as response:
            registered=json.load(response)
        write_json(out/'served_models.json',registered)
        known={m['id'] for m in registered.get('data',[])}
        for tag in tags:
            alias=f'minicpm5-dynv31a-{tag}'
            if alias not in known: raise RuntimeError(f'Artifact was not imported: {alias}')
            artifact=root/f'benchmarks_dynv31/runtime/.litert-lm/models/minicpm5-dynv31a-{tag}/model.litertlm'
            floor=json.loads((repo/'config/minicpm5-dynv31a-precision-floor.json').read_text(encoding='utf-8'))
            expected=floor['reference_artifact_sha256'][tag]
            digest=hashlib.sha256()
            with artifact.open('rb') as f:
                while chunk:=f.read(8*1024*1024): digest.update(chunk)
            if digest.hexdigest()!=expected: raise RuntimeError(f'Artifact hash mismatch: {tag}')
            record=summary['variants'].setdefault(tag,{'artifact':f'MiniCPM5-2B-LiteRT-DynV31A-{tag}.litertlm','sha256':expected,'bytes':artifact.stat().st_size,'benchmarks':{}})
            if record['sha256']!=expected or record['bytes']!=artifact.stat().st_size:
                raise RuntimeError(f'Resumed artifact identity mismatch: {tag}')
            for bench,cap in [('ifeval',512),('gpqa-diamond',2048)]:
                retained=record['benchmarks'].get(bench,{})
                if args.resume and retained.get('status')=='completed' and retained.get('score_percent') is not None:
                    print(f'AUTOBENCHER_RETAIN {tag} {bench} run={retained["run_id"]}',flush=True)
                    continue
                benchout=out/tag/bench; benchout.mkdir(parents=True,exist_ok=True)
                generation={'max_tokens':cap,'temperature':0.0,'top_p':1.0,'timeout':3600,'retries':1,'retry_interval':0,
                            'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}}
                cfg=re.sub(r'^generation_config\s*=.*$',"generation_config = '"+json.dumps(generation,separators=(',',':'))+"'",profile,flags=re.M)
                config=benchout/'profile.toml'; config.write_text(cfg,encoding='utf-8')
                command=[str(binary),'--config',str(config),'run','--model',alias,'--engine','external',
                         '--api-base','http://127.0.0.1:9392','--only',bench,'--limit','1','--parallel','1','--no-setup','--no-card-check']
                before={p.name for p in (repo/'.autobencher/runs').iterdir() if p.is_dir()}
                print(f'AUTOBENCHER_START {tag} {bench} cap={cap}',flush=True)
                start=time.monotonic()
                with (benchout/'orchestrator.log').open('w',encoding='utf-8') as log:
                    result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=3900)
                candidates=[]
                for p in (repo/'.autobencher/runs').iterdir():
                    if p.name in before or not (p/'manifest.json').is_file(): continue
                    manifest=json.loads((p/'manifest.json').read_text())
                    if manifest['model']==alias and manifest['selected_benchmarks']==[bench]: candidates.append(p)
                if len(candidates)!=1: raise RuntimeError(f'Cannot unambiguously identify run for {tag}/{bench}: {candidates}')
                run=candidates[0]
                shutil.copytree(run,benchout/'run',dirs_exist_ok=True)
                rows=json.loads((run/'results.json').read_text())
                row=next(r for r in rows if r['benchmark_id']==bench)
                score=None
                if row['status']=='completed' and row.get('delta_from_reference') is not None:
                    score=row['reference_score']+row['delta_from_reference']
                bf=source['bf16_baseline']['benchmarks'][bench]
                old=source['variants'][tag]['benchmarks']['gpqa-diamond-2048' if bench=='gpqa-diamond' else bench]
                r={'run_id':run.name,'status':row['status'],'exit_code':result.returncode,
                   'score_percent':score,'bf16_score_percent':bf['primary_score_percent'],
                   'old_dynv3_score_percent':old['primary_score_percent'],
                   'delta_vs_bf16_pp':None if score is None else score-bf['primary_score_percent'],
                   'delta_vs_old_dynv3_pp':None if score is None else score-old['primary_score_percent'],
                   'elapsed_seconds':time.monotonic()-start,'max_tokens':cap,'metrics':row.get('metrics',{}),
                   'error':row.get('error')}
                record['benchmarks'][bench]=r
                write_json(out/'comparison.json',summary)
                print('AUTOBENCHER_RESULT',json.dumps({'tag':tag,'benchmark':bench,**r}),flush=True)
        summary['status']='completed'
    except Exception as exc:
        summary['status']='failed'; summary['error']=str(exc)
        raise
    finally:
        write_json(out/'comparison.json',summary)
        # These are this driver's processes, not any pre-existing model servers.
        pidfile=root/'benchmarks_dynv31/runtime/server-9391.pid'
        if children and pidfile.is_file():
            try:
                pid=int(pidfile.read_text().strip())
                if pid>1: subprocess.run(['wsl.exe','-d','Ubuntu','--','kill','-TERM',str(pid)],timeout=15,check=False)
            except Exception: pass
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill()
        for log in logs: log.close()
    print('MATRIX_DONE',out,flush=True)

if __name__=='__main__': main()
