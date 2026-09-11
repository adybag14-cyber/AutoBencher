#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the actual Rust AutoBencher against one local LiteRT candidate in WSL.

Owns its server/proxy children, uses a unique model alias and port, hashes the
artifact, probes inference before benchmarking, and preserves all run evidence.
Does not upload artifacts or alter existing model aliases.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, socket, subprocess, sys, time, urllib.request
from pathlib import Path


def request(url: str, payload=None, timeout=30):
    req=urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=15)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--artifact',type=Path,required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--context',type=int,required=True)
    ap.add_argument('--port',type=int,default=9461)
    ap.add_argument('--only',default='gpqa-diamond,ifeval')
    ap.add_argument('--limit',type=int,default=1)
    ap.add_argument('--max-tokens',type=int,default=2048)
    ap.add_argument('--threads',type=int,default=32)
    args=ap.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]+',args.label): raise ValueError('Unsafe model alias')
    artifact=args.artifact.resolve(strict=True)
    repo=Path(__file__).resolve().parent.parent
    exe=repo/'target/release/autobencher.exe'
    if not exe.is_file(): raise FileNotFoundError(exe)
    run=repo/'.autobencher/candidates'/f'{args.label}-{time.time_ns()}'
    run.mkdir(parents=True)
    # A separate alias prevents the endpoint from accidentally benchmarking a
    # previously loaded artifact. Existing aliases are accepted only if identical.
    alias=Path.home()/'.litert-lm/models'/args.label/'model.litertlm'
    alias.parent.mkdir(parents=True,exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if alias.resolve()!=artifact: raise RuntimeError('Existing alias points to another artifact')
    else: alias.symlink_to(artifact)
    for port in (args.port,args.port+1):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',port))
    server_config=run/'server.json'
    server_config.write_text(json.dumps({'default':{'backend':'cpu','cpu_thread_count':args.threads,
        'cache':'disk','temperature':0.0,'top_p':1.0,'seed':42,'thinking':False},
        'models':{args.label:{'max_num_tokens':args.context}}},indent=2))
    config=run/'autobencher.toml'
    base=(repo/'autobencher-litert-dynv3.toml').read_text()
    workspace_win=subprocess.check_output(['wslpath','-w',str(repo/'.autobencher')],text=True).strip()
    base=re.sub(r'^workspace = .*$',lambda _: 'workspace = '+json.dumps(workspace_win),base,flags=re.M)
    base=re.sub(r'^model = .*$',f'model = "{args.label}"',base,flags=re.M)
    base=re.sub(r'^api_base = .*$',f'api_base = "http://127.0.0.1:{args.port+1}"',base,flags=re.M)
    base=re.sub(r'^generation_config = .*$',"generation_config = '"+json.dumps(
        {'max_tokens':args.max_tokens,'temperature':0.0,'top_p':1.0,'seed':42,'timeout':3600,'retries':1,
         'extra_body':{'reasoning_effort':'none'}})+"'",base,flags=re.M)
    config.write_text(base)
    hasher=hashlib.sha256()
    with artifact.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): hasher.update(chunk)
    manifest={'artifact':str(artifact),'sha256':hasher.hexdigest(),'bytes':artifact.stat().st_size,
        'label':args.label,'context':args.context,'threads':args.threads,'backend':'cpu',
        'sample_limit':args.limit,'benchmarks':args.only.split(','),'max_tokens':args.max_tokens,
        'started_unix':time.time(),'status':'starting','evidence_directory':str(run)}
    def save(): (run/'candidate.json').write_text(json.dumps(manifest,indent=2))
    save(); print('CANDIDATE_EVIDENCE',run,flush=True)
    children=[]
    try:
        with (run/'server.log').open('wb') as server_log,(run/'proxy.log').open('wb') as proxy_log:
            server=subprocess.Popen([str(Path(sys.executable).parent/'litert-lm'),'serve',
                '--config',str(server_config),'--host','127.0.0.1','--port',str(args.port)],
                stdout=server_log,stderr=subprocess.STDOUT)
            children.append(server)
            proxy=subprocess.Popen([sys.executable,str(repo/'scripts/litert_openai_proxy.py'),
                '--listen-port',str(args.port+1),'--backend',f'http://127.0.0.1:{args.port}'],
                stdout=proxy_log,stderr=subprocess.STDOUT)
            children.append(proxy)
            url=f'http://127.0.0.1:{args.port+1}'
            for _ in range(120):
                if any(c.poll() is not None for c in children): raise RuntimeError('Serving child exited; inspect logs')
                try:
                    models=request(url+'/v1/models')
                    if args.label not in [x.get('id') for x in models.get('data',[])]:
                        raise RuntimeError('Requested alias absent from endpoint')
                    break
                except (OSError,RuntimeError): time.sleep(1)
            else: raise TimeoutError('Endpoint did not become ready')
            (run/'endpoint-models.json').write_text(json.dumps(models,indent=2))
            smoke=request(url+'/v1/chat/completions',{'model':args.label,
                'messages':[{'role':'user','content':'What is 2 + 2? Reply with just the number.'}],
                'max_tokens':32,'temperature':0,'top_p':1,'reasoning_effort':'none'},timeout=600)
            (run/'smoke.json').write_text(json.dumps(smoke,indent=2))
            if not smoke.get('choices'): raise RuntimeError('No smoke completion')
            print('SMOKE',json.dumps(smoke['choices']),flush=True)
            manifest['status']='benchmarking'; save()
            # Windows interop inherits the mapped repository working directory.
            command=[str(exe),'--config',str(config.relative_to(repo)).replace('/','\\'),'run',
                '--only',args.only,'--limit',str(args.limit),'--no-setup','--no-card-check','--strict']
            print('AUTOBENCHER_START',args.label,args.only,flush=True)
            with (run/'autobencher.log').open('wb') as log:
                result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
            output=(run/'autobencher.log').read_text(errors='replace')
            ids=re.findall(r'Run ([0-9a-f-]{36}) complete',output)
            manifest.update({'status':'completed' if result.returncode==0 else 'failed',
                'autobencher_exit_code':result.returncode,'autobencher_run_ids':ids,'finished_unix':time.time()})
            save(); print(output[-12000:],flush=True)
            if result.returncode: raise RuntimeError(f'AutoBencher failed: {result.returncode}')
    except Exception as error:
        manifest.update(status='failed',error=str(error),finished_unix=time.time());save();raise
    finally:
        for child in reversed(children): stop(child)
    print('CANDIDATE_BENCH_DONE',args.label,flush=True)

if __name__=='__main__': main()
