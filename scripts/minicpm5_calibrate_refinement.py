#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-independent, context-aware calibration for MiniCPM5 LiteRT PTQ.

This is an Unsloth Dynamic-v3-inspired adaptation, not Unsloth's implementation.
No benchmark questions, answers, or existing evaluation outputs are used.
The tensor simulator follows AEQ OCTAV's 10 iterations, signed ranges, and
BF16->FP16 block-scale storage; final LiteRT evaluation remains mandatory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def qdq(weight: torch.Tensor, bits: int, block: int = 32) -> torch.Tensor:
    """AEQ-compatible symmetric OCTAV-4 / per-output-channel minmax-8 simulator.

    All-zero blocks are explicitly preserved rather than producing NaN/zero
    scales. Returned values are FP32; compute-dtype casting is a separate step.
    """
    w = weight.detach().float()
    if w.ndim != 2 or bits not in (4, 8):
        raise ValueError('Expected a matrix and 4 or 8 bits')
    if bits == 8:
        scale = w.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 127.0
        return (w / scale).round().clamp(-127, 127) * scale
    if w.shape[-1] % block:
        raise ValueError('Input dimension must be divisible by block size')
    x = w.reshape(w.shape[0], -1, block)
    guess = torch.ones_like(x[..., :1])
    alpha = 4.0 ** (-bits) / 3.0
    # AEQ can early-stop globally; ten iterations are the conservative upper bound.
    for _ in range(10):
        high, low = x >= guess, x <= -guess
        count = high.sum(-1, keepdim=True) + low.sum(-1, keepdim=True)
        new_guess = (torch.where(high, x, 0).sum(-1, keepdim=True)
                     - torch.where(low, x, 0).sum(-1, keepdim=True)) / (count * (1 - alpha) + alpha * block)
        done = torch.allclose(guess, new_guess, rtol=1e-5, atol=1e-8)
        guess = new_guess
        if done:
            break
    bound = torch.minimum(x.abs().amax(-1, keepdim=True).clamp_min(1e-9), guess)
    scale = (bound / 7).to(torch.bfloat16).to(torch.float16).float()
    # Use a representable scale for zero/underflowed blocks. Exact zero stays zero.
    scale = scale.clamp_min(2.0 ** -24)
    return ((x / scale).round().clamp(-8, 7) * scale).reshape_as(w)


def corpus(seed: int) -> list[dict]:
    """Original synthetic calibration dialogues, deliberately not benchmark data."""
    rng = random.Random(seed)
    rows = []
    for i in range(12):
        a, b, n = rng.randint(11, 89), rng.randint(4, 23), rng.randint(8, 30)
        cases = [
            ('math', f'A workshop has {a} boxes with {b} sensors each. It removes {n} sensors. Explain the count.',
             f'Initially there are {a} times {b}, or {a*b}, sensors. Removing {n} gives {a*b-n}. A reverse check adds {n} back to obtain {a*b}. The final count is {a*b-n} sensors.'),
            ('code', f'Write a Python function returning the first {n} squares, with input validation and tests.',
             'def squares(n: int) -> list[int]:\n    if isinstance(n, bool) or not isinstance(n, int) or n < 0:\n        raise ValueError("n must be a nonnegative integer")\n    return [i * i for i in range(n)]\n\nassert squares(0) == []\nassert squares(3) == [0, 1, 4]\nThe function takes O(n) time and output space; validation handles negative and noninteger input.'),
            ('agent', f'An API allows {b} requests per minute. Explain a safe plan for fetching {a} pages without losing progress.',
             'First confirm pagination and rate-limit headers. Keep a durable cursor after each completed page. Respect Retry-After, apply exponential backoff with jitter to transient failures, and use bounded retries. Never retry a non-idempotent operation blindly. Record failures separately from successful pages. Validate the final item count and avoid logging credentials.'),
            ('chat', f'Return JSON with item="sensor", quantity={a}, unit_price={b}, and the calculated total. Then explain the calculation.',
             json.dumps({'item':'sensor','quantity':a,'unit_price':b,'total':a*b}) + f'\nThe total is quantity multiplied by unit price: {a} × {b} = {a*b}.'),
            ('multilingual', f'用中文和西班牙语解释：{a} 个盒子，每盒 {b} 件，一共有多少件？',
             f'中文：共有 {a} × {b} = {a*b} 件。乘法把相同数量的组相加。\nEspañol: Hay {a} cajas de {b} unidades cada una; por tanto, hay {a*b} unidades. La multiplicación representa la suma de grupos iguales.'),
            ('reasoning', f'A service must use less than {a} MB of memory and respond in {b} milliseconds. Explain the limits of choosing a design without measurements.',
             'Requirements do not by themselves establish feasibility. Measure representative requests, peak memory, and tail latency on the target device. Distinguish cached and uncached operation, include concurrency, and report the workload and sample count. A smaller data structure may save memory but increase lookup cost. Prefer a prototype and explicit failure thresholds over an unsupported guarantee.'),
        ]
        for category, question, answer in cases:
            rows.append({'id':f'cal-{i:02d}-{category}', 'category':category,
                         'messages':[{'role':'user','content':question},{'role':'assistant','content':answer}]})
    return rows


def long_document(tokens: int, tok, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    records = []
    # Fictional procedural records, not any NoLiMa corpus or evaluation needle.
    for i in range(max(400, tokens // 14)):
        k = rng.randrange(10000,99999)
        records.append(f'Record {i:05d}: station S{k} stores {rng.randrange(1,99)} units of alloy; inspection cycle {i%17} requires temperature verification. Maintenance notes distinguish available stock from pending deliveries.\n')
    messages = [{'role':'user','content':'Review these fictional maintenance records. Summarize inventory and audit considerations.\n' + ''.join(records)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
    if ids.shape[1] < tokens:
        raise RuntimeError('Insufficient long calibration text')
    return ids[:, :tokens]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=730219)
    ap.add_argument('--long-lengths', default='12288,24576,40960')
    ap.add_argument('--mlp-int8-count', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
               attn_implementation='sdpa', local_files_only=True).to(args.device).eval()
    layers = model.model.layers
    modules = {}
    for i, layer in enumerate(layers):
        for part, parent, names in [('attn', layer.self_attn, ('q_proj','k_proj','v_proj','o_proj')),
                                    ('mlp',layer.mlp,('gate_proj','up_proj','down_proj'))]:
            for name in names:
                modules[f'layer.{i}.{part}.{name}'] = getattr(parent, name)
    stats = {name: {'sum':torch.zeros(mod.in_features, device=args.device), 'count':0}
             for name, mod in modules.items()}
    handles = []
    def hook(name):
        def capture(mod, inputs):
            flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            # Bound temporary memory even for the 40K document and MLP inputs.
            for chunk in flat.split(512):
                stats[name]['sum'].add_(chunk.float().square().sum(0))
            stats[name]['count'] += flat.shape[0]
        return capture
    for name, mod in modules.items():
        handles.append(mod.register_forward_pre_hook(hook(name)))
    rows = corpus(args.seed)
    save_json(args.out/'calibration_corpus.json', rows)
    corpus_sha = hashlib.sha256((args.out/'calibration_corpus.json').read_bytes()).hexdigest()
    start = time.monotonic()
    token_counts = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            text = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
            ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)
            model.model(input_ids=ids, use_cache=False)
            token_counts.append(ids.shape[1])
            if (index+1)%12 == 0:
                print(f'CALIBRATION {index+1}/{len(rows)} tokens={sum(token_counts)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    short_stats = {n:(s['sum'].detach().cpu().clone()/s['count']) for n,s in stats.items()}
    torch.save(short_stats,args.out/'short_second_moments.pt')
    contexts = {12288:16384,24576:32768,40960:65536}
    all_results = []
    with torch.inference_mode():
        for long_len in [int(v) for v in args.long_lengths.split(',')]:
            context = contexts.get(long_len)
            if context is None:
                raise ValueError('Expected 12288, 24576, or 40960 long-token sample')
            for s in stats.values():
                s['sum'].zero_(); s['count']=0
            ids = long_document(long_len, tok, args.seed+context).to(args.device)
            print(f'LONG_FORWARD {long_len} tokens', flush=True)
            model.model(input_ids=ids, use_cache=False)
            del ids
            # Equal weight for short diverse dialogues and this context's long text;
            # long documents do not overwhelm all other categories by token count.
            moments = {n:(short_stats[n].to(args.device)*0.5 + s['sum']/s['count']*0.5)
                       for n,s in stats.items()}
            module_rows=[]
            for index,(name,mod) in enumerate(modules.items()):
                second = moments[name][None,:]
                errs=[0.0,0.0]; signal=0.0
                for w in mod.weight.split(128):
                    w=w.float()
                    signal += (w.square()*second).sum().item()
                    for j,bits in enumerate((4,8)):
                        errs[j] += ((w-qdq(w,bits)).square()*second).sum().item()
                layer=int(name.split('.')[1]); group=name.split('.')[2]
                module_rows.append({'name':name,'layer':layer,'group':group,'params':mod.weight.numel(),
                         'int4_err':errs[0], 'int8_err':errs[1], 'signal':signal,
                         'relative_gain':max(0.,errs[0]-errs[1])/max(signal,1e-30)})
                if (index+1)%49==0:
                    print(f'SENSITIVITY context={context} {index+1}/{len(modules)}', flush=True)
            groups=[]
            for layer in range(len(layers)):
                for group in ('attn','mlp'):
                    rs=[r for r in module_rows if r['layer']==layer and r['group']==group]
                    e4=sum(r['int4_err'] for r in rs); e8=sum(r['int8_err'] for r in rs)
                    signal=sum(r['signal'] for r in rs)
                    groups.append({'layer':layer,'group':group,'int4_err':e4,'int8_err':e8,'signal':signal,
                                   'relative_gain':max(0.,e4-e8)/max(signal,1e-30)})
            ranking=sorted((g for g in groups if g['group']=='mlp'),key=lambda g:(-g['relative_gain'],g['layer']))
            # Keep both boundary blocks protected, then spend the remaining fixed-size budget.
            chosen={0,len(layers)-1}
            for g in ranking:
                if len(chosen)>=args.mlp_int8_count: break
                chosen.add(g['layer'])
            result={'schema_version':1,'method':'Dynamic-v3-inspired context-aware LiteRT PTQ; not official Unsloth',
                'calibration':{'source':'original synthetic dialogues and fictional maintenance records',
                  'corpus_sha256':corpus_sha,'seed':args.seed,'dialogues':len(rows),
                  'short_tokens':sum(token_counts),'long_tokens':long_len,
                  'short_long_weights':[0.5,0.5],'no_benchmark_data':True},
                'context_capacity':context,
                'selection':{'attention_int8_layers':list(range(len(layers))), 'mlp_int8_layers':sorted(chosen),
                     'lm_head_int8':True,'embedding_int8':True,'default':'block32 INT4 + OCTAV',
                     'profile':f'v3.2-context-aware-{context}'},
                'simulator':'OCTAV10; int4 [-8,7]; BF16->FP16 scales; int8 narrow [-127,127]',
                'limitations':['Sensitivity is a diagonal activation-weighted local-error proxy, not end-to-end quality.',
                  'Synthetic calibration is limited; candidate acceptance requires held-out BF16-relative AutoBencher tests.',
                  'The three precision maps require independent exports; context metadata alone is insufficient.'],
                'groups':groups,'modules':module_rows}
            save_json(args.out/f'sensitivity_{context}.json',result)
            all_results.append({'context':context,'mlp_int8_layers':sorted(chosen)})
            print('CONTEXT_COMPLETE',json.dumps(all_results[-1]),flush=True)
            del moments
    for h in handles: h.remove()
    save_json(args.out/'summary.json',{'status':'calibration_completed','contexts':all_results,
                        'elapsed_seconds':time.monotonic()-start,'corpus_sha256':corpus_sha})
    print('CALIBRATION_DONE',flush=True)

if __name__=='__main__':
    main()
