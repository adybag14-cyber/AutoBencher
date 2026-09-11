#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibrate a LiteRT mixed INT4/INT8 candidate without training or benchmark labels.

Uses the installed AI Edge Quantizer's real OCTAV/block32 quantization, including
its rounded scales, rather than a min/max error proxy. Writes auditable input
hashes and a layer map understood by the existing MiniCPM5 exporter. This is an
independent Dynamic-v3-inspired LiteRT adaptation, not an official Unsloth quant.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import octav, naive_min_max_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt


def dump(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def quantize_weight(weight: np.ndarray, bits: int, operation=qtyping.TFLOperationName.FULLY_CONNECTED) -> np.ndarray:
    gran = qtyping.QuantGranularity.BLOCKWISE_32 if bits == 4 else qtyping.QuantGranularity.CHANNELWISE
    cfg = qtyping.TensorQuantizationConfig(num_bits=bits, symmetric=True, granularity=gran)
    op = SimpleNamespace(op=None, op_name=operation,
        subgraph_op_index=0, op_quant_config=SimpleNamespace(weight_tensor_config=cfg))
    algorithm = octav if bits == 4 else naive_min_max_quantize
    with np.errstate(divide='ignore', invalid='ignore'):
        qp = algorithm.get_tensor_quant_params(op, cfg, tensor_content=weight)
    # Known exporter zero-row issue: a truly all-zero block has exactly zero
    # reconstructed weights. Nonzero blocks must never have invalid scales.
    dq = uqt.uniform_dequantize(qp.quantized_data, qp).astype(np.float32)
    bad = ~np.isfinite(dq)
    if bad.any():
        if np.any(weight[bad] != 0):
            raise ValueError('Nonfinite quantization reconstruction on nonzero weights')
        dq[bad] = 0
    return dq


def load_prompts(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'PROMPTS' for x in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, list) and len(value) >= 16 and all(isinstance(x,str) for x in value):
                return value
    raise ValueError('Expected a literal PROMPTS list; source is parsed, never executed')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompt-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--response-tokens', type=int, default=96)
    ap.add_argument('--mlp-int8', type=int, default=20)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this calibration workload')
    prompts = load_prompts(args.prompt_source)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    total_layers = len(model.model.layers)
    if not 2 <= args.mlp_int8 <= total_layers:
        raise ValueError('Invalid MLP INT8 quota')
    records = []
    start = time.monotonic()
    # Generate actual assistant text, not an instruction pretending to be a trace.
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            thinking = i % 4 == 0
            ids = tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=thinking, return_tensors='pt', return_dict=False).to('cuda')
            out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                max_new_tokens=args.response_tokens, use_cache=True, pad_token_id=tok.eos_token_id)
            records.append({'id':f'chat-{i:03d}', 'kind':'chat_response', 'thinking':thinking,
                'token_ids':out[0].cpu().tolist(), 'text':tok.decode(out[0], skip_special_tokens=False)})
            if (i+1)%8 == 0: print(f'GENERATED {i+1}/{len(prompts)} {time.monotonic()-start:.1f}s',flush=True)
    # Benchmark-independent long documents: IDs, code/config, multilingual prose,
    # and cross-references. Never ingest the GPQA/NoLiMa regression cases.
    paragraph = ('Project memorandum: component Vega uses schema revision 17. '
        'The routing table maps cobalt to bucket 37, amber to 64, and silver to 91. '
        'Rust: fn checked_add(a: u32,b: u32)->Option<u32>{a.checked_add(b)}. '
        '实验记录必须注明日期、对照组和不确定性。 '
        'La documentacion distingue resultados observados de hipotesis. '
        'A cache invalidation event does not imply permanent storage failure.\n')
    for length in (2048,4096,8192,12288):
        text = '\n'.join(f'Section {i:04d}. '+paragraph for i in range(300))
        prefix = tok.apply_chat_template([{'role':'user','content':text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids = prefix[:length-1] + [prefix[-1]]
        records.append({'id':f'document-{length}', 'kind':'long_document', 'thinking':False,
            'token_ids':ids, 'text':tok.decode(ids,skip_special_tokens=False)})
    corpus = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    (args.output/'calibration.jsonl').write_text(corpus,encoding='utf-8')
    modules = {}
    for i, layer in enumerate(model.model.layers):
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            modules[f'layer.{i}.attn.{name}'] = getattr(layer.self_attn,name)
        for name in ('gate_proj','up_proj','down_proj'):
            modules[f'layer.{i}.mlp.{name}'] = getattr(layer.mlp,name)
    # Sample equally from each sequence, retaining cross-channel covariance for
    # direct projection-error measurement and mean second moments for auditing.
    moments = {k:torch.zeros(m.in_features,device='cuda',dtype=torch.float32) for k,m in modules.items()}
    samples = {k:[] for k in modules}
    counts = {k:0 for k in modules}
    handles = []
    def hook(name):
        def collect(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            positions = torch.linspace(0,x.shape[0]-1,min(16,x.shape[0]),device=x.device).long()
            chosen=x.index_select(0,positions)
            moments[name].add_(chosen.float().square().mean(0))
            samples[name].append(chosen.cpu())
            counts[name]+=1
        return collect
    for name,mod in modules.items(): handles.append(mod.register_forward_pre_hook(hook(name)))
    with torch.inference_mode():
        for i,record in enumerate(records):
            ids = torch.tensor([record['token_ids']],device='cuda')
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids),use_cache=False)
            if (i+1)%8 == 0: print(f'CALIBRATED {i+1}/{len(records)}',flush=True)
    for handle in handles: handle.remove()
    rows=[]
    with torch.inference_mode():
        for j,(name,mod) in enumerate(modules.items()):
            w=mod.weight.detach().float().cpu().numpy()
            x=torch.cat(samples.pop(name)).to('cuda',dtype=torch.float32)
            # Bound projection scratch to 128 evenly-spaced calibration tokens.
            x=x[torch.linspace(0,len(x)-1,min(128,len(x)),device='cuda').long()]
            wf=mod.weight.detach().float()
            signal=(x @ wf.T).square().mean().item()
            errors={}
            for bits in (4,8):
                dq=torch.from_numpy(quantize_weight(w,bits)).to('cuda')
                errors[bits]=(x @ (wf-dq).T).square().mean().item()
                del dq
            relative4=errors[4]/max(signal,1e-20); relative8=errors[8]/max(signal,1e-20)
            rows.append({'name':name,'params':mod.weight.numel(),'sample_count':len(x),
                'sequence_count':counts[name],'projection_signal':signal,
                'octav4_projection_mse':errors[4],'int8_projection_mse':errors[8],
                'relative4':relative4,'relative8':relative8,
                'relative_gain':max(0,relative4-relative8),
                'activation_second_moment_mean':float((moments[name]/counts[name]).mean())})
            del x,wf,w
            if (j+1)%28==0: print(f'QUANT_SENSITIVITY {j+1}/{len(modules)}',flush=True)
    groups=[]
    for i in range(total_layers):
        rs=[r for r in rows if r['name'].startswith(f'layer.{i}.mlp.')]
        # Relative error avoids the old preference for layers merely having
        # larger residual-stream magnitudes. Protect full MLP groups for exporter
        # compatibility; projection-level data remains available for later work.
        groups.append({'layer':i,'relative_gain':sum(r['relative_gain'] for r in rs)/len(rs),
            'params':sum(r['params'] for r in rs)})
    ranking=sorted(groups,key=lambda g:(-g['relative_gain'],g['layer']))
    selected={0,total_layers-1}
    for row in ranking:
        if len(selected)>=args.mlp_int8: break
        selected.add(row['layer'])
    result={'method':'Independent Dynamic-v3-inspired LiteRT OCTAV-aware response/long-document calibration; no QAT/QAD',
        'created_unix':time.time(),'seed':args.seed,'calibration_sha256':hashlib.sha256(corpus.encode()).hexdigest(),
        'calibration_sequence_count':len(records),'calibration_token_count':sum(len(r['token_ids']) for r in records),
        'long_document_token_counts':[len(r['token_ids']) for r in records if r['kind']=='long_document'],
        'scope':'Calibration includes up to 12288 contiguous tokens. 32K/64K execution and fidelity remain separate validation requirements.',
        'selection':{'attention_int8_layers':list(range(total_layers)),'mlp_int8_layers':sorted(selected),
            'lm_head_int8':True,'embedding_int8':True,'default':'block32 INT4 + OCTAV',
            'profile':'v3.2-response-longdoc-octav-aware'},'mlp_ranking':ranking,'modules':rows}
    dump(args.output/'sensitivity.json',result)
    print('SELECTION',json.dumps(result['selection']),flush=True)
    print('CALIBRATION_DONE',result['calibration_token_count'],flush=True)

if __name__=='__main__': main()
