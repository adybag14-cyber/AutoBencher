#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Held-out BF16-relative logit and greedy-trajectory diagnostics for precision maps.

Dequantized weights run in PyTorch BF16: this is NOT exported LiteRT inference,
benchmark accuracy, official Divergence-300, or long-context capacity validation.
Neither these prompts nor their outputs are inputs to calibration or selection.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from calibrate_dynamic_litert import quantize_weight
from ai_edge_quantizer import qtyping

PROMPTS=[
    'Explain why subtracting two almost equal floating-point numbers can lose precision.',
    'A train covers 135 kilometres in 90 minutes. Calculate its average speed in kilometres per hour.',
    'Write a Rust function returning the larger of two signed integers without allocation.',
    'Explain the difference between a database transaction rollback and a backup restore.',
    'Give an example of a many-to-one mapping that is not invertible.',
    'Explain how a control group helps distinguish an intervention from background changes.',
    'Provide a JSON object describing a book with title, author, and publication_year fields.',
    'Identify the logical error in: All squares are rectangles, so all rectangles are squares.',
    'Translate into French: The experiment was repeated under identical conditions.',
    '用中文解释为什么海水结冰后盐分会发生变化。',
    'Explica por que una muestra pequena puede producir una estimacion imprecisa.',
    'Describe how a checksum differs from encryption.',
    'A list contains repeated elements. Describe a stable deduplication algorithm.',
    'Compare breadth-first search and depth-first search for finding an unweighted shortest path.',
    'Explain why a solar eclipse does not happen at every new moon.',
    'Give two distinct reasons a program may deadlock despite using locks correctly in isolation.',
]


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--profile',action='append',required=True,help='LABEL=selection.json')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'surrogate-results.json').exists():
        raise FileExistsError('Choose a new output directory; prior validation evidence is not overwritten')
    profiles={};profile_hashes={}
    for spec in args.profile:
        label,path=spec.split('=',1)
        if label in profiles: raise ValueError('Duplicate profile label')
        payload=Path(path).read_bytes();profiles[label]=json.loads(payload)['selection']
        profile_hashes[label]=hashlib.sha256(payload).hexdigest()
    torch.manual_seed(90210);torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=True).to('cuda').eval()
    records=[];reference=[]
    with torch.inference_mode():
        for i,prompt in enumerate(PROMPTS):
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False).to('cuda')
            output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=32,
                pad_token_id=tok.eos_token_id,use_cache=True)
            # Distributions predict the first 16 teacher tokens; no target token
            # is fed at its own prediction position.
            length=min(16,output.shape[1]-ids.shape[1])
            if length<1: raise RuntimeError('No baseline continuation')
            inp=output[:,:ids.shape[1]+length-1]
            h=model.model(input_ids=inp,use_cache=False).last_hidden_state[:,-length:]
            logits=model.lm_head(h).float()
            reference.append(logits.log_softmax(-1).cpu())
            records.append({'id':f'heldout-{i:02d}','prompt':prompt,'token_ids':inp[0].cpu().tolist(),
                'prompt_token_ids':ids[0].cpu().tolist(),
                'reference_generated_ids':output[0,ids.shape[1]:].cpu().tolist(),
                'scored_positions':length,'bf16_continuation':tok.decode(output[0,ids.shape[1]:])})
            print('BF16_HOLDOUT',i+1,len(PROMPTS),flush=True)
    corpus=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records)
    (args.output/'heldout.jsonl').write_text(corpus,encoding='utf-8')
    modules={}
    for i,layer in enumerate(model.model.layers):
        for sub in ('q_proj','k_proj','v_proj','o_proj'): modules[f'layer.{i}.attn.{sub}']=getattr(layer.self_attn,sub)
        for sub in ('gate_proj','up_proj','down_proj'): modules[f'layer.{i}.mlp.{sub}']=getattr(layer.mlp,sub)
    modules['embedding']=model.model.embed_tokens;modules['lm_head']=model.lm_head
    originals={name:module.weight.detach().cpu().clone() for name,module in modules.items()}
    cache={}
    def bits_for(name,selection):
        if name in ('embedding','lm_head'): return 8
        _,layer,kind,_=name.split('.')
        key='attention_int8_layers' if kind=='attn' else 'mlp_int8_layers'
        return 8 if int(layer) in selection[key] else 4
    with torch.inference_mode():
        for j,(name,module) in enumerate(modules.items()):
            needed={bits_for(name,s) for s in profiles.values()}
            w=originals[name].float().numpy()
            op=qtyping.TFLOperationName.EMBEDDING_LOOKUP if name=='embedding' else qtyping.TFLOperationName.FULLY_CONNECTED
            for bits in needed:
                cache[name,bits]=torch.from_numpy(quantize_weight(w,bits,operation=op)).to(dtype=torch.bfloat16)
            if (j+1)%28==0: print('SURROGATE_WEIGHTS',j+1,len(modules),flush=True)
    report={'schema_version':1,'kind':'pytorch_bf16_dequantized_weight_surrogate',
        'scope':'16 held-out prompts; up to 16 teacher-forced positions and 32 greedy generated tokens each; no LiteRT runtime claim',
        'corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest(),'prompt_count':len(records),
        'profile_sha256':profile_hashes,'seed':90210,'thinking':False,'do_sample':False,
        'toolchain':{n:importlib.metadata.version(n) for n in ('torch','transformers','ai-edge-quantizer-nightly')},
        'near_lossless_demonstrated':False,'profiles':{},'created_unix':time.time()}
    with torch.inference_mode():
        for label,selection in profiles.items():
            for name,module in modules.items(): module.weight.copy_(cache[name,bits_for(name,selection)])
            rows=[]
            for record,ref in zip(records,reference):
                ids=torch.tensor([record['token_ids']],device='cuda')
                h=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-record['scored_positions']:]
                candidate=model.lm_head(h).float().log_softmax(-1)
                ref=ref.to('cuda');kl=(ref.exp()*(ref-candidate)).sum(-1)
                prompt_ids=torch.tensor([record['prompt_token_ids']],device='cuda')
                generated=model.generate(prompt_ids,attention_mask=torch.ones_like(prompt_ids),do_sample=False,
                    max_new_tokens=32,pad_token_id=tok.eos_token_id,use_cache=True)
                candidate_ids=generated[0,prompt_ids.shape[1]:].cpu().tolist()
                reference_ids=record['reference_generated_ids']
                shared_prefix=0
                for a,b in zip(reference_ids,candidate_ids):
                    if a!=b: break
                    shared_prefix+=1
                rows.append({'id':record['id'],'positions':kl.numel(),'kl_sum':float(kl.sum()),
                    'mean_kl_nats':float(kl.mean()),'argmax_matches':int((ref.argmax(-1)==candidate.argmax(-1)).sum()),
                    'greedy_exact_trajectory_match':reference_ids==candidate_ids,
                    'greedy_shared_prefix_tokens':shared_prefix,
                    'greedy_token_matches':sum(a==b for a,b in zip(reference_ids,candidate_ids)),
                    'greedy_comparison_positions':max(len(reference_ids),len(candidate_ids)),
                    'greedy_generated_ids':candidate_ids})
            n=sum(r['positions'] for r in rows)
            trajectory_n=sum(r['greedy_comparison_positions'] for r in rows)
            result={'positions':n,'mean_kl_nats':sum(r['kl_sum'] for r in rows)/n,
                'argmax_agreement':sum(r['argmax_matches'] for r in rows)/n,
                'greedy_exact_trajectory_fraction':sum(r['greedy_exact_trajectory_match'] for r in rows)/len(rows),
                'greedy_token_agreement':sum(r['greedy_token_matches'] for r in rows)/trajectory_n,
                'greedy_mean_shared_prefix_tokens':sum(r['greedy_shared_prefix_tokens'] for r in rows)/len(rows),
                'cases':rows,'selection':selection}
            report['profiles'][label]=result;save(args.output/'surrogate-results.json',report)
            print('SURROGATE_RESULT',label,json.dumps({k:v for k,v in result.items() if k not in ('cases','selection')}),flush=True)
    print('SURROGATE_DONE',args.output,flush=True)

if __name__=='__main__': main()
