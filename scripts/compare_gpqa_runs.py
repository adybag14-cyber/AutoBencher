#!/usr/bin/env python3
"""Compare two complete EvalScope GPQA runs without redistributing questions."""
import argparse, hashlib, json, math, pathlib, random

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(root, kind):
    files=sorted((root/'gpqa-diamond'/kind).rglob('gpqa_diamond_*.jsonl'))
    if len(files)!=1: raise ValueError(f'Expected one {kind} file, found {len(files)}')
    rows={}
    # JSONL is delimited by LF, not Unicode NEL/LINE SEPARATOR characters that
    # may legally occur inside an ensure_ascii=False JSON string.
    for line in files[0].read_text(encoding='utf-8').split('\n'):
        if not line.strip():continue
        row=json.loads(line);index=row['index']
        if type(index)!=int or index in rows:raise ValueError('Invalid or duplicate sample index')
        rows[index]=row
    if set(rows)!=set(range(198)):raise ValueError(f'{kind} must contain exactly sample IDs 0..197')
    return rows,{'file':str(files[0]),'sha256':digest(files[0])}

def prompt_hash(messages):
    # EvalScope review messages include the generated final assistant message;
    # IDs and timing fields are per-run metadata, not prompt content.
    if messages and messages[-1]['role']=='assistant':messages=messages[:-1]
    canonical=[{'role':m['role'],'content':m['content']} for m in messages]
    if not canonical or canonical[-1]['role']!='user':raise ValueError('Missing final user prompt')
    return hashlib.sha256(json.dumps(canonical,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def load_run(root):
    results=json.loads((root/'results.json').read_text())
    result=next(r for r in results if r['benchmark_id']=='gpqa-diamond')
    if result['status']!='completed' or result['coverage']['scope']!='full':
        raise ValueError('Only completed full-coverage runs can enter this comparison')
    cfg=json.loads((root/'provenance/config.json').read_text())
    reviews,review_source=read_rows(root,'reviews')
    predictions,prediction_source=read_rows(root,'predictions')
    rows={}
    for index,review in reviews.items():
        score=review['sample_score']['score']
        value=score['value']['accuracy']
        if score.get('status')!='success' or value not in (0,1):raise ValueError('Invalid accuracy or evaluator error')
        if review['sample_score']['sample_id']!=index:raise ValueError('Review ID mismatch')
        prediction=predictions[index]
        if prompt_hash(review['messages'])!=prompt_hash(prediction['messages']):raise ValueError('Prediction/review prompt mismatch')
        output=prediction['model_output']
        if output.get('error'):raise ValueError('Prediction contains an error')
        usage=output['usage'];choice=output['choices'][0]
        rows[index]={'question_sha256':prompt_hash(review['messages']),
                     'target':review['target'],'answer':score.get('extracted_prediction'),
                     'correct':int(value),'input_tokens':usage['input_tokens'],
                     'output_tokens':usage['output_tokens'],
                     'truncated':choice.get('stop_reason') in ('length','max_tokens','model_length'),
                     'evaluator_seconds':output.get('time')}
    correct=sum(r['correct'] for r in rows.values())
    if abs(100*correct/198-result['primary_score'])>.011:raise ValueError('Raw correctness disagrees with reported score')
    return cfg,rows,{'run_id':result['run_id'],'model':cfg['model'],
        'review_source':review_source,'prediction_source':prediction_source,
        'config_sha256':digest(root/'provenance/config.json'),
        'total_attempt_duration_seconds':result.get('total_attempt_duration_seconds')}

def summarize_pairs(baseline, candidate):
    if set(baseline)!=set(candidate) or len(baseline)!=198:raise ValueError('Both runs must have 198 matching IDs')
    paired=[]
    for index in sorted(baseline):
        ref,quant=baseline[index],candidate[index]
        if ref['question_sha256']!=quant['question_sha256'] or ref['target']!=quant['target']:
            raise ValueError(f'Question or answer-option ordering differs at sample {index}')
        paired.append({'id':index,'question_sha256':ref['question_sha256'],
                       'bf16_answer':ref['answer'],'candidate_answer':quant['answer'],
                       'bf16_correct':ref['correct'],'candidate_correct':quant['correct']})
    improvements=sum(not r['bf16_correct'] and r['candidate_correct'] for r in paired)
    regressions=sum(r['bf16_correct'] and not r['candidate_correct'] for r in paired)
    discordant=improvements+regressions
    pvalue=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(improvements,regressions)+1))/2**discordant)
    differences=[r['candidate_correct']-r['bf16_correct'] for r in paired]
    rng=random.Random(20260912)
    samples=sorted(100*sum(rng.choices(differences,k=198))/198 for _ in range(10000))
    def stats(rows):
        values=list(rows.values());correct=sum(r['correct'] for r in values)
        return {'correct':correct,'questions':198,'accuracy_percent':100*correct/198,
                'unparsed_answers':sum(r['answer'] not in ['A','B','C','D'] for r in values),
                'truncated_outputs':sum(r['truncated'] for r in values),
                'input_tokens':sum(r['input_tokens'] for r in values),
                'output_tokens':sum(r['output_tokens'] for r in values)}
    return {'baseline':stats(baseline),'candidate':stats(candidate),
            'accuracy_delta_percentage_points':100*(improvements-regressions)/198,
            'improvements':improvements,'regressions':regressions,
            'both_correct':sum(r['bf16_correct'] and r['candidate_correct'] for r in paired),
            'both_incorrect':sum(not r['bf16_correct'] and not r['candidate_correct'] for r in paired),
            'answer_agreement':sum(r['bf16_answer']==r['candidate_answer'] for r in paired)/198,
            'mcnemar_exact_two_sided_p':pvalue,
            'paired_question_bootstrap_95pct_delta_interval':[samples[249],samples[9749]],
            'bootstrap_seed':20260912,'bootstrap_resamples':10000,
            'scope':'This complete 198-question split under the recorded protocol; not a universal lossless claim or model-card reproduction.',
            'pairs':paired}

def main():
    p=argparse.ArgumentParser();p.add_argument('--bf16',type=pathlib.Path,required=True)
    p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args()
    bcfg,baseline,bsource=load_run(a.bf16);qcfg,candidate,qsource=load_run(a.candidate)
    protocol={k:bcfg[k] for k in ['seed','evalscope_spec','generation_config']}
    for k,value in protocol.items():
        left=json.loads(value) if k=='generation_config' else value
        right=json.loads(qcfg[k]) if k=='generation_config' else qcfg[k]
        if left!=right:raise ValueError('Evaluation protocol differs: '+k)
    report=summarize_pairs(baseline,candidate)
    report.update(protocol=protocol,baseline_provenance=bsource,candidate_provenance=qsource)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'}),flush=True)

if __name__=='__main__':main()
