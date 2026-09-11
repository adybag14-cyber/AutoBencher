#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit benchmark identity, output caps, and the frozen artifacts actually served.

This is an evidence-integrity check, not a quality acceptance test. No question
text or model response text is copied into the public audit. Runtime failures
remain unscored and prevent the all-runs-completed flag from becoming true.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

TAGS = {'16k', '32k', '64k'}
BENCHMARKS = {'ifeval': 512, 'gpqa-diamond': 2048}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def prediction(repo: Path, run_id: str, benchmark: str) -> dict:
    run = repo / '.autobencher' / 'runs' / run_id
    files = list((run / benchmark / 'predictions').rglob('*.jsonl'))
    rows = []
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    if len(rows) != 1:
        raise RuntimeError(f'Expected one diagnostic item for {run_id}/{benchmark}, found {len(rows)}')
    row = rows[0]
    output = row['model_output']
    choices = output.get('choices') or []
    if len(choices) != 1:
        raise RuntimeError(f'Expected one completed model choice for {run_id}')
    choice = choices[0]
    text = choice['message']['content']
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f'Empty or nontext response in {run_id}')
    prompt = [{'role': m['role'], 'content': m['content']} for m in row['messages'] if m.get('source') != 'generate']
    if prompt and prompt[-1]['role'] == 'assistant' and prompt[-1]['content'] == text:
        prompt.pop()
    if not prompt:
        raise RuntimeError(f'No benchmark prompt in {run_id}')
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    return {
        'prompt_sha256': hashlib.sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'output_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'output_tokens': output.get('usage', {}).get('output_tokens'),
        'input_tokens': output.get('usage', {}).get('input_tokens'),
        'reported_stop_reason': choice.get('stop_reason'),
        'explicit_answer_line': bool(re.search(r'(?im)^\s*ANSWER:\s*[A-D]\s*$', text)) if benchmark == 'gpqa-diamond' else None,
        'backend_error': output.get('error'),
        'run_model': manifest.get('model'),
        'selected_benchmarks': manifest.get('selected_benchmarks'),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--comparison', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--root', type=Path, default=Path(r'\\wsl.localhost\Ubuntu\home\tdamre\minicpm5-litert-20260910'))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data = json.loads(args.comparison.read_text(encoding='utf-8'))
    if data.get('status') != 'completed':
        raise RuntimeError('Comparison orchestration is not complete')
    if data.get('family') != 'DynV31A':
        raise RuntimeError('This audit is specific to DynV31A imported snapshots')
    if set(data.get('variants', {})) != TAGS:
        raise RuntimeError('The comparison must include all three context capacities')
    audits, errors, incomplete = {}, [], []
    for tag, row in data['variants'].items():
        if set(row.get('benchmarks', {})) != set(BENCHMARKS):
            raise RuntimeError(f'{tag}: missing or unexpected diagnostic')
        frozen = args.root / f'benchmarks_dynv31/runtime/.litert-lm/models/minicpm5-dynv31a-{tag}/model.litertlm'
        actual = sha(frozen)
        matches = actual == row['sha256'] and frozen.stat().st_size == row['bytes']
        if not matches:
            errors.append(f'{tag}: served artifact identity differs from result manifest')
        pair = {'served_artifact_sha256': actual, 'artifact_matches_manifest': matches, 'benchmarks': {}}
        for bench, result in row['benchmarks'].items():
            if result['max_tokens'] != BENCHMARKS[bench]:
                errors.append(f'{tag}/{bench}: unexpected output allowance')
            if result['status'] != 'completed':
                incomplete.append(f'{tag}/{bench}')
                pair['benchmarks'][bench] = {'status': result['status'], 'score_valid': False, 'error': result.get('error')}
                if result.get('score_percent') is not None:
                    errors.append(f'{tag}/{bench}: noncompleted run has a numeric score')
                continue
            current = prediction(repo, result['run_id'], bench)
            baseline_record = data['bf16_baseline']['benchmarks'][bench]
            baseline = prediction(repo, baseline_record['run_id'], bench)
            same_prompt = current['prompt_sha256'] == baseline['prompt_sha256']
            same_tokens = current['input_tokens'] == baseline['input_tokens']
            run_valid = current['run_model'] == f'minicpm5-dynv31a-{tag}' and current['selected_benchmarks'] == [bench]
            if not same_prompt:
                errors.append(f'{tag}/{bench}: benchmark prompt differs from BF16')
            if not same_tokens:
                errors.append(f'{tag}/{bench}: input token count differs from BF16')
            if not run_valid:
                errors.append(f'{tag}/{bench}: run manifest model/benchmark identity mismatch')
            if current['backend_error'] or baseline['backend_error']:
                errors.append(f'{tag}/{bench}: backend error attached to completed output')
            if baseline_record['max_tokens'] != result['max_tokens']:
                errors.append(f'{tag}/{bench}: BF16 and candidate output allowances differ')
            count = current['output_tokens']
            if not isinstance(count, int) or not 0 < count <= result['max_tokens']:
                errors.append(f'{tag}/{bench}: missing, empty or over-budget generation')
            if baseline['output_sha256'] != baseline_record['output_sha256']:
                errors.append(f'{tag}/{bench}: BF16 output differs from saved baseline receipt')
            current.update({
                'same_prompt_as_bf16': same_prompt,
                'same_input_token_count_as_bf16': same_tokens,
                'same_output_as_bf16': current['output_sha256'] == baseline['output_sha256'],
                'at_output_cap': count == result['max_tokens'],
                'bf16': baseline,
                'score_valid': same_prompt and same_tokens and run_valid and current['backend_error'] is None,
            })
            pair['benchmarks'][bench] = current
        audits[tag] = pair
    data['audit'] = {
        'evidence_integrity_passed': not errors,
        'all_runs_completed': not incomplete,
        'errors': errors,
        'incomplete_runs': incomplete,
        'variants': audits,
        'cap_note': 'An output count equal to its cap is reported separately from the adapter stop reason. It is not assumed to be a natural full completion.',
        'scope_note': 'Matched prompts, artifact identities and one-item scores do not establish full-benchmark retention or near-lossless quantization.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'evidence_integrity_passed': not errors, 'all_runs_completed': not incomplete, 'errors': errors, 'out': str(args.out)}, indent=2), flush=True)
    if errors:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
