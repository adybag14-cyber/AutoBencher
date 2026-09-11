#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish audited experimental snapshots without replacing existing releases.

The explicitly supplied token file is read only after evidence checks. Its
contents are never printed, copied into metadata, or passed to a subprocess.
A parent-commit guard protects concurrent model-card edits.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = 'Tdamre/MiniCPM5-2B-LiteRT-LongContext'
PREFIX = 'experimental/dynv31a-2026-09-11'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def validate_evidence(evidence: dict) -> None:
    if evidence.get('status') != 'completed' or set(evidence.get('variants', {})) != {'16k', '32k', '64k'}:
        raise ValueError('All three context diagnostics must finish before publication')
    if evidence.get('family') != 'DynV31A':
        raise ValueError('Wrong candidate family')
    audit = evidence.get('audit', {})
    if audit.get('evidence_integrity_passed') is not True or audit.get('errors'):
        raise ValueError('Run audit_minicpm5_comparison.py and resolve all evidence-integrity errors first')
    if audit.get('all_runs_completed') is not True or audit.get('incomplete_runs'):
        raise ValueError('Experimental artifact publication requires completed diagnostic runs, not transport failures')
    if evidence.get('near_lossless_demonstrated') is not False:
        raise ValueError('This diagnostic cannot certify near-lossless quality')
    for tag, row in evidence['variants'].items():
        if set(row.get('benchmarks', {})) != {'ifeval', 'gpqa-diamond'}:
            raise ValueError(f'{tag}: a diagnostic is missing')
        for bench, result in row['benchmarks'].items():
            audited = audit['variants'][tag]['benchmarks'][bench]
            if result['status'] != 'completed' or audited.get('score_valid') is not True:
                raise ValueError(f'{tag}/{bench}: completed valid inference is required')
            if not isinstance(result.get('score_percent'), (int, float)):
                raise ValueError(f'{tag}/{bench}: missing score')


def report_text(evidence: dict, receipts: list[dict]) -> str:
    lines = ['# DynV31A experimental artifacts — 11 September 2026', '',
        '**Experimental; not a replacement for existing root artifacts. Near-lossless quality has not been established.**', '',
        'Dynamic-v3-inspired LiteRT adaptation, not an official Unsloth quantization or GGUF. All attention blocks, 20 selected MLP blocks, the vocabulary embedding and output head use INT8. Remaining MLP blocks use block-32 OCTAV INT4.', '',
        'This is a two-item development regression diagnostic, not full IFEval or GPQA leaderboard accuracy. Each percentage describes one selected item. The BF16 controls are earlier matched AutoBencher runs, with exact prompt and output-receipt identity checked by the accompanying audit.', '',
        '| Context | IFEval candidate / BF16 | GPQA candidate / BF16 | GPQA tokens |',
        '|---|---:|---:|---:|']
    for tag in ('16k', '32k', '64k'):
        b = evidence['variants'][tag]['benchmarks']
        a = evidence['audit']['variants'][tag]['benchmarks']['gpqa-diamond']
        cell = lambda key: f"{b[key]['score_percent']:g}% / {b[key]['bf16_score_percent']:g}%"
        lines.append(f'| {tag} | {cell("ifeval")} | {cell("gpqa-diamond")} | {a["output_tokens"]} / {b["gpqa-diamond"]["max_tokens"]} |')
    lines.extend(['', '## Completion and formatting audit', ''])
    for tag in ('16k', '32k', '64k'):
        a = evidence['audit']['variants'][tag]['benchmarks']['gpqa-diamond']
        lines.append(f'{tag}: output at cap = **{a["at_output_cap"]}**; adapter-reported stop reason = `{a["reported_stop_reason"]}`; explicit `ANSWER: [LETTER]` line = **{a["explicit_answer_line"]}**. The score is the evaluator\'s extracted-choice score, not a separate formatting score.')
        lines.append('')
    lines.extend([
        'An output exactly at its allowance is not silently treated as a complete natural termination, even when the adapter reports `stop`.', '',
        'Matched protocol: seed 42, temperature 0, top-p 1, thinking disabled, evaluation batch size 1. IFEval allows 512 generated tokens and GPQA 2048. LiteRT-LM uses CPU inference with 32 threads and AutoBencher\'s token-cap compatibility proxy.', '',
        'Long-context NoLiMa retention has not been established for this family. Running a short question in a 64K-capacity artifact does not validate 64K retrieval. The previous DynV3 family had demonstrated long-context failures. End-to-end differences versus BF16 can include conversion, tokenizer and runtime effects, not solely weight quantization. BF16 GPU and LiteRT CPU latency must not be compared as a quantization speedup.', '',
        'DynV32 context-aware calibration is a separate candidate family. Its results must not be substituted for these measurements.', '',
        '## Frozen artifact hashes', '', '```text'])
    lines.extend(f"{r['sha256']}  {r['file']}" for r in receipts)
    lines.extend(['```', ''])
    return '\n'.join(lines)


def main() -> None:
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-file', required=True, type=Path)
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--comparison', required=True, type=Path)
    ap.add_argument('--receipt', required=True, type=Path)
    ap.add_argument('--experimental', required=True, action='store_true')
    args = ap.parse_args()
    evidence = json.loads(args.comparison.read_text(encoding='utf-8'))
    validate_evidence(evidence)
    ops, receipts = [], []
    for tag, row in evidence['variants'].items():
        artifact = args.root / f'benchmarks_dynv31/runtime/.litert-lm/models/minicpm5-dynv31a-{tag}/model.litertlm'
        actual = digest(artifact)
        if actual != row['sha256'] or artifact.stat().st_size != row['bytes']:
            raise RuntimeError(f'Served-artifact identity mismatch for {tag}; do not publish these scores')
        name = f'MiniCPM5-2B-LiteRT-DynV31A-{tag}.litertlm'
        ops.append(CommitOperationAdd(path_in_repo=f'{PREFIX}/{name}', path_or_fileobj=str(artifact)))
        receipts.append({'file': name, 'sha256': actual, 'bytes': artifact.stat().st_size})
    report = report_text(evidence, receipts)
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'family': 'DynV31A', 'publication_channel': 'experimental', 'near_lossless_demonstrated': False,
        'served_artifact_hashes_verified': True, 'artifacts': receipts}
    payloads = {f'{PREFIX}/REPORT.md': report.encode(),
        f'{PREFIX}/comparison.json': json.dumps(evidence, indent=2).encode(),
        f'{PREFIX}/artifact_manifest.json': json.dumps(manifest, indent=2).encode()}
    for path, data in payloads.items():
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=io.BytesIO(data)))
    matches = re.findall(r'\bhf_[A-Za-z0-9]+\b', args.token_file.read_text(encoding='utf-8-sig'))
    if len(set(matches)) != 1:
        raise ValueError('Token file must contain exactly one distinct Hugging Face token')
    token = matches[0]
    api = HfApi(token=token)
    parent = api.model_info(REPO_ID).sha
    card = Path(hf_hub_download(REPO_ID, 'README.md', revision=parent, token=token)).read_text(encoding='utf-8')
    marker = '<!-- EXPERIMENTAL_DYNV31A_20260911 -->'
    if marker in card:
        raise RuntimeError('An experiment entry already exists; inspect it instead of overwriting it')
    card += '\n\n' + marker + '\n## Experimental DynV31A candidates\n\n'
    card += 'Additional 16K, 32K and 64K candidates are available under `' + PREFIX + '`. '
    card += 'They do not replace existing artifacts and are not certified near-lossless. '
    card += 'See [the experimental diagnostic report](' + PREFIX + '/REPORT.md) for audited, matched BF16-relative results and limitations.\n'
    ops.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=io.BytesIO(card.encode())))
    result = api.create_commit(repo_id=REPO_ID, repo_type='model', operations=ops,
        commit_message='Add experimental DynV31A artifacts with audited AutoBencher diagnostics', parent_commit=parent)
    receipt = {'repo_id': REPO_ID, 'commit_oid': result.oid, 'commit_url': result.commit_url,
        'parent_commit': parent, 'prefix': PREFIX, 'artifacts': receipts, 'experimental': True}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (args.receipt.parent / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)

if __name__ == '__main__':
    main()
