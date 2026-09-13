#!/usr/bin/env python3
"""Remove oversized prefill entry points without altering weights or cache shapes.

This deliberately supports straight-line LiteRT exports with StableHLO composite
subgraphs. Unsupported control-flow/reference types fail closed. FlatBuffer
tables and external buffers never move: only vector entries/lengths and their
subgraph indices change. Every other byte is verified against the source.
"""
import argparse
import hashlib
import io
import json
import mmap
import pathlib
import re
import shutil
import struct
import tempfile

from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import peek_litertlm_file


def tflite_sections(path):
    with tempfile.TemporaryDirectory(prefix='litert_mobile_metadata_') as tmp:
        output = io.StringIO()
        peek_litertlm_file(str(path), tmp, output)
        pattern = r'Section (\d+):\n  Items:\n.*?\n  Begin Offset: (\d+)\n  End Offset:\s+(\d+)\n  Data Type:\s+TFLiteModel'
        sections = [tuple(int(x) for x in m.groups()) for m in re.finditer(pattern, output.getvalue(), re.S)]
    if not sections:
        raise ValueError('No TFLite model sections found')
    return sections


def plan_section(data, base, end, max_prefill, prefill_lengths=None):
    model = schema.Model.GetRootAsModel(memoryview(data)[base:end], 0)
    before = []
    keep_signatures = []
    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        name = sig.SignatureKey().decode()
        before.append(name)
        match = re.fullmatch(r'prefill_(?:embedder_)?(\d+)', name)
        if name not in ('decode', 'decode_embedder') and match is None:
            raise ValueError(f'Unsupported signature name: {name}')
        if match is None or (int(match.group(1)) <= max_prefill and
                             (prefill_lengths is None or int(match.group(1)) in prefill_lengths)):
            keep_signatures.append(i)
    kept_names = [before[i] for i in keep_signatures]
    if not any(name in ('decode', 'decode_embedder') for name in kept_names):
        raise ValueError('No decode entry point remains')
    if not any(name.startswith('prefill_') for name in kept_names):
        raise ValueError('No prefill entry point remains')

    edges = {}
    fields = {}
    unsupported_ops = {getattr(schema.BuiltinOperator, k) for k in
                       ('IF', 'WHILE', 'CALL', 'CALL_ONCE', 'CUSTOM') if hasattr(schema.BuiltinOperator, k)}
    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        edges[s] = []
        fields[s] = []
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if code in unsupported_ops:
                raise ValueError(f'Unsupported control-flow/custom opcode: {code}')
            if op.BuiltinOptions2Type() == schema.BuiltinOptions2.StableHLOCompositeOptions:
                table = op.BuiltinOptions2()
                opts = schema.StableHLOCompositeOptions()
                opts.Init(table.Bytes, table.Pos)
                child = opts.DecompositionSubgraphIndex()
                if not 0 <= child < model.SubgraphsLength():
                    raise ValueError('Invalid composite subgraph reference')
                edges[s].append(child)
                fields[s].append((opts, 6, child))
            elif op.BuiltinOptions2Type() != schema.BuiltinOptions2.NONE:
                raise ValueError(f'Unsupported BuiltinOptions2 type: {op.BuiltinOptions2Type()}')

    reachable = set()
    pending = [model.SignatureDefs(i).SubgraphIndex() for i in keep_signatures]
    while pending:
        s = pending.pop()
        if s in reachable:
            continue
        if s not in edges:
            raise ValueError('Invalid signature subgraph index')
        reachable.add(s)
        pending.extend(edges[s])
    kept_graphs = sorted(reachable)
    remap = {old: new for new, old in enumerate(kept_graphs)}
    patches = {}

    def put(relative, value):
        absolute = base + relative
        encoded = struct.pack('<I', value)
        if data[absolute:absolute + 4] != encoded:
            if absolute in patches and patches[absolute] != encoded:
                raise ValueError('Conflicting metadata patch')
            patches[absolute] = encoded

    def scalar(table, field, old, new):
        if old == new:
            return
        offset = table._tab.Offset(field)
        if not offset:
            raise ValueError('Cannot insert absent FlatBuffer scalar in place')
        put(table._tab.Pos + offset, new)

    def vector(field, kept):
        offset = model._tab.Offset(field)
        if not offset:
            raise ValueError('Missing FlatBuffer vector')
        start = model._tab.Vector(offset)
        put(start - 4, len(kept))
        for new_index, old_index in enumerate(kept):
            old_slot = start + old_index * 4
            target = old_slot + struct.unpack_from('<I', data, base + old_slot)[0]
            new_slot = start + new_index * 4
            put(new_slot, target - new_slot)

    for i in keep_signatures:
        sig = model.SignatureDefs(i)
        old = sig.SubgraphIndex()
        scalar(sig, 12, old, remap[old])
    for old in kept_graphs:
        for table, field, child in fields[old]:
            scalar(table, field, child, remap[child])
    vector(8, kept_graphs)
    vector(18, keep_signatures)
    return patches, {'signatures_before': before, 'signatures_after': kept_names,
                     'subgraphs_before': model.SubgraphsLength(), 'subgraphs_after': len(kept_graphs),
                     'removed_subgraphs': [i for i in range(model.SubgraphsLength()) if i not in reachable]}


def verify_only_patches(source, candidate, patches):
    """Compare every byte, permitting only the exact planned four-byte edits."""
    if source.stat().st_size != candidate.stat().st_size:
        raise ValueError('File size changed')
    offsets = sorted(patches)
    if any(b < a + len(patches[a]) for a, b in zip(offsets, offsets[1:])):
        raise ValueError('Overlapping metadata patches')
    with source.open('rb') as a, candidate.open('rb') as b:
        pos = 0
        for offset in offsets + [source.stat().st_size]:
            remaining = offset - pos
            while remaining:
                n = min(8 * 1024 * 1024, remaining)
                if a.read(n) != b.read(n):
                    raise ValueError(f'Unexpected non-metadata modification near {pos}')
                pos += n
                remaining -= n
            if offset in patches:
                width = len(patches[offset])
                a.read(width)
                if b.read(width) != patches[offset]:
                    raise ValueError('Metadata patch does not match plan')
                pos += width


def prepare(source, candidate, max_prefill, prefill_lengths=None):
    source, candidate = pathlib.Path(source), pathlib.Path(candidate)
    if max_prefill < 1 or source.resolve() == candidate.resolve():
        raise ValueError('Invalid prefill limit or output equals input')
    if candidate.exists():
        raise FileExistsError(candidate)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    section_ranges = tflite_sections(source)
    patches, reports = {}, []
    with source.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for sid, base, end in section_ranges:
            edits, report = plan_section(data, base, end, max_prefill, prefill_lengths)
            patches.update(edits)
            reports.append({'section': sid, **report})
    shutil.copyfile(source, candidate)
    with candidate.open('r+b') as f:
        for offset, encoded in sorted(patches.items()):
            f.seek(offset); f.write(encoded)
    verify_only_patches(source, candidate, patches)
    # Reparse the result and verify that its graph is closed and pruning again
    # is idempotent; this catches remapping/pointer errors before device use.
    with candidate.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for _, base, end in section_ranges:
            edits, _ = plan_section(data, base, end, max_prefill, prefill_lengths)
            if edits:
                raise ValueError('Candidate is not a closed, idempotently pruned graph')
    def digest(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    return {'source': str(source), 'source_sha256': digest(source), 'candidate': str(candidate),
            'sha256': digest(candidate), 'bytes': candidate.stat().st_size,
            'max_prefill_tokens': max_prefill, 'metadata_fields_changed': len(patches),
            'selected_prefill_lengths': sorted(prefill_lengths) if prefill_lengths else None,
            'all_other_bytes_identical': True, 'weights_and_cache_shapes_unchanged': True,
            'sections': reports, 'patches': [{'offset': p, 'hex': patches[p].hex()} for p in sorted(patches)],
            'device_validation': 'not_performed_by_this_tool'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--max-prefill', type=int, default=256)
    ap.add_argument('--prefill-lengths', help='Optional comma-separated existing prefill lengths to retain')
    ap.add_argument('--report', type=pathlib.Path, required=True)
    args = ap.parse_args()
    lengths = {int(x) for x in args.prefill_lengths.split(',')} if args.prefill_lengths else None
    if lengths is not None and (not lengths or min(lengths) < 1):
        raise ValueError('Invalid prefill length selection')
    result = prepare(args.source, args.output, args.max_prefill, lengths)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('patches', 'sections')}), flush=True)


if __name__ == '__main__':
    main()
