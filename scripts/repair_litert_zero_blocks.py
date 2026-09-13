#!/usr/bin/env python3
"""Repair MLP blocks below the range of FP16-scaled INT4 in LiteRT bundles.

Some block quantizers emit scale=0 and packed -8 codes for tiny source weights.
Replacing only the scale invents much larger weights. This repair checks the
original BF16 checkpoint and clears a block only when every source value is
closer to zero than the smallest nonzero value representable with its scale
dtype. It also repairs earlier scale-only patches. Every other byte is verified
unchanged. Tiny BF16 values are deliberately rounded to zero, so this is not a
claim of lossless BF16 storage or whole-model accuracy.
"""
import argparse
import gc
import hashlib
import json
import mmap
import pathlib
import re
import shutil

import numpy as np
import torch
from safetensors import safe_open
from ai_edge_litert import schema_py_generated as schema
from prepare_litert_mobile import tflite_sections, verify_only_patches


def zero_block_plan(reference, weight_shape, scale_shape, block_size, packed, scales,
                    min_positive_scale=float(np.nextafter(np.float16(0), np.float16(1)))):
    axes = [axis for axis in range(len(weight_shape)) if all(
        weight_shape[d] == scale_shape[d] * (block_size if d == axis else 1)
        for d in range(len(weight_shape)))]
    if len(axes) != 1 or axes[0] != 1 or len(weight_shape) != 2:
        raise ValueError('Only verified row-major two-dimensional MLP blocks are supported')
    if tuple(reference.shape) != tuple(weight_shape):
        raise ValueError('Reference and exported weight shapes differ')
    magnitudes = np.abs(reference.reshape(weight_shape[0], scale_shape[1], block_size)).max(axis=2)
    # For integer q and a positive scale >= min_positive_scale, every nonzero
    # q*scale is farther away than zero below this half-step threshold.
    source_zero = magnitudes < (min_positive_scale / 2)
    invalid = (~np.isfinite(scales)) | (scales <= 0)
    if np.any(invalid.reshape(scale_shape) & ~source_zero):
        raise ValueError('Invalid scale covers representable BF16 weights; requantization is required')
    positions = np.flatnonzero(source_zero)
    coords = [np.repeat(c, block_size) for c in np.unravel_index(positions, scale_shape)]
    coords[1] = coords[1] * block_size + np.tile(np.arange(block_size), len(positions))
    flat = np.ravel_multi_index(coords, weight_shape)
    changed_bytes = {}
    nonzero_codes = 0
    for index in flat:
        byte_index, shift = int(index // 2), int(4 * (index % 2))
        original = int(packed[byte_index])
        if (original >> shift) & 15:
            nonzero_codes += 1
            value = changed_bytes.get(byte_index, original)
            changed_bytes[byte_index] = value & ~(15 << shift)
    scale_indices = np.flatnonzero(invalid & source_zero.reshape(-1)).tolist()
    return changed_bytes, scale_indices, len(positions), nonzero_codes


def repair(source, output, reference_dir, report_path):
    source, output, reference_dir = map(pathlib.Path, (source, output, reference_dir))
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError('Choose a new output path')
    shards = sorted(reference_dir.glob('*.safetensors'))
    if len(shards) != 1:
        raise ValueError('This verified MiniCPM5-2B repair expects its single BF16 shard')
    torch.set_num_threads(4)
    patches, records, seen = {}, [], set()
    sections = tflite_sections(source)
    with source.open('rb') as f, safe_open(str(shards[0]), framework='pt', device='cpu') as reference:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for sid, base, end in sections:
            model = schema.Model.GetRootAsModel(memoryview(data)[base:end], 0)
            for sg_index in range(model.SubgraphsLength()):
                sg = model.Subgraphs(sg_index)
                for i in range(sg.TensorsLength()):
                    weight = sg.Tensors(i)
                    q = weight.Quantization()
                    if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
                        continue
                    bq = schema.BlockwiseQuantization()
                    table = q.Details(); bq.Init(table.Bytes, table.Pos)
                    scale_tensor = sg.Tensors(bq.Scales())
                    key = (sid, weight.Buffer(), scale_tensor.Buffer())
                    if key in seen:
                        continue
                    seen.add(key)
                    name = weight.Name().decode()
                    match = re.search(r'LlamaDecoderLayer_(\d+)/.*LlamaMLP_mlp/.*Linear_(up_proj|gate_proj|down_proj);', name)
                    if not match:
                        continue
                    if weight.Type() != schema.TensorType.INT4 or bq.ZeroPoints() != -1:
                        raise ValueError('Expected symmetric signed INT4 with no zero-point tensor')
                    source_key = f'model.layers.{match.group(1)}.mlp.{match.group(2)}.weight'
                    original = reference.get_tensor(source_key)
                    if original.dtype != torch.bfloat16:
                        raise ValueError('Reference weights are not BF16')
                    original = original.float().numpy()
                    ws, ss = list(weight.ShapeAsNumpy()), list(scale_tensor.ShapeAsNumpy())
                    def buffer_view(tensor):
                        b = model.Buffers(tensor.Buffer())
                        off = b._tab.Vector(b._tab.Offset(4)) if b.DataLength() else b.Offset()
                        size = b.DataLength() or b.Size()
                        return base + off, size
                    w_offset, w_size = buffer_view(weight)
                    s_offset, s_size = buffer_view(scale_tensor)
                    if scale_tensor.Type() not in (schema.TensorType.FLOAT16, schema.TensorType.FLOAT32):
                        raise ValueError('Unsupported scale dtype')
                    dtype = np.dtype('<f2' if scale_tensor.Type() == schema.TensorType.FLOAT16 else '<f4')
                    packed = np.ndarray(w_size, dtype=np.uint8, buffer=data, offset=w_offset)
                    scales = np.ndarray(s_size // dtype.itemsize, dtype=dtype, buffer=data, offset=s_offset)
                    byte_edits, scale_edits, zero_blocks, changed_codes = zero_block_plan(
                        original, ws, ss, bq.BlockSize(), packed, scales,
                        float(np.nextafter(dtype.type(0), dtype.type(1))))
                    for offset, value in byte_edits.items():
                        patches[w_offset + offset] = bytes([value])
                    for index in scale_edits:
                        patches[s_offset + index * dtype.itemsize] = np.array([1], dtype=dtype).tobytes()
                    if byte_edits or scale_edits:
                        records.append({'tensor': name, 'reference_key': source_key,
                            'zero_optimal_blocks': zero_blocks, 'cleared_nonzero_nibbles': changed_codes,
                            'zero_rounding_threshold': float(np.nextafter(dtype.type(0), dtype.type(1))) / 2,
                            'repaired_invalid_scales': len(scale_edits)})
                    del original, packed, scales
        # All planned edits are scalar/packed-code bytes, not moved buffers.
        if not records:
            raise ValueError('No repairable INT4 MLP blocks were found; no output is written')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        with output.open('r+b') as destination:
            for offset, value in sorted(patches.items()):
                destination.seek(offset); destination.write(value)
        verify_only_patches(source, output, patches)
        # Release parser views before closing the mmap.
        del model, sg, weight, q, bq, table, scale_tensor
        gc.collect(); data.close()
    def digest(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    report = {'source': str(source), 'source_sha256': digest(source), 'output': str(output),
        'sha256': digest(output), 'bytes': output.stat().st_size,
        'bf16_shard_sha256': digest(shards[0]), 'changed_bytes': sum(map(len, patches.values())),
        'non_repaired_bytes_identical': True, 'repairs': records,
        'scope': 'Nearest-representable-zero repair for BF16 MLP blocks below the scale dtype range; all other bytes preserved. No whole-model lossless claim.'}
    pathlib.Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'repairs'}), flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bf16-model',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();repair(a.source,a.output,a.bf16_model,a.report)

if __name__=='__main__':main()
