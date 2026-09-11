#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adapt an existing MiniCPM exporter to an explicit unquantized LiteRT control.

The installed export() drops Python None and otherwise defaults to INT8.
Its maybe_quantize_model() recognizes the literal string 'none'. This wrapper
changes only the child process's export call; it never edits installed libraries
or the supplied exporter. Audit exported tensor dtypes before reporting FP32.
"""
from __future__ import annotations
import argparse, functools, hashlib, importlib, json, runpy, sys
from pathlib import Path


def explicit_unquantized_kwargs(kwargs):
    result=dict(kwargs)
    if result.get('quantization_recipe') not in (None,'none','NONE','FP32'):
        raise ValueError('Refusing to silently replace an explicitly quantized recipe')
    result.update(quantization_recipe='none',experimental_use_mixed_precision=False,
                  experimental_use_fp16=False)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exporter',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    module=importlib.import_module('litert_torch.generative.export_hf.export')
    core=importlib.import_module('litert_torch.generative.export_hf.core.export_lib')
    config_module=importlib.import_module('litert_torch.generative.export_hf.core.exportable_module_config')
    config=config_module.ExportableModuleConfig(model=str(args.model),quantization_recipe='none')
    if config.quantization_recipe!='none' or core.maybe_quantize_model('not-opened.tflite','none')!='not-opened.tflite':
        raise RuntimeError('This toolchain does not honor the explicit no-quantization sentinel')
    print('EXPLICIT_UNQUANTIZED_CONFIG_CHECK_PASSED',flush=True)
    if args.check_only: return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite a control export directory')
    original=module.export
    @functools.wraps(original)
    def unquantized_export(*positional,**kwargs):
        return original(*positional,**explicit_unquantized_kwargs(kwargs))
    module.export=unquantized_export
    original_argv=sys.argv
    try:
        sys.argv=[str(args.exporter),str(args.model),str(args.output),str(args.model/'chat_template.jinja'),'NONE']
        runpy.run_path(str(args.exporter),run_name='__main__')
        record={'requested_precision':'unquantized_float32','explicit_recipe':'none',
            'dtype_audit':'pending','exporter_sha256':hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
            'note':'This record is configuration evidence, not a tensor-dtype audit or runtime parity result.'}
        (args.output/'control-configuration.json').write_text(json.dumps(record,indent=2))
    finally:
        sys.argv=original_argv;module.export=original

if __name__=='__main__': main()
