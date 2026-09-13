#!/usr/bin/env python3
"""Export MiniCPM5 with a bounded prefill and an explicit calibrated precision map.

Uses the upstream LiteRT-Torch exporter and quantizer. No benchmark questions
are loaded here, and the selected precision is not a lossless guarantee.
"""
import argparse, copy, hashlib, importlib.metadata, json, pathlib, shutil, time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=pathlib.Path,required=True)
    p.add_argument('--selection',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--context',type=int,choices=[16384,32768,65536],required=True)
    p.add_argument('--prefill',type=int,default=64,choices=[32,64,128,256])
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): raise ValueError('Choose an empty output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.output).free < 18*1024**3:
        raise RuntimeError('At least 18 GiB free scratch space is required for this single export')
    model_config=json.loads((a.model/'config.json').read_text())
    if model_config.get('num_hidden_layers')!=42 or model_config.get('vocab_size')!=130560:
        raise ValueError('This verified recipe is scoped to MiniCPM5-2B')
    selection=json.loads(a.selection.read_text())['selection']
    for key in ['attention_int8_layers','mlp_int8_layers']:
        if not isinstance(selection[key],list) or any(type(i)!=int or not 0<=i<42 for i in selection[key]):
            raise ValueError('Invalid precision map: '+key)
    if not selection['embedding_int8'] or not selection['lm_head_int8']:
        raise ValueError('This recipe protects embeddings and the output head at INT8')

    import ai_edge_quantizer.recipe as recipe
    from litert_torch.generative.export_hf.export import export
    int4=copy.deepcopy(recipe.dynamic_wi4_afp32()[0])
    int8=copy.deepcopy(int4)
    int8['op_config']['weight_tensor_config']['num_bits']=8
    int4['algorithm_key']=recipe.AlgorithmName.OCTAV
    int4['op_config']['weight_tensor_config']['granularity']='BLOCKWISE_32'
    def rule(regex,operation='FULLY_CONNECTED'):
        value=copy.deepcopy(int8);value.update(regex=regex,operation=operation);return value
    def selected_recipe():
        rules=[copy.deepcopy(int4),rule('.*','EMBEDDING_LOOKUP')]
        for key,kind in [('attention_int8_layers','LlamaAttention_self_attn'),('mlp_int8_layers','LlamaMLP_mlp')]:
            layers='|'.join(map(str,selection[key]))
            if layers:rules.append(rule(rf'.*LlamaDecoderLayer_(?:{layers})/.*{kind}.*'))
        rules.append(rule(r'^decode_logits_output;$'))
        return rules
    recipe.MINICPM_MOBILE_CALIBRATED=selected_recipe
    report={'source_model':str(a.model),'context':a.context,'prefill':[a.prefill],
            'selection_sha256':hashlib.sha256(a.selection.read_bytes()).hexdigest(),
            'selection':selection,'fp16_activations':True,'external_embedder':True,
            'tokenizer':'original HF tokenizer and original Jinja template',
            'toolchain':{name:importlib.metadata.version(name) for name in
                         ['torch','transformers','litert-torch','litert-converter','ai-edge-quantizer-nightly','litert-lm-builder']}}
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    start=time.monotonic()
    export(model=str(a.model),output_dir=str(a.output),prefill_lengths=[a.prefill],
           cache_length=a.context,quantization_recipe='MINICPM_MOBILE_CALIBRATED',
           use_jinja_template=True,experimental_use_fp16=True,
           experimental_use_mixed_precision=False,externalize_embedder=True,
           keep_temporary_files=False,trust_remote_code=False)
    path=a.output/'model.litertlm'
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    report.update(sha256=h.hexdigest(),bytes=path.stat().st_size,elapsed_seconds=time.monotonic()-start,
                  device_validation='pending')
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
