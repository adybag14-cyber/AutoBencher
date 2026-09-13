import copy
import pathlib
import tempfile
import unittest

import flatbuffers
from flatbuffers import flexbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter

from repair_litert_npu_mask import rewrite as rewrite_mask
from repair_litert_npu_cache_inputs import repair, rewrite as rewrite_cache


def tensor(name, shape):
    value = s.TensorT()
    value.name, value.shape, value.type, value.buffer = name, shape, s.TensorType.INT16, 0
    q = s.QuantizationParametersT()
    q.scale, q.zeroPoint = [0.25], [0]
    value.quantization = q
    return value


def code(builtin, version=1):
    value = s.OperatorCodeT()
    value.builtinCode = builtin
    value.deprecatedBuiltinCode = min(builtin, 127)
    value.version = version
    return value


def signature(name, graph_index, inputs, outputs):
    value = s.SignatureDefT()
    value.signatureKey, value.subgraphIndex = name, graph_index
    def mapping(items):
        result = []
        for name, index in items:
            entry = s.TensorMapT()
            entry.name, entry.tensorIndex = name, index
            result.append(entry)
        return result
    value.inputs, value.outputs = mapping(inputs), mapping(outputs)
    return value


def pack(model):
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


def mask_fixture():
    model, graph = s.ModelT(), s.SubGraphT()
    model.version, model.buffers = 3, [s.BufferT()]
    model.operatorCodes = [code(s.BuiltinOperator.CONCATENATION, 3),
                           code(s.BuiltinOperator.ADD, 3), code(s.BuiltinOperator.RESHAPE)]
    graph.name = b'prefill_3'
    graph.tensors = [tensor(b'scores', [1, 2, 12, 5]), tensor(b'input_mask_global', [1, 1, 3, 5]),
                     tensor(b'repeated_mask', [1, 1, 12, 5]), tensor(b'sum', [1, 2, 12, 5])]
    concat, add = s.OperatorT(), s.OperatorT()
    concat.opcodeIndex, concat.inputs, concat.outputs = 0, [1]*4, [2]
    concat.builtinOptionsType = s.BuiltinOptions.ConcatenationOptions
    concat.builtinOptions = s.ConcatenationOptionsT()
    concat.builtinOptions.axis = 2
    add.opcodeIndex, add.inputs, add.outputs = 1, [0, 2], [3]
    add.builtinOptionsType = s.BuiltinOptions.AddOptions
    add.builtinOptions = s.AddOptionsT()
    add.builtinOptions.potScaleInt16 = False
    graph.inputs, graph.outputs, graph.operators = [0, 1], [3], [concat, add]
    model.subgraphs = [graph]
    return model


def evaluate(model, scores, mask):
    interpreter = Interpreter(model_content=pack(model), num_threads=1)
    interpreter.allocate_tensors()
    interpreter.set_tensor(0, scores)
    interpreter.set_tensor(1, mask)
    interpreter.invoke()
    return interpreter.get_tensor(3)


def cache_fixture():
    model, prefill = s.ModelT(), s.SubGraphT()
    model.version, model.buffers = 3, [s.BufferT()]
    dispatch = code(s.BuiltinOperator.CUSTOM)
    dispatch.customCode = b'DISPATCH_OP'
    model.operatorCodes = [dispatch]
    prefill.name = b'prefill_64'
    prefill.tensors = [tensor(b'kv_cache_k_0', [1]), tensor(b'kv_cache_k_1', [1]),
                       tensor(b'kv_slice_k_1', [1])]
    op = s.OperatorT()
    op.opcodeIndex, op.inputs, op.outputs = 0, [0], [2]
    op.customOptions = np.frombuffer(flexbuffers.Dumps({'bytecode_offset':4096, 'bytecode_size':64}), dtype=np.uint8)
    prefill.inputs, prefill.outputs, prefill.operators = [0, 1], [2], [op]
    decode = copy.deepcopy(prefill)
    decode.name, decode.operators[0].inputs = b'decode', [0, 1]
    model.subgraphs = [prefill, decode]
    inputs = [(b'kv_cache_k_0', 0), (b'kv_cache_k_1', 1)]
    outputs = [(b'kv_slice_k_1', 2)]
    model.signatureDefs = [signature(b'prefill_64', 0, inputs, outputs), signature(b'decode', 1, inputs, outputs)]
    return model


class NpuRepairTest(unittest.TestCase):
    def test_mask_rewrite_preserves_real_interpreter_outputs_for_every_head(self):
        model = mask_fixture()
        before = copy.deepcopy(model)
        report = rewrite_mask(model)
        self.assertEqual(report[0]['groups'], 4)
        rng = np.random.default_rng(42)
        for _ in range(3):
            scores = rng.integers(-100, 100, (1, 2, 12, 5), dtype=np.int16)
            mask = np.array([0, -100, -100, -100, -100, 0, 0, -100, -100, -100,
                             0, 0, 0, -100, -100], dtype=np.int16).reshape(1, 1, 3, 5)
            np.testing.assert_array_equal(evaluate(before, scores, mask), evaluate(model, scores, mask))
        self.assertFalse(any(model.operatorCodes[op.opcodeIndex].builtinCode == s.BuiltinOperator.CONCATENATION
                             for op in model.subgraphs[0].operators))

    def test_mask_requantization_and_unsupported_dimensions_are_rejected(self):
        model = mask_fixture()
        model.subgraphs[0].tensors[2].quantization.scale = [0.5]
        with self.assertRaisesRegex(ValueError, 'requantizes'):
            rewrite_mask(model)
        model = mask_fixture()
        model.subgraphs[0].tensors[0].shape = [1, 2, 11, 5]
        with self.assertRaisesRegex(ValueError, 'shape'):
            rewrite_mask(model)

    def test_unused_prefill_cache_is_removed_but_decode_and_code_are_preserved(self):
        model = cache_fixture()
        before_decode = copy.deepcopy(model.signatureDefs[1].inputs)
        report = rewrite_cache(model)
        self.assertEqual([row['name'] for row in report], ['kv_cache_k_1'])
        self.assertEqual(model.subgraphs[0].inputs, [0])
        self.assertEqual(model.subgraphs[1].inputs, [0, 1])
        self.assertEqual([(item.name, item.tensorIndex) for item in model.signatureDefs[1].inputs],
                         [(item.name, item.tensorIndex) for item in before_decode])
        with tempfile.TemporaryDirectory() as folder:
            source, output = pathlib.Path(folder)/'source.tflite', pathlib.Path(folder)/'fixed.tflite'
            payload = bytes(range(64))
            blob = pack(cache_fixture())
            self.assertLess(len(blob), 4096)
            source.write_bytes(blob + b'\0'*(4096-len(blob)) + payload)
            result = repair(source, output)
            self.assertEqual(output.read_bytes()[4096:], payload)
            self.assertTrue(result['all_bytes_after_metadata_identical'])
            self.assertTrue(result['buffers_identical'])

    def test_missing_decode_cache_and_shared_graph_are_rejected(self):
        model = cache_fixture()
        model.signatureDefs[1].inputs = model.signatureDefs[1].inputs[:1]
        with self.assertRaisesRegex(ValueError, 'no decode input'):
            rewrite_cache(model)
        model = cache_fixture()
        model.signatureDefs[1].subgraphIndex = 0
        with self.assertRaisesRegex(ValueError, 'shared'):
            rewrite_cache(model)


if __name__ == '__main__':
    unittest.main()
