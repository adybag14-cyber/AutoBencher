import pathlib
import tempfile
import unittest

import flatbuffers
from ai_edge_litert import schema_py_generated as s
from prepare_litert_mobile import plan_section, verify_only_patches


def fixture(unsafe=False):
    composite = s.StableHLOCompositeOptionsT(name='odml.rms_norm', decompositionSubgraphIndex=2)
    op = s.OperatorT()
    op.opcodeIndex = 0
    op.builtinOptions2Type = s.BuiltinOptions2.StableHLOCompositeOptions
    op.builtinOptions2 = composite
    graph0, graph1, graph2 = s.SubGraphT(), s.SubGraphT(), s.SubGraphT()
    graph0.name, graph1.name, graph2.name = 'large', 'small', 'norm'
    graph0.operators = []
    graph1.operators = [op]
    graph2.operators = []
    code = s.OperatorCodeT()
    code.builtinCode = s.BuiltinOperator.IF if unsafe else s.BuiltinOperator.STABLEHLO_COMPOSITE
    code.deprecatedBuiltinCode = 127
    signatures = []
    for name, index in [('prefill_1024', 0), ('prefill_64', 1), ('decode', 1)]:
        sig = s.SignatureDefT()
        sig.signatureKey = name
        sig.subgraphIndex = index
        signatures.append(sig)
    buffer = s.BufferT()
    buffer.data = list(range(256)) * 32
    model = s.ModelT()
    model.version = 3
    model.subgraphs = [graph0, graph1, graph2]
    model.signatureDefs = signatures
    model.operatorCodes = [code]
    model.buffers = [buffer]
    builder = flatbuffers.Builder(16384)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


class PruningTest(unittest.TestCase):
    def test_composite_indices_remap_and_every_other_byte_is_preserved(self):
        original = fixture()
        patches, report = plan_section(original, 0, len(original), 256)
        candidate = bytearray(original)
        for offset, value in patches.items():
            candidate[offset:offset+4] = value
        self.assertEqual(report['signatures_after'], ['prefill_64', 'decode'])
        m = s.Model.GetRootAsModel(candidate, 0)
        self.assertEqual(m.SubgraphsLength(), 2)
        self.assertEqual(m.SignatureDefs(0).SubgraphIndex(), 0)
        op = m.Subgraphs(0).Operators(0)
        table = op.BuiltinOptions2()
        opts = s.StableHLOCompositeOptions()
        opts.Init(table.Bytes, table.Pos)
        self.assertEqual(opts.DecompositionSubgraphIndex(), 1)
        self.assertEqual(bytes(m.Buffers(0).DataAsNumpy()), bytes(range(256))*32)
        self.assertEqual(plan_section(candidate, 0, len(candidate), 256)[0], {})
        with tempfile.TemporaryDirectory() as temp:
            a, b = pathlib.Path(temp)/'a', pathlib.Path(temp)/'b'
            a.write_bytes(original); b.write_bytes(candidate)
            verify_only_patches(a, b, patches)
            candidate[-1] ^= 1
            b.write_bytes(candidate)
            with self.assertRaises(ValueError):
                verify_only_patches(a, b, patches)

    def test_missing_prefill_and_unknown_control_flow_fail_closed(self):
        data = fixture()
        with self.assertRaises(ValueError):
            plan_section(data, 0, len(data), 4)
        data = fixture(unsafe=True)
        with self.assertRaises(ValueError):
            plan_section(data, 0, len(data), 256)


if __name__ == '__main__':
    unittest.main()
