import unittest
import flatbuffers
from ai_edge_litert import schema_py_generated as s
from audit_litert_precision import inspect_section
from prepare_litert_mobile import parse_tflite_sections

def fixture(bits):
    model=s.ModelT();model.version=3;model.buffers=[s.BufferT()]
    code=s.OperatorCodeT();code.builtinCode=s.BuiltinOperator.FULLY_CONNECTED
    model.operatorCodes=[code]
    graph=s.SubGraphT();graph.name=b'decode';graph.inputs=[0];graph.outputs=[2]
    graph.tensors=[]
    for name,kind,shape in [(b'input',s.TensorType.FLOAT32,[1,2048]),
                            (b'LlamaForCausalLM/Linear_lm_head;',bits,[130560,2048]),
                            (b'renamed_after_FP16_conversion',s.TensorType.FLOAT32,[1,130560])]:
        tensor=s.TensorT();tensor.name=name;tensor.type=kind;tensor.shape=shape
        graph.tensors.append(tensor)
    op=s.OperatorT();op.opcodeIndex=0;op.inputs=[0,1];op.outputs=[2]
    graph.operators=[op];model.subgraphs=[graph]
    builder=flatbuffers.Builder(1024);value=model.Pack(builder);builder.Finish(value,file_identifier=b'TFL3')
    return bytes(builder.Output())

class VocabularyAuditTests(unittest.TestCase):
    def test_metadata_section_does_not_steal_the_next_model_section_id(self):
        peek=''.join(f'Section {index}:\n  Items:\n    Key: example\n  Begin Offset: {start}\n  End Offset:   {end}\n  Data Type:    {kind}\n\n'
                     for index,start,end,kind in [(0,128,256,'LlmMetadataProto'),
                                                 (3,1024,2048,'TFLiteModel'),
                                                 (4,4096,8192,'TFLiteModel')])
        self.assertEqual(parse_tflite_sections(peek),[(3,1024,2048),(4,4096,8192)])

    def test_int4_head_cannot_pass_an_int8_head_declaration(self):
        data=fixture(s.TensorType.INT4)
        result=inspect_section(data,0,len(data),0,{'lm_head_int8':True})
        self.assertEqual(result['vocab_weight_types']['lm_head'],['INT4'])
        self.assertIn('lm_head: expected INT8, stored INT4',result['precision_mismatches'])

    def test_renamed_output_still_identifies_a_correct_int8_head(self):
        data=fixture(s.TensorType.INT8)
        result=inspect_section(data,0,len(data),0,{'lm_head_int8':True})
        self.assertEqual(result['vocab_weight_types']['lm_head'],['INT8'])
        self.assertEqual(result['precision_mismatches'],[])

if __name__=='__main__':unittest.main()
