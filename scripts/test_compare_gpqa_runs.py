import copy, json, pathlib, tempfile, unittest
from compare_gpqa_runs import prompt_hash, read_rows, summarize_pairs

class ComparisonTests(unittest.TestCase):
    def rows(self):
        return {i:{'question_sha256':str(i),'target':'A','answer':'A',
                   'correct':1,'input_tokens':10,'output_tokens':5,'truncated':False}
                for i in range(198)}

    def test_paired_regression_counts_and_protocol_mismatch(self):
        ref=self.rows();candidate=copy.deepcopy(ref)
        candidate[5].update(correct=0,answer='B')
        result=summarize_pairs(ref,candidate)
        self.assertEqual(result['regressions'],1)
        self.assertEqual(result['improvements'],0)
        self.assertEqual(result['candidate']['correct'],197)
        self.assertAlmostEqual(result['accuracy_delta_percentage_points'],-100/198)
        candidate[6]['input_tokens']=11
        mismatched=summarize_pairs(ref,candidate)
        self.assertFalse(mismatched['rendered_input_token_counts_match'])
        self.assertEqual(mismatched['input_token_count_mismatches'],[{'id':6,'baseline':10,'candidate':11}])
        candidate[6]['input_tokens']=10
        candidate[6]['question_sha256']='different-question'
        with self.assertRaises(ValueError):summarize_pairs(ref,candidate)
        with self.assertRaises(ValueError):summarize_pairs({0:ref[0]},{0:ref[0]})

    def test_generated_answer_and_message_ids_do_not_change_prompt_identity(self):
        left=[{'id':'one','role':'user','content':'Question'},
              {'role':'assistant','content':'Answer A'}]
        right=[{'id':'two','role':'user','content':'Question'},
               {'role':'assistant','content':'Answer B'}]
        self.assertEqual(prompt_hash(left),prompt_hash(right))
        right[0]['content']='Another question'
        self.assertNotEqual(prompt_hash(left),prompt_hash(right))

    def test_missing_answers_do_not_inflate_agreement(self):
        ref=self.rows();candidate=copy.deepcopy(ref)
        ref[0].update(answer=None,correct=0)
        candidate[0].update(answer=None,correct=0)
        candidate[1].update(answer=None,correct=0)
        result=summarize_pairs(ref,candidate)
        self.assertAlmostEqual(result['answer_agreement'],196/198)
        self.assertEqual(result['both_parsed_questions'],196)
        self.assertEqual(result['both_unparsed_questions'],1)
        self.assertEqual(result['answer_agreement_when_both_parsed'],1)
        for row in ref.values():row.update(answer=None,correct=0)
        result=summarize_pairs(ref,ref)
        self.assertEqual(result['answer_agreement'],0)
        self.assertIsNone(result['answer_agreement_when_both_parsed'])

    def test_jsonl_unicode_line_separator_is_part_of_the_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            directory=root/'gpqa-diamond/reviews/model'
            directory.mkdir(parents=True)
            path=directory/'gpqa_diamond_default.jsonl'
            path.write_text(''.join(json.dumps({'index':i,'text':'alpha\u2028beta\u0085gamma'},
                            ensure_ascii=False)+'\n' for i in range(198)),encoding='utf-8')
            rows,_=read_rows(root,'reviews')
            self.assertEqual(len(rows),198)
            self.assertEqual(rows[0]['text'],'alpha\u2028beta\u0085gamma')

if __name__=='__main__':unittest.main()
