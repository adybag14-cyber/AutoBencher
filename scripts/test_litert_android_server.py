import unittest
from litert_android_server import validate_request


class RequestValidationTest(unittest.TestCase):
    def request(self, **changes):
        return {'model': 'candidate', 'messages': [{'role': 'user', 'content': '2+2?'}], **changes}

    def test_completion_budget_takes_precedence(self):
        n, thinking, _ = validate_request(self.request(max_tokens=10, max_completion_tokens=16), 'candidate')
        self.assertEqual(n, 16)
        self.assertFalse(thinking)

    def test_sampling_and_wrong_model_are_rejected(self):
        for change in ({'model': 'other'}, {'temperature': 1}, {'top_p': .9}, {'stream': True}, {'n': 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(self.request(**change), 'candidate')

    def test_invalid_limits_and_nontext_messages_are_rejected(self):
        for change in ({'max_tokens': 0}, {'max_tokens': True}, {'messages': []},
                       {'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(self.request(**change), 'candidate')

    def test_unsupported_generation_options_are_not_silently_ignored(self):
        for change in ({'seed':1}, {'top_k':40}, {'stop':['STOP']},
                       {'frequency_penalty':.2}, {'response_format':{'type':'json_object'}},
                       {'chat_template_kwargs':{'thinking_budget':10}},
                       {'chat_template_kwargs':None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(self.request(**change), 'candidate')


if __name__ == '__main__':
    unittest.main()
