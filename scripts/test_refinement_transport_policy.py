#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Control-plane tests only; these do not test quantizer math or model quality."""
import copy
import importlib.util
import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

floor_module = load('minicpm5_apply_precision_floor')
publisher = load('publish_minicpm5_experimental')
proxy_module = load('litert_refinement_proxy')


def evidence():
    data = {'status': 'completed', 'family': 'DynV31A', 'near_lossless_demonstrated': False,
        'variants': {}, 'audit': {'evidence_integrity_passed': True, 'errors': [],
        'all_runs_completed': True, 'incomplete_runs': [], 'variants': {}}}
    for tag in ('16k', '32k', '64k'):
        data['variants'][tag] = {'benchmarks': {}}
        data['audit']['variants'][tag] = {'benchmarks': {}}
        for benchmark in ('ifeval', 'gpqa-diamond'):
            data['variants'][tag]['benchmarks'][benchmark] = {'status': 'completed', 'score_percent': 100.0}
            data['audit']['variants'][tag]['benchmarks'][benchmark] = {'score_valid': True}
    return data


class PrecisionFloorTests(unittest.TestCase):
    def setUp(self):
        self.floor = json.loads((HERE.parent / 'config/minicpm5-dynv31a-precision-floor.json').read_text())
        self.cal = {'context_capacity': 16384, 'selection': {'attention_int8_layers': list(range(42)),
            'mlp_int8_layers': [0,5,12,13,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,41],
            'embedding_int8': True, 'lm_head_int8': True}}

    def test_union_has_32_blocks_without_demotions(self):
        result = floor_module.merge(self.cal, self.floor)
        selected = result['selection']['mlp_int8_layers']
        self.assertEqual(len(selected), 32)
        self.assertTrue(set(self.floor['selection']['mlp_int8_layers']).issubset(selected))
        self.assertEqual(result['precision_policy']['additional_mlp_int8_layers'], [0,5,12,13,15,16,17,18,19,20,21,23])

    def test_inputs_are_not_modified(self):
        before = copy.deepcopy((self.cal, self.floor))
        floor_module.merge(self.cal, self.floor)
        self.assertEqual((self.cal, self.floor), before)

    def test_duplicate_layers_are_deduplicated(self):
        self.cal['selection']['mlp_int8_layers'] *= 2
        selected = floor_module.merge(self.cal, self.floor)['selection']['mlp_int8_layers']
        self.assertEqual(selected, sorted(set(selected)))

    def test_invalid_indices_are_rejected(self):
        for value in (-1, 42, True, '5', 5.0):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.cal)
                changed['selection']['mlp_int8_layers'].append(value)
                with self.assertRaises(ValueError):
                    floor_module.merge(changed, self.floor)


class PublicationPolicyTests(unittest.TestCase):
    def test_valid_audited_experiment_is_eligible(self):
        publisher.validate_evidence(evidence())

    def test_missing_audit_is_rejected(self):
        data = evidence(); data.pop('audit')
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_incomplete_matrix_is_rejected(self):
        data = evidence(); data['variants'].pop('64k')
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_wrong_family_is_rejected(self):
        data = evidence(); data['family'] = 'DynV32'
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_integrity_error_is_rejected(self):
        data = evidence(); data['audit']['errors'].append('artifact mismatch')
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_transport_failure_is_not_a_score(self):
        data = evidence(); data['audit']['all_runs_completed'] = False
        data['audit']['incomplete_runs'] = ['64k/gpqa-diamond']
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_losslessness_claim_is_rejected(self):
        data = evidence(); data['near_lossless_demonstrated'] = True
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_failed_run_is_rejected(self):
        data = evidence(); data['variants']['16k']['benchmarks']['ifeval']['status'] = 'failed'
        with self.assertRaises(ValueError): publisher.validate_evidence(data)

    def test_wrong_answer_is_allowed_but_not_hidden(self):
        data = evidence(); data['variants']['16k']['benchmarks']['ifeval']['score_percent'] = 0.0
        publisher.validate_evidence(data)

    def test_missing_score_is_rejected(self):
        data = evidence(); data['variants']['16k']['benchmarks']['ifeval']['score_percent'] = None
        with self.assertRaises(ValueError): publisher.validate_evidence(data)


class Backend(BaseHTTPRequestHandler):
    seen = []
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.seen.append(payload)
        data = json.dumps({'choices': [{'message': {'content': 'test response'}}]}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        proxy_module.Proxy.backend_host = '127.0.0.1'
        proxy_module.Proxy.backend_port = cls.backend.server_port
        proxy_module.Proxy.request_timeout = 5
        cls.proxy = ThreadingHTTPServer(('127.0.0.1', 0), proxy_module.Proxy)
        cls.threads = []
        for server in (cls.backend, cls.proxy):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start(); cls.threads.append(thread)

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown(); cls.backend.shutdown()
        cls.proxy.server_close(); cls.backend.server_close()
        for thread in cls.threads: thread.join(timeout=5)

    def request(self, payload, path='/v1/chat/completions'):
        conn = HTTPConnection('127.0.0.1', self.proxy.server_port, timeout=5)
        try:
            conn.request('POST', path, body=json.dumps(payload), headers={'Content-Type': 'application/json'})
            response = conn.getresponse(); body = response.read()
            return response.status, json.loads(body)
        finally: conn.close()

    def test_legacy_token_allowance_is_mirrored(self):
        code, _ = self.request({'model': 'test', 'max_tokens': 4})
        self.assertEqual(code, 200)
        self.assertEqual(Backend.seen[-1]['max_completion_tokens'], 4)

    def test_conflicting_allowances_are_rejected(self):
        before = len(Backend.seen)
        self.assertEqual(self.request({'max_tokens': 4, 'max_completion_tokens': 8})[0], 400)
        self.assertEqual(len(Backend.seen), before)

    def test_missing_allowance_is_rejected(self):
        self.assertEqual(self.request({'model': 'test'})[0], 400)

    def test_streaming_is_explicitly_rejected(self):
        self.assertEqual(self.request({'max_tokens': 4, 'stream': True})[0], 400)

    def test_unknown_endpoint_is_rejected(self):
        self.assertEqual(self.request({'max_tokens': 4}, '/not-supported')[0], 404)

if __name__ == '__main__': unittest.main(verbosity=2)
