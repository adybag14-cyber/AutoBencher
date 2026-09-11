#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded, nonstreaming OpenAI compatibility proxy for long CPU diagnostics."""
from __future__ import annotations
import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class Proxy(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    backend_host = '127.0.0.1'
    backend_port = 9391
    request_timeout = 3600

    def log_message(self, fmt, *args):
        print('[refinement-proxy] ' + fmt % args, flush=True)

    def error_json(self, code, message):
        data = json.dumps({'error': {'message': message, 'type': 'transport_error'}}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def forward(self):
        allowed = (self.command == 'GET' and self.path in ('/v1/models', '/models')) or (
            self.command == 'POST' and self.path in ('/v1/chat/completions', '/chat/completions'))
        if not allowed:
            self.error_json(404, 'Unsupported endpoint')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 32 * 1024 * 1024:
                raise ValueError('Request body exceeds 32 MiB')
            body = self.rfile.read(length) if length else None
            if self.command == 'POST':
                payload = json.loads(body or b'{}')
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object')
                if payload.get('stream'):
                    raise ValueError('This bounded diagnostic proxy supports nonstreaming requests only')
                cap = payload.get('max_completion_tokens', payload.get('max_tokens'))
                if type(cap) is not int or not 1 <= cap <= 65536:
                    raise ValueError('An explicit positive output-token allowance is required')
                if 'max_tokens' in payload and payload['max_tokens'] != cap:
                    raise ValueError('Conflicting output-token allowances')
                payload['max_completion_tokens'] = cap
                body = json.dumps(payload, separators=(',', ':')).encode()
                print(f'[refinement-proxy] model={payload.get("model")} cap={cap}', flush=True)
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            self.error_json(400, str(exc))
            return
        headers = {'Content-Type': 'application/json'}
        if body is not None:
            headers['Content-Length'] = str(len(body))
        conn = http.client.HTTPConnection(self.backend_host, self.backend_port, timeout=self.request_timeout)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError('Backend response exceeds 64 MiB')
            self.send_response(response.status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.log_message('Client disconnected; no model score is inferred from this transport failure')
        except (TimeoutError, OSError, ValueError, http.client.HTTPException) as exc:
            self.error_json(502, f'Backend request failed: {type(exc).__name__}')
        finally:
            conn.close()

    do_GET = forward
    do_POST = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen-port', type=int, default=9392)
    ap.add_argument('--backend', default='http://127.0.0.1:9391')
    ap.add_argument('--request-timeout', type=float, default=3600)
    args = ap.parse_args()
    backend = urlsplit(args.backend)
    if backend.scheme != 'http' or backend.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Only a localhost HTTP backend is permitted')
    if args.request_timeout <= 0:
        raise ValueError('Timeout must be positive')
    Proxy.backend_host, Proxy.backend_port = backend.hostname, backend.port or 80
    Proxy.request_timeout = args.request_timeout
    server = ThreadingHTTPServer(('127.0.0.1', args.listen_port), Proxy)
    print(f'Refinement proxy listening on 127.0.0.1:{args.listen_port}; timeout={args.request_timeout}s', flush=True)
    server.serve_forever()

if __name__ == '__main__':
    main()
