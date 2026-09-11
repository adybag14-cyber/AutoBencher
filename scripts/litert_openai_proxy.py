#!/usr/bin/env python3
"""OpenAI compatibility shim for LiteRT-LM.

EvalScope/OpenAI-compatible clients may still send `max_tokens`, while
LiteRT-LM 0.17 reads `max_completion_tokens`. This proxy preserves the
request and mirrors the legacy field to the LiteRT field when needed.
"""
from __future__ import annotations

import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backend_host = "127.0.0.1"
    backend_port = 9379

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[litert-proxy] {self.address_string()} {fmt % args}", flush=True)

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        if body and self.path.endswith("/chat/completions"):
            try:
                payload = json.loads(body)
                if "max_tokens" in payload and "max_completion_tokens" not in payload:
                    payload["max_completion_tokens"] = payload["max_tokens"]
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                print(
                    "[litert-proxy] chat model=%s max_tokens=%s max_completion_tokens=%s"
                    % (
                        payload.get("model"),
                        payload.get("max_tokens"),
                        payload.get("max_completion_tokens"),
                    ),
                    flush=True,
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        headers = {}
        for key in ("Content-Type", "Authorization", "Accept"):
            if key in self.headers:
                headers[key] = self.headers[key]
        if body:
            headers["Content-Length"] = str(len(body))

        conn = http.client.HTTPConnection(self.backend_host, self.backend_port, timeout=1800)
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            response = conn.getresponse()
            data = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "transfer-encoding", "content-length"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)
        finally:
            conn.close()

    do_GET = _forward
    do_POST = _forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=9380)
    parser.add_argument("--backend", default="http://127.0.0.1:9379")
    args = parser.parse_args()
    backend = urlsplit(args.backend)
    ProxyHandler.backend_host = backend.hostname or "127.0.0.1"
    ProxyHandler.backend_port = backend.port or 80
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    print(
        f"LiteRT OpenAI compatibility proxy listening on {args.listen_host}:{args.listen_port} "
        f"-> {ProxyHandler.backend_host}:{ProxyHandler.backend_port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
