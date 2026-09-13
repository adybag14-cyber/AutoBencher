#!/usr/bin/env python3
"""Serve a hash-verified Android LiteRT C API runner through a local OpenAI API.

The runner stays resident; each request creates a fresh conversation. Greedy
decoding is enforced so a requested sampling configuration is never ignored.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import queue
import shlex
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def validate_request(data, model_id):
    if not isinstance(data, dict):
        raise ValueError('request must be a JSON object')
    if data.get('model') != model_id:
        raise ValueError('model identity mismatch')
    if data.get('stream') or data.get('n', 1) != 1:
        raise ValueError('only non-streaming n=1 requests are supported')
    if float(data.get('temperature', 0)) != 0 or float(data.get('top_p', 1)) != 1:
        raise ValueError('device parity runner requires greedy decoding')
    for field, expected in [('seed', 42), ('top_k', 1), ('frequency_penalty', 0),
                            ('presence_penalty', 0), ('min_p', 0)]:
        if data.get(field, expected) != expected:
            raise ValueError(f'unsupported {field}; the pinned runner uses {expected}')
    for field in ('tools', 'tool_choice', 'stop', 'logit_bias', 'logprobs'):
        if data.get(field):
            raise ValueError(f'{field} is unsupported by the device parity runner')
    if data.get('response_format', {'type':'text'}) != {'type':'text'}:
        raise ValueError('only text response format is supported')
    budget = data.get('max_completion_tokens', data.get('max_tokens', 2048))
    if type(budget) is not int or not 1 <= budget <= 32768:
        raise ValueError('invalid output token budget')
    messages = data.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
           or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('only text system/user/assistant messages are supported')
    if messages[-1]['role'] != 'user':
        raise ValueError('final message must be a user message')
    template = data.get('chat_template_kwargs', {})
    if not isinstance(template, dict) or set(template) - {'enable_thinking'}:
        raise ValueError('only enable_thinking is supported in chat_template_kwargs')
    thinking = template.get('enable_thinking', False)
    if type(thinking) is not bool:
        raise ValueError('enable_thinking must be boolean')
    return budget, thinking, messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adb', required=True)
    ap.add_argument('--serial', required=True)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--runner', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--backend', choices=['cpu', 'gpu', 'npu'], required=True)
    ap.add_argument('--dispatch-dir')
    ap.add_argument('--port', type=int, default=9396)
    ap.add_argument('--request-timeout', type=int, default=1200)
    ap.add_argument('--evidence', type=pathlib.Path, required=True)
    ap.add_argument('--max-lifetime', type=int, default=28800)
    args = ap.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    if any((args.evidence/name).exists() for name in
           ('owner.json','startup-owner.json','native-stderr.log','native-stdout.log','requests.jsonl')):
        raise ValueError('Choose a new evidence directory; prior device evidence is preserved')
    adb = [args.adb, '-s', args.serial]

    def shell(*parts, timeout=120):
        return subprocess.run(adb + ['shell', shlex.join(parts)], capture_output=True,
                              check=True, timeout=timeout)

    actual = shell('sha256sum', args.model_path).stdout.decode().split()[0]
    if actual != args.sha256:
        raise RuntimeError(f'Device artifact SHA mismatch: {actual}')
    shell('mkdir', '-p', args.cache_dir)
    remote = [args.runner, args.model_path, args.backend, args.cache_dir]
    if args.dispatch_dir:
        remote.append(args.dispatch_dir)
    stderr = (args.evidence / 'native-stderr.log').open('w', encoding='utf-8')
    child = subprocess.Popen(adb + ['shell', 'echo START $$; exec ' + shlex.join(remote)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                             text=True, encoding='utf-8', bufsize=1,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    events = queue.Queue()

    def reader():
        with (args.evidence / 'native-stdout.log').open('w', encoding='utf-8') as log:
            for line in child.stdout:
                log.write(line); log.flush()
                if line.startswith(('START ', 'READY ', 'RESULT ')):
                    events.put(line)
        events.put(None)

    threading.Thread(target=reader, daemon=True).start()
    launch_pid = None
    launch_ticks = None
    try:
        launch = events.get(timeout=15)
        if launch is None or not launch.startswith('START '):
            raise RuntimeError('Device launcher did not report its identity')
        launch_pid = int(launch[6:])
        launch_stat = shell('cat', f'/proc/{launch_pid}/stat', timeout=10).stdout.decode()
        launch_ticks = launch_stat.rsplit(')',1)[1].split()[19]
        (args.evidence/'startup-owner.json').write_text(json.dumps(
            {'pid':launch_pid,'start_ticks':launch_ticks,'remote_command':remote},indent=2))
        first = events.get(timeout=180)
        if first is None or not first.startswith('READY '):
            raise RuntimeError('Device engine failed to initialize; inspect native logs')
    except Exception:
        # The shell exec preserves its PID and start time. A timed-out engine
        # can therefore be stopped before it has emitted READY. Match both its
        # identity and the exact model argument; never kill a reused PID.
        if launch_pid is not None and launch_ticks is not None:
            try:
                stat = shell('cat',f'/proc/{launch_pid}/stat',timeout=10).stdout.decode()
                cmd = shell('cat',f'/proc/{launch_pid}/cmdline',timeout=10).stdout.rstrip(b'\0').split(b'\0')
                exe = shell('readlink',f'/proc/{launch_pid}/exe',timeout=10).stdout.decode().strip()
                owned = exe == args.runner or (exe == '/system/bin/app_process64'
                            and b'com.google.ai.edge.litertlm.DeviceRunnerJNI' in cmd)
                if stat.rsplit(')',1)[1].split()[19] == launch_ticks and owned and args.model_path.encode() in cmd:
                    shell('kill','-TERM',str(launch_pid),timeout=10)
            except (subprocess.SubprocessError,OSError):
                pass
        stderr.close()
        raise
    ready = json.loads(first[6:])
    remote_pid = int(ready['pid'])
    native_stat = shell('cat', f'/proc/{remote_pid}/stat', timeout=10).stdout.decode()
    native_start_ticks = native_stat.rsplit(')', 1)[1].split()[19]
    native_command_line = shell('cat', f'/proc/{remote_pid}/cmdline', timeout=10).stdout
    native_executable = shell('readlink', f'/proc/{remote_pid}/exe', timeout=10).stdout.decode().strip()
    shutdown_key = uuid.uuid4().hex
    identity = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'adb_pid': child.pid, 'native_pid': remote_pid, 'serial': args.serial,
                'remote_command': remote, 'model_id': args.model_id, 'sha256': actual,
                'backend': args.backend, 'runner_ready': ready,
                'native_start_ticks': native_start_ticks, 'native_executable': native_executable,
                'native_command_line': native_command_line.decode(errors='replace'),
                'shutdown_key': shutdown_key}
    (args.evidence / 'owner.json').write_text(json.dumps(identity, indent=2))
    guard = threading.Lock()

    def stop_owned_native():
        # Verify the entire NUL-delimited command, not just a reused PID.
        try:
            current = shell('cat', f'/proc/{remote_pid}/cmdline', timeout=10).stdout
            stat = shell('cat', f'/proc/{remote_pid}/stat', timeout=10).stdout.decode()
            executable = shell('readlink', f'/proc/{remote_pid}/exe', timeout=10).stdout.decode().strip()
            if (current == native_command_line and stat.rsplit(')', 1)[1].split()[19] == native_start_ticks
                    and executable == native_executable):
                shell('kill', '-TERM', str(remote_pid), timeout=10)
        except (subprocess.SubprocessError, OSError):
            pass

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *items):
            print(fmt % items, flush=True)

        def reply(self, code, body):
            raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/v1/models':
                self.reply(200, {'object': 'list', 'data': [{'id': args.model_id,
                           'object': 'model', 'owned_by': 'verified-android-device'}]})
            elif self.path == '/health':
                self.reply(200 if child.poll() is None else 503, public_identity)
            else:
                self.reply(404, {'error': 'unknown route'})

        def do_POST(self):
            if self.path == '/shutdown' and self.headers.get('X-Task-Stop-Key') == shutdown_key:
                self.reply(200, {'status': 'shutting_down'})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if self.path != '/v1/chat/completions':
                self.reply(404, {'error': 'unknown route'}); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 2_000_000:
                    raise ValueError('invalid request size')
                data = json.loads(self.rfile.read(size))
                budget, thinking, messages = validate_request(data, args.model_id)
                with guard:
                    if child.poll() is not None:
                        raise RuntimeError('native runner has exited')
                    child.stdin.write(f'{budget} {int(thinking)}\n')
                    native_messages = [{'role': m['role'], 'content': [{'type': 'text', 'text': m['content']}]} for m in messages]
                    child.stdin.write(json.dumps(native_messages[:-1], ensure_ascii=False) + '\n')
                    child.stdin.write(json.dumps(native_messages[-1], ensure_ascii=False) + '\n')
                    child.stdin.flush()
                    line = events.get(timeout=args.request_timeout)
                    if line is None or not line.startswith('RESULT '):
                        raise RuntimeError('native generation failed')
                    result = json.loads(line[7:])
                    body = result['response']
                    content = body.get('content', '')
                    if isinstance(content, list):
                        content = ''.join(x.get('text', '') for x in content if isinstance(x, dict))
                    if not isinstance(content, str):
                        raise RuntimeError('unexpected native response schema')
                    inp, out = result['input_tokens'], result['output_tokens']
                    if inp < 0 or out < 0:
                        raise RuntimeError('native runner did not provide verified token counts')
                    response = {'id': 'android-' + uuid.uuid4().hex, 'object': 'chat.completion',
                        'created': int(time.time()), 'model': args.model_id,
                        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                     'finish_reason': 'length' if out >= budget else 'stop'}],
                        'usage': {'prompt_tokens': inp, 'completion_tokens': out, 'total_tokens': inp + out}}
                    with (args.evidence / 'requests.jsonl').open('a', encoding='utf-8') as f:
                        request_sha = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                        f.write(json.dumps({'response_id': response['id'], 'request_sha256': request_sha, 'max_output_tokens': budget,
                                           'thinking': thinking, 'result': result}) + '\n')
                self.reply(200, response)
            except ValueError as e:
                self.reply(400, {'error': str(e)})
            except queue.Empty:
                stop_owned_native()
                self.reply(504, {'error': 'device deadline exceeded; native runner stopped'})
            except Exception as e:
                self.reply(500, {'error': str(e)})

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    public_identity = {k: v for k, v in identity.items() if k != 'shutdown_key'}
    print(json.dumps({'status': 'ready', 'port': args.port, **public_identity}), flush=True)
    deadline = threading.Timer(args.max_lifetime, server.shutdown)
    deadline.daemon = True
    deadline.start()
    try:
        server.serve_forever()
    finally:
        deadline.cancel()
        if child.poll() is None:
            child.stdin.close()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                stop_owned_native()
        server.server_close()
        stderr.close()


if __name__ == '__main__':
    main()
