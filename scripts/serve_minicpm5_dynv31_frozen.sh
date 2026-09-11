#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve only the frozen copies already imported for this experiment.
set -euo pipefail
ROOT=/home/tdamre/minicpm5-litert-20260910
CLI="$ROOT/.venv/bin/litert-lm"
REPO=/mnt/c/Users/adyba/docker-chatgpt-devbox/workspace/AutoBencher
export HOME="$ROOT/benchmarks_dynv31/runtime"
PIDFILE="$HOME/server-9391.pid"
if [[ -f "$PIDFILE" ]]; then
  OLD="$(cat "$PIDFILE")"
  if [[ "$OLD" =~ ^[0-9]+$ && -f "/proc/$OLD/cmdline" ]]; then
    CMD="$(tr '\0' ' ' < "/proc/$OLD/cmdline")"
    if [[ "$CMD" == *"$CLI"* && "$CMD" == *"--port 9391"* ]]; then
      kill -TERM "$OLD"
      for attempt in {1..100}; do
        [[ ! -d "/proc/$OLD" ]] && break
        sleep 0.1
      done
      if kill -0 "$OLD" 2>/dev/null; then
        echo 'Prior isolated server did not stop; refusing to overlap inference.' >&2
        exit 3
      fi
    fi
  fi
fi
for tag in 16k 32k 64k; do
  test -f "$HOME/.litert-lm/models/minicpm5-dynv31a-$tag/model.litertlm"
done
echo $$ > "$PIDFILE"
exec "$CLI" --config "$REPO/config/litert-dynv31-server.json" serve --host 127.0.0.1 --port 9391
