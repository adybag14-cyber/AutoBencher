#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT=/home/tdamre/minicpm5-litert-20260910
CLI="$ROOT/.venv/bin/litert-lm"
REPO=/mnt/c/Users/adyba/docker-chatgpt-devbox/workspace/AutoBencher
export HOME="$ROOT/benchmarks_dynv31/runtime"
mkdir -p "$HOME"
echo $$ > "$HOME/server-9391.pid"
for tag in 16k 32k 64k; do
  file="$ROOT/out_dynv31_a/$tag/MiniCPM5-2B-LiteRT-DynV31A-$tag.litertlm"
  if [[ -f "$ROOT/out_dynv31_a/$tag/sha256.txt" ]]; then
    "$CLI" import "$file" "minicpm5-dynv31a-$tag"
  fi
done
exec "$CLI" --config "$REPO/config/litert-dynv31-server.json" serve --host 127.0.0.1 --port 9391
