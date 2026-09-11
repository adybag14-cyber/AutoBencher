#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Independently export one calibrated context capacity. Existing artifacts are never overwritten.
set -euo pipefail
CTX="${1:?Usage: build_minicpm5_refinement.sh 16384|32768|65536}"
case "$CTX" in 16384) TAG=16k;; 32768) TAG=32k;; 65536) TAG=64k;; *) echo 'Unsupported context' >&2; exit 2;; esac
ROOT=/home/tdamre/minicpm5-litert-20260910
HOLD="$ROOT/dynv32/export_scheduler_gate.txt"
if [[ "$CTX" != 16384 && -f "$HOLD" && "$(cat "$HOLD")" == hold ]]; then
  echo "DYNV32_${TAG}_DEFERRED_AT_CONTEXT_BOUNDARY_FOR_BENCHMARKS" >&2
  exit 75
fi
WORK=/mnt/c/Users/adyba/docker-chatgpt-devbox/workspace
REPO="$WORK/AutoBencher"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/out_dynv32/$TAG"
MAP="$ROOT/dynv32/sensitivity_$CTX.json"
FINAL="$OUT/MiniCPM5-2B-LiteRT-DynV32-$TAG.litertlm"
test -f "$MAP" || { echo "Calibration missing: $MAP" >&2; exit 3; }
if [[ -f "$OUT/sha256.txt" ]]; then
  sha256sum --check "$OUT/sha256.txt"
  echo "Existing verified export retained: $FINAL"
  exit 0
fi
if [[ -e "$OUT/raw" ]]; then
  echo "Partial export exists at $OUT/raw; inspect before choosing a new output directory." >&2
  exit 4
fi
mkdir -p "$OUT"
"$PY" "$REPO/scripts/minicpm5_apply_precision_floor.py" --calibration "$MAP" \
  --floor "$REPO/config/minicpm5-dynv31a-precision-floor.json" --out "$OUT/sensitivity.json"
sha256sum "$OUT/sensitivity.json" > "$OUT/sensitivity.sha256"
cd "$ROOT/src/hf-to-litertlm"
export CACHE="$CTX" PREFILL=1024,256,64,16,4,1 EXTERNALIZE_EMBEDDER=1 USE_JINJA=1
export DYNV3_SENSITIVITY="$OUT/sensitivity.json"
"$PY" "$WORK/export_minicpm_dynv31.py" "$ROOT/model/MiniCPM5-2B" "$OUT/raw" \
  "$ROOT/model/MiniCPM5-2B/chat_template.jinja" MINICPM_DYNV3 > "$OUT/export.log" 2>&1
"$PY" "$ROOT/src/hf-to-litertlm/minicpm_work/fix_zero_scales_inplace.py" \
  "$OUT/raw/model.litertlm" "$FINAL" > "$OUT/zero_scale_fix.log" 2>&1
sha256sum "$FINAL" | tee "$OUT/sha256.txt"
stat -c '%s' "$FINAL" | tee "$OUT/size.txt"
echo "DYNV32_${TAG}_EXPORTED_NOT_YET_QUALITY_ACCEPTED"
