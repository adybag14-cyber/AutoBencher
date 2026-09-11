#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for context in 16384 32768 65536; do
  bash "$HERE/build_minicpm5_refinement.sh" "$context"
done
