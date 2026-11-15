#!/usr/bin/env bash
# Run the GPU-marked tests on a host that has the SM70 environment.
#
#   scripts/test_gpu.sh                 # all gpu tests
#   scripts/test_gpu.sh tests/test_sm70_fp8_moe.py -x
#
# Requires the interpreter that has torch and sglang; set PYTHON to override.
set -euo pipefail
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m pytest -q -m gpu "$@"
