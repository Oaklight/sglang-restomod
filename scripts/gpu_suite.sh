#!/usr/bin/env bash
# Produce a reproducible evidence bundle for a GPU run.
#
# Run from the repository checkout at the commit under test. This script never
# signs; it writes evidence/ and packs it. Signing happens on the operator host
# (see docs/gpu-receipts.md).
#
# Usage:
#   scripts/gpu_suite.sh --out evidence
#   scripts/gpu_suite.sh --out evidence \
#       --bench-host 127.0.0.1:11435 --model qwen38next-nvfp4 \
#       --model-dir /path/to/checkpoint --contexts 8192,32768
#
# Environment:
#   OOB      path to oob (default: oob from PATH)
#   PYTHON   interpreter with torch (default: python3)
#   GPUS     CUDA_VISIBLE_DEVICES for the probe (default: 0)
#
# --concurrency takes a comma list (for example 1,2,4); the bench runs once per
# value and appends to bench.log, so one bundle carries the single-stream rate
# and what batching buys on the same launch.
#
# The suite runs `pytest -m gpu tests/` and records the result as
# gpu_tests.txt; a failure fails the suite after the bundle is packed. Use
# --skip-gpu-tests on a host without the SM70 environment.
set -euo pipefail

OUT=evidence
BENCH_HOST=""
MODEL=""
MODEL_DIR=""
CONTEXTS="8192,32768"
OUTPUT_TOKENS=128
CONCURRENCY=""
SKIP_GPU_PROBE=0
SKIP_GPU_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --bench-host) BENCH_HOST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --contexts) CONTEXTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --skip-gpu-probe) SKIP_GPU_PROBE=1; shift ;;
    --skip-gpu-tests) SKIP_GPU_TESTS=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OOB="${OOB:-oob}"
PYTHON="${PYTHON:-python3}"
GPUS="${GPUS:-0}"

command -v git >/dev/null 2>&1 || {
  echo "error: git is required; the runtime image may not ship it" >&2
  exit 2
}
git rev-parse --git-dir >/dev/null 2>&1 || {
  echo "error: run this script from a git checkout" >&2
  exit 2
}
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Measure dirt before creating the output directory. Otherwise a default
# in-repo --out marks the run dirty and every receipt needs --allow-dirty.
DIRTY="$(git status --porcelain | wc -l)"
mkdir -p "$OUT"

# Bind the evidence to the exact commit and tree, not the branch.
{
  echo "repo=$(git config --get remote.origin.url 2>/dev/null || true)"
  echo "commit=$(git rev-parse HEAD)"
  echo "tree=$(git rev-parse HEAD^{tree})"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "dirty_files=$DIRTY"
  echo "date=$(date -Is)"
  # Kernel and machine class only: uname -a would print the node name.
  uname -srm
} > "$OUT/commit.txt"

if [[ "$SKIP_GPU_PROBE" != "1" ]]; then
  nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1 || true
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,clocks.current.sm,clocks.max.sm,memory.total,temperature.gpu --format=csv > "$OUT/gpus.csv" 2>&1 || true
  nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" - > "$OUT/runtime.txt" 2>&1 <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("gpu", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("matmul_sum", float((x @ y).float().sum()))
PY
fi

{
  "$PYTHON" -c 'import sglang; print("sglang", sglang.__version__)' 2>&1 || true
  "$PYTHON" -c 'import torch; print("torch", torch.__version__)' 2>&1 || true
  "$OOB" --version 2>&1 || true
  # The launch configuration of the server under test. /server_info covers the
  # topology and the speculative settings; the component list exists only in
  # the launcher's environment, so the operator declares it here.
  echo "plugins=${SGLANG_PLUGINS:-<unset>}"
  echo "components=${SGLANG_RESTOMOD_COMPONENTS:-<unset>}"
} > "$OUT/versions.txt"

if [[ -n "$BENCH_HOST" ]]; then
  if [[ -z "$MODEL" || -z "$MODEL_DIR" ]]; then
    echo "--model and --model-dir are required with --bench-host" >&2
    exit 2
  fi
  # bench.py builds its URL as http://<host>, so accept a scheme here and strip
  # it once for both the probe and the bench.
  SERVER_HOST="${BENCH_HOST#http://}"
  SERVER_HOST="${SERVER_HOST#https://}"
  SERVER_HOST="${SERVER_HOST%/}"
  # Record what the endpoint is actually serving. The bench numbers are only
  # meaningful next to the spec-decoding setting: MTP off and MTP 3/4 differ by
  # 3-4x at 32k, and a bundle without this cannot tell them apart.
  # Write to a dotfile first: a plain redirect would leave a 0-byte
  # server_info.json in the bundle when both endpoints refuse, which reads as
  # "recorded" to anyone listing the archive.
  for endpoint in /server_info /get_server_info; do
    if "$PYTHON" -c 'import json, sys, urllib.request
host, endpoint = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}{endpoint}", timeout=30) as r:
    json.dump(json.load(r), sys.stdout, indent=1, sort_keys=True)
' "$SERVER_HOST" "$endpoint" > "$OUT/.server_info.tmp" 2>"$OUT/.server_info.err"; then
      mv "$OUT/.server_info.tmp" "$OUT/server_info.json"
      rm -f "$OUT/.server_info.err"
      break
    fi
  done
  if [[ ! -f "$OUT/server_info.json" ]]; then
    {
      echo "server_info unavailable from $SERVER_HOST"
      echo "tried: /server_info, /get_server_info"
      head -3 "$OUT/.server_info.err" 2>/dev/null || true
    } > "$OUT/server_info.unavailable.txt"
    rm -f "$OUT/.server_info.tmp" "$OUT/.server_info.err"
  fi

  # A component that fails to register is skipped silently by the plugin
  # loader, and the launch then serves with the engine's own backend under the
  # same flag. The endpoint's own view is the only thing that can tell the two
  # apart, so record it and fail when what served is not what the launch asked
  # a selected component for.
  if [[ -f "$OUT/server_info.json" ]]; then
    check_rc=0
    "$PYTHON" - "$OUT/server_info.json" "${SGLANG_RESTOMOD_COMPONENTS:-}" "$REPO_ROOT" \
      > "$OUT/launch_check.txt" 2>&1 <<'PY' || check_rc=$?
import json
import sys

path, components, repo_root = sys.argv[1], sys.argv[2], sys.argv[3]
# The heredoc inherits the operator's working directory, which is not
# necessarily the checkout, so put the checkout on the path here rather than
# exporting PYTHONPATH, which the inner pytest would inherit too.
sys.path.insert(0, repo_root)
try:
    from sglang_restomod.attention.v100.backend_registry import BACKEND_NAMES
except ImportError as exc:
    print(f"MISMATCH\tBACKEND_NAMES\timportable from {repo_root}\t{exc}")
    sys.exit(1)

with open(path) as fh:
    info = json.load(fh)
selected = {c.strip() for c in components.split(",") if c.strip()}

failures = 0


def report(ok, key, want, got):
    global failures
    failures += not ok
    print(f"{'ok' if ok else 'MISMATCH'}\t{key}\twant={want}\tgot={got}")


def check(key, want, got):
    report(got == want, key, want, got)


launch_command = info.get("launch_command")
argv = launch_command.split() if isinstance(launch_command, str) else None


def unquote(token):
    """Drop one matched pair of quotes: launch_command is argv joined by spaces."""
    if len(token) > 1 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def requested(name):
    """The value the launch passed for a flag, or None when it passed none."""
    for i, token in enumerate(argv or ()):
        if token == name:
            return unquote(argv[i + 1]) if i + 1 < len(argv) else None
        if token.startswith(name + "="):
            return unquote(token.split("=", 1)[1])
    return None


def attention_v100():
    want = requested("--attention-backend")
    got = info.get("attention_backend")
    if want is None:
        # The component installs choices, not a default: it promises only that
        # one of its names served, not which one.
        report(got in BACKEND_NAMES, "attention_backend", " or ".join(BACKEND_NAMES), got)
    elif want in BACKEND_NAMES:
        check("attention_backend", want, got)
    else:
        report(False, "--attention-backend", " or ".join(BACKEND_NAMES), want)


def lsa_sm70():
    check("attention_backend", "lsa_sm70", info.get("attention_backend"))


def spec_sm70():
    # Installing the draft override changes nothing unless the launch asks for
    # it, so only a launch that asks can be checked.
    want = requested("--speculative-draft-attention-backend")
    got = info.get("speculative_draft_attention_backend")
    if want is not None:
        check("speculative_draft_attention_backend", want, got)
    else:
        print(f"ok\tspeculative_draft_attention_backend\tnot requested by the launch\tgot={got}")


# One entry per component whose effect reaches server_info, and the only place
# this script names a component. A selection with none of them has nothing to
# compare.
CHECKS = {
    "attention_v100": attention_v100,
    "lsa_sm70": lsa_sm70,
    "spec_sm70": spec_sm70,
}

if not selected & CHECKS.keys():
    print("ok\tno component in the selection maps to a server_info field")
elif argv is None:
    # Not every server_info carries the launch, and without it every row below
    # would compare against a guess and print ok. Fail instead.
    report(False, "launch_command", "present in server_info", "missing")
else:
    for component, check_component in CHECKS.items():
        if component in selected:
            check_component()
sys.exit(1 if failures else 0)
PY
    if [[ "$check_rc" -ne 0 ]]; then
      echo "error: the serving configuration does not match SGLANG_RESTOMOD_COMPONENTS; see $OUT/launch_check.txt" >&2
      exit 1
    fi
  fi

  if [[ -n "$CONCURRENCY" ]]; then
    CONCURRENCY_LIST="$CONCURRENCY"
  else
    CONCURRENCY_LIST="1"
  fi
  : > "$OUT/bench.log"
  for c in ${CONCURRENCY_LIST//,/ }; do
    echo "== concurrency $c" >> "$OUT/bench.log"
    # A cold prefix only at one request. Above that, every stream shares the
    # prefix so the batch starts together and the all-active window is real;
    # with cold prefills the later streams are still prefilling long after the
    # first one decodes, and the rate collapses into the prefill stagger.
    # bench.py now salts by default, so the shared case has to opt out.
    FRESH="--no-fresh"
    if [[ "$c" == "1" ]]; then
      FRESH="--fresh"
    fi
    "$PYTHON" scripts/bench.py \
      --host "$SERVER_HOST" \
      --model "$MODEL" \
      --model-dir "$MODEL_DIR" \
      --contexts "$CONTEXTS" \
      --output-tokens "$OUTPUT_TOKENS" \
      --concurrency "$c" \
      --chat --no-think $FRESH --reps 2 --warmups 1 \
      >> "$OUT/bench.log" 2>&1 || echo "bench failed for concurrency $c; see bench.log" >> "$OUT/bench.log"
  done
fi

# Component tests that need a card. The bench measures speed, the launch check
# measures routing, and only these measure whether a kernel still computes what
# it claims. Recorded even when they fail, so the bundle explains a failed run.
GPU_TEST_RC=0
if [[ "$SKIP_GPU_TESTS" != "1" ]]; then
  "$PYTHON" -m pytest -q -m gpu tests/ > "$OUT/gpu_tests.txt" 2>&1 || GPU_TEST_RC=$?
fi

# The archive goes to a public release asset. Strip paths under a home or site
# mount, NVIDIA serials, UUIDs and bus ids, and any IDENTITY_DENYLIST name
# before anything is packed.
"$PYTHON" scripts/redact_evidence.py "$OUT"

PACK="${OUT%/}.tgz"
"$OOB" pack --evidence-dir "$OUT" --out "$PACK"
echo "evidence: $PACK"
if [[ "$GPU_TEST_RC" -ne 0 ]]; then
  echo "error: a GPU test failed; see $OUT/gpu_tests.txt" >&2
  exit 1
fi
if [[ "$DIRTY" != "0" ]]; then
  echo "warning: worktree is dirty ($DIRTY files); sign with --allow-dirty or clean it first" >&2
fi
