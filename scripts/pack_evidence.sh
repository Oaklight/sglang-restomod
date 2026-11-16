#!/usr/bin/env bash
# pack_evidence.sh -- redact an evidence directory, then pack it for a receipt.
#
# The rule this exists to enforce: the redaction that runs is the one on
# `origin/main`, at the time of packing, with a denylist supplied. The 2026-11-16
# receipts were redacted with whatever copy the packer's checkout held, and three
# published bundles kept the operator account name and a host name because the
# rules that remove them had landed on main after that checkout was cut. Nothing
# failed; the bundles looked clean; the leak was in the release assets.
#
#   IDENTITY_DENYLIST="host-a,host-b,operator" scripts/pack_evidence.sh \
#       <evidence-dir> <out.tgz>
#
# Both the denylist and the redaction script are required. A missing denylist
# cannot catch host names, and a local script may be older than the rules.
#
# Output: a `redaction-script.txt` inside the evidence dir naming the commit and
# the script's sha256, then two clearly labelled numbers from `oob pack`:
#
#   manifest digest: sha256:...      <- what a receipt binds
#   archive sha256:  ...             <- what a downloader can verify
#
# They are different numbers and three summaries in one evening quoted the wrong
# one. Print both, labelled.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: IDENTITY_DENYLIST=name[,name...] scripts/pack_evidence.sh <evidence-dir> <out.tgz>

  <evidence-dir>   directory of evidence to redact and pack, rewritten in place
  <out.tgz>        archive to write

  IDENTITY_DENYLIST   required; comma or newline separated names to remove
  OOB                 oob binary (default: oob on PATH)
  MAIN_REF            git ref the redaction script is taken from (default: origin/main)
USAGE
  exit 2
}

die() { echo "pack_evidence: $*" >&2; exit 2; }

[[ $# -eq 2 ]] || usage
EVIDENCE=$1
OUT=$2
OOB=${OOB:-oob}
MAIN_REF=${MAIN_REF:-origin/main}

# bash's own `${VAR:?}` exits 1 with a bare message; this gate exits 2 and says
# what to do, because a caller that treats 1 as a transient error would retry a
# configuration defect.
[[ -n "${IDENTITY_DENYLIST:-}" ]] ||
  die "IDENTITY_DENYLIST is required; without it host names cannot be removed"

[[ -d "$EVIDENCE" ]] || die "not a directory: $EVIDENCE"
[[ -n "${IDENTITY_DENYLIST//[[:space:],]/}" ]] || die "IDENTITY_DENYLIST is empty"
command -v "$OOB" >/dev/null || die "no oob binary ($OOB); packing without it writes no digest"

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a git checkout"
MAIN_COMMIT=$(git -C "$ROOT" rev-parse --verify --quiet "$MAIN_REF^{commit}") ||
  die "cannot resolve $MAIN_REF in $ROOT; fetch it rather than packing with a local script"

# Fail closed: no silent fall back to the checkout's copy. That fallback is the
# defect this script exists for.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
REDACT=$TMP/redact_evidence.py
git -C "$ROOT" show "$MAIN_REF:scripts/redact_evidence.py" > "$REDACT" 2>/dev/null ||
  die "cannot read scripts/redact_evidence.py at $MAIN_REF"
[[ -s "$REDACT" ]] || die "scripts/redact_evidence.py at $MAIN_REF is empty"
REDACT_SHA=$(sha256sum "$REDACT" | cut -d' ' -f1)

# A record that names a different script means a previous pack used other rules.
# Refuse rather than mix two rule sets in one bundle's history.
RECORD=$EVIDENCE/redaction-script.txt
if [[ -f "$RECORD" ]]; then
  RECORDED=$(sed -n 's/^file sha256: //p' "$RECORD" | head -n 1)
  if [[ -n "$RECORDED" && "$RECORDED" != "$REDACT_SHA" ]]; then
    die "the bundle records redaction script $RECORDED but $MAIN_REF has $REDACT_SHA; start a fresh directory"
  fi
fi

echo "== redacting $EVIDENCE with scripts/redact_evidence.py at $MAIN_REF ($MAIN_COMMIT)"
if ! python3 "$REDACT" "$EVIDENCE" --denylist "$IDENTITY_DENYLIST"; then
  die "redaction failed (residual identity is reported above, by file and pattern name); not packing"
fi

cat > "$RECORD" <<RECORD
redaction script: scripts/redact_evidence.py at $MAIN_REF $MAIN_COMMIT
file sha256: $REDACT_SHA
run from $MAIN_REF, not from the packer checkout
denylist supplied: yes (names are not recorded here)
RECORD

echo "== packing $EVIDENCE -> $OUT"
PACK_JSON=$("$OOB" pack --evidence-dir "$EVIDENCE" --out "$OUT" --json)
python3 - "$PACK_JSON" "$OUT" <<'PY'
import hashlib
import json
import pathlib
import sys

payload = json.loads(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
print(f"files:           {payload['count']}")
print(f"manifest digest: {payload['digest']}")
print(f"archive sha256:  {hashlib.sha256(archive.read_bytes()).hexdigest()}")
PY
