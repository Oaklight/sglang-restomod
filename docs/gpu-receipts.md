# GPU receipts

SGLang on V100 cannot be verified on a GitHub-hosted runner. This repository
accepts a signed receipt produced where the hardware is, and publishes it as
the `gpu-receipt` check.

The mechanism is [out-of-band](https://github.com/Oaklight/out-of-band). The
consumer side lives in `.github/workflows/restomod-gpu-receipt.yml`; the
operator key's public half is in `.github/allowed_signers`.

## What a receipt proves

That the holder of the listed key signed these exact bytes, binding one
repository, commit, tree, job, result and evidence digest. It does not prove
that the job ran correctly or that the host was uncompromised. Treat it as an
attestation, not a proof of correctness.

The gate has four states and never collapses them:

| Receipt | Verdict | Check |
|---|---|---|
| valid, `result: pass` | verified | success |
| valid, `result: fail` | failed | failure |
| absent or invalid | unverified | failure |
| not required by policy | not applicable | success |

A missing receipt is `unverified` (failure), never a skip.

## Roles

| Host | Holds | Does |
|---|---|---|
| GPU host | the checkout, the GPU | runs the suite, writes evidence, packs the archive |
| Operator host | the private key, `oob` | signs the statement |
| GitHub | `.github/allowed_signers` | verifies and publishes the check |

The key never leaves the operator host. The GPU host only produces data.

`oob` 0.5.0 is installed in the shared conda prefix and works on every GPU
node without a container:

```sh
OOB=/path/to/conda/bin/oob
```

On the operator host it is a dedicated venv's `oob`.

## Operator runbook

### 1. Run the suite on the GPU host

Check out the exact commit under test (the receipt binds the commit and its
tree, not a branch):

```sh
git clone https://github.com/Oaklight/sglang-restomod.git
cd sglang-restomod
git checkout <full-sha>

# Standard bundle: commit/tree, GPU inventory, clocks, topology, runtime,
# versions; add a cold-prefix benchmark when an endpoint is already serving.
scripts/gpu_suite.sh --out evidence \
  --bench-host 127.0.0.1:11435 --model qwen38next-nvfp4 \
  --model-dir /path/to/checkpoint --contexts 8192,32768

# Manual fallback: keep every log in evidence/ and pack it yourself.
$OOB pack --evidence-dir evidence --out /tmp/evidence.tgz
```

The script records `dirty_files`; sign with `--allow-dirty` only when that
number is deliberate, because the receipt binds the commit's tree, not the
worktree.

With `--bench-host`, the suite also stores the endpoint's `/server_info` as
`server_info.json`. If neither `/server_info` nor `/get_server_info` answers,
the bundle gets `server_info.unavailable.txt` instead and no
`server_info.json`: a zero-byte file would read as "recorded". A benchmark row is only meaningful next to the serving
configuration: MTP off and MTP 3/4 differ by 3-4x at 32k on the 27B, so a bundle
without `speculative_algorithm`, `speculative_num_steps` and
`speculative_num_draft_tokens` cannot tell two rows apart. For a spec-decoding
regression check, bench at 8k and 32k against the launch under test and read the
32k decode rate against the recipe's expected floor.

The suite also compares the declared component list against that
`server_info.json` and writes `launch_check.txt`: the components whose
effect reaches a server argument (`attention_v100`, `spec_sm70`, `lsa_sm70`)
are checked by value, and a mismatch stops the run before the bench. A component that raises while registering is skipped by the loader and
the launch then serves under the same flag with the engine's own backend, which
is the failure this catches.

`--concurrency` takes a comma list (`--concurrency 1,2,4`). The bench runs
once per value and appends to `bench.log`, so one bundle carries the
single-stream rate and what batching buys on the same launch. Without it the
bench runs at concurrency 1, as before.

The suite also runs the GPU-marked component tests (`pytest -m gpu tests/`) and
records them as `gpu_tests.txt`. The bench measures speed and the launch check
measures routing; these are the only step that measures whether a kernel still
computes what it claims. A failure fails the suite after the bundle is packed,
so the evidence still explains it. `--skip-gpu-tests` exists for a host without
the SM70 environment, and a bundle made with it says so by its absence.

The suite inherits the environment, and `versions.txt` is written by the
interpreter it runs, not by the server it benchmarks. Export `PYTHONPATH` to
the engine under test before running it, or the bundle will name whatever
`sglang` that interpreter finds first. A host can carry more than one install
(for example a fork under `/var/tmp`), and then a correct benchmark is
described by the wrong version line.

The bundle is published, so the suite scrubs it before packing.
`scripts/redact_evidence.py` rewrites absolute paths under a home or site
mount, NVIDIA serial numbers, UUIDs and bus ids, the `host=` line, and every
name in `IDENTITY_DENYLIST`. `redaction.txt` records the count per rule and
never the replaced value. Pass the same list the CI check reads:

Export the repository secret `IDENTITY_DENYLIST` before running the suite.
Do not write its value into a shell history, a script, or this repository.

`scripts/check_identity.py` is the same rule set applied to the committed tree.
It fails on a fully-qualified domain name, a home-directory path, an absolute
path under a site mount, a long digit-only token, or a denylisted name.

Bundles published before `evidence-68ccaaf-identity` (2026-11-03) predate the
redaction step. They carry the node name, `uname -a`, GPU serial numbers, UUIDs
and bus ids, and the absolute checkpoint path. Each is bound by the receipt
that cites it, so none were rewritten; read them as disclosing that much about
the host that produced them.

Copy `/tmp/evidence.tgz` to the operator host, and upload it to an HTTPS
location the verifier can reach (a release asset, an internal static host, or
any signed public URL).

### 2. Sign on the operator host

```sh
oob sign \
  --job gpu-bench \
  --commit <full-sha> \
  --result pass \
  --evidence-dir evidence/<sha> \
  --key <operator-key> \
  --repo Oaklight/sglang-restomod \
  --claims evidence_url=https://example.invalid/<sha>.tgz \
  --out receipt.md
```

`--claims evidence_url=...` lets the verifier find the archive without a
separate input. If the run failed but was executed correctly, sign
`--result fail`; the check then reports `failure` with the receipt attached,
which is different from `unverified`.

Render a readable comment before posting:

```sh
oob render --block receipt.md
```

### 3. Deliver the receipt

Either post the rendered comment on the pull request, or dispatch the workflow
with `commit`, `receipt`, and `evidence-url`. The workflow accepts a comment
only from a trusted author (`trusted-authors: Oaklight`), and only when the
body contains the `<!-- receipt:v1 -->` marker.

A manual dispatch may set `require-evidence=false` for a digest-only dry run.
Pull request and comment events always require the archive.

## Policy

- Pull requests that touch `sglang_restomod/`, `patches/`, or `recipes/`
  require a receipt.
- Documentation-only pull requests pass as `not applicable`.
- A comment or a manual dispatch is an explicit request to verify, so it is
  always required.
- To make the gate blocking, add `gpu-receipt` to the branch protection rules
  for `main` after this workflow has landed on the default branch (the verifier
  reads `.github/allowed_signers` from the default branch).

## Key rotation

1. Generate a new ed25519 key on the operator host:
   `ssh-keygen -t ed25519 -N "" -C "gpu operator" -f <key-file>`.
2. Add its public half to `.github/allowed_signers` as a second line.
3. Switch signing to the new key and verify one receipt.
4. Remove the old line in a follow-up commit.

Keep both lines during the overlap; a receipt signed before rotation stays
verifiable until its line is removed.

## Current scope

The first receipted job is `gpu-bench`: the V100 benchmark matrix for the
component under review. As components land, the job name stays stable and the
suite inside `evidence/` grows.

See also: `COMPATIBILITY.md` for which rows have a recorded run, and
`CONTRIBUTING.md` for the component-selection process.
