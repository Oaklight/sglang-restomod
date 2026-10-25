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
| GPU host (`lambda1`, `lambda4`, `lambda12`) | the checkout, the GPU | runs the suite, writes evidence, packs the archive |
| Operator host (`cloud.usa3`) | `/root/.ssh/outboard` (private key), `oob` | signs the statement |
| GitHub | `.github/allowed_signers` | verifies and publishes the check |

The key never leaves the operator host. The GPU host only produces data.

`oob` 0.5.0 is installed in the shared conda prefix and works on every lambda
node without a container:

```sh
OOB=/nfs/lambda_stor_01/homes/pding/qlab/env/conda/sglang-v100-py312/bin/oob
```

On the operator host it is `/root/outboard-venv/bin/oob`.

## Operator runbook

### 1. Run the suite on the GPU host

Check out the exact commit under test (the receipt binds the commit and its
tree, not a branch):

```sh
git clone https://github.com/Oaklight/sglang-restomod.git
cd sglang-restomod
git checkout <full-sha>
mkdir -p evidence
# Run the benchmark/recipe under review; keep every log in evidence/.
<command> > evidence/run.log 2>&1
$OOB pack --evidence-dir evidence --out /tmp/evidence.tgz
sha256sum /tmp/evidence.tgz > /tmp/evidence.tgz.sha256
```

Copy `/tmp/evidence.tgz` to the operator host, and upload it to an HTTPS
location the verifier can reach (a release asset, an internal static host, or
any signed public URL).

### 2. Sign on the operator host

```sh
/root/outboard-venv/bin/oob sign \
  --job gpu-bench \
  --commit <full-sha> \
  --result pass \
  --evidence-dir /root/evidence/<sha> \
  --key /root/.ssh/outboard \
  --repo Oaklight/sglang-restomod \
  --claims evidence_url=https://example.invalid/<sha>.tgz \
  --out /root/receipt.md
```

`--claims evidence_url=...` lets the verifier find the archive without a
separate input. If the run failed but was executed correctly, sign
`--result fail`; the check then reports `failure` with the receipt attached,
which is different from `unverified`.

Render a readable comment before posting:

```sh
/root/outboard-venv/bin/oob render --block /root/receipt.md
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
   `ssh-keygen -t ed25519 -N "" -C "gpu operator" -f /root/.ssh/outboard-next`.
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
