# Self-hosted GPU runner: provisioning protocol

This page states the protocol for bringing a GPU host into the CI as a
self-hosted runner. It is generic on purpose: every concrete value is supplied
out of band by whoever operates the hardware, and none of them belongs in this
repository. An example in this file that named a real host, account or token
would be the leak the protocol exists to prevent.

## Before anything else

Clear the deployment with whoever operates the hardware. Ask three questions
and record the answers outside this repository:

1. Is a long-lived outbound daemon permitted on this class of node?
2. In which windows may it run?
3. Does this use count against the allocation?

Nothing below overrides those answers. If any answer is no, the runner does not
go up.

## Protocol

1. **Run as an unprivileged identity.** Do not use the account that holds the
   operator key or the model checkpoints. Where the host allows it, run inside
   a container or sandbox that does not mount the operator's home directory.
2. **Mint the registration token per session.** The token goes to the command
   line and nowhere else: not to a file, not to shell history, not to this
   repository. Let it expire when the window closes.
3. **Start the runner only for the window** in which a job should run.
4. **Label by capability, never by machine.** The labels describe what the host
   can do, not which host it is. `runs-on: [self-hosted, v100, gpu-2]` is the
   shape; a label that names a machine is an identity leak.
5. **At the end of the window**, remove the runner and stop the process:

   ```sh
   "$RUNNER_DIR/config.sh" remove --token "$TOKEN"
   "$RUNNER_DIR/run.sh" --stop || true
   ```

## Values

| Variable | Meaning | Source |
|---|---|---|
| `$RUNNER_DIR` | where the runner action is unpacked | operator |
| `$WORK_DIR` | where jobs check out and write | operator |
| `$TOKEN` | registration token for one window | minted per session |
| `$LABELS` | capability labels for this host class | operator |

All four are supplied out of band. This repository never records them.

## What the job may assume

- The checkout is at the commit the workflow names; the job does not fetch
  branches.
- The GPU and its driver are present; the workflow does not install them.
- The container image, model checkpoints and the SM70 library are provided by
  the operator through the job's environment, as `scripts/gpu_suite.sh`
  documents.

## What the job must not do

- No `actions/cache` restore: a cache written by a hosted job would be code
  this host executes.
- No artifact download from a run this job did not produce.
- No `env` dump, no `set -x` around environment access, and no hostname or
  user-name lookup. See `.github/workflows/restomod-gpu-receipt.yml` for the
  workflow that follows this rule.

## Related

- Issue #12 covers the workflow-side gates (trigger, permissions, environment
  approval). `.github/workflows/restomod-gpu.yml` is the workflow, and
  `scripts/check_workflow_gates.py` fails `restomod-ci` when it gains a
  pull-request trigger, a `secrets.` reference, a cache restore, an unpinned
  action, or loses its environment gate.
- Issue #13 covers this protocol and its audit.
- `docs/gpu-receipts.md` covers what a job produces and how it is signed.
