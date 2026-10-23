# sglang-restomod

A maintained downstream of [SGLang](https://github.com/sgl-project/sglang) for
NVIDIA V100 / SM70 and for a heterogeneous lab fleet. The repository is a
private copy (a mirror, not a GitHub fork) so that the product tree can diverge
without making the fork-network relationship public.

The V100 baseline is imported from
[heislera763/sglang-v100-plus](https://github.com/heislera763/sglang-v100-plus),
which rebuilt the original
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) port on
current mainline. Our own work is additive and lives under `restomod/`.

## Branch model

| Branch | Meaning | Rules |
|---|---|---|
| `upstream` | Pristine mirror at the commit in `UPSTREAM_PIN.toml` | Never commit here; only the pin-sync workflow updates it |
| `main` | `upstream` + restomod overlay + recorded core patches | All development and releases |
| `bump/upstream-*` | Proposed upstream pin updates | Opened by `upstream-watch`; merge after review |
| `upstream-<tag>` | Immutable tag for a historical pin | Created by `pin-sync` |

`main` merges upstream once per pin bump. Do not rebase the fork onto upstream:
the V100 delta is large and a rebase spreads conflict resolution across every
commit. Merge once, resolve at the recorded hook points, then run the checks.

## Layout

```
restomod/                     our overlay; additive code and overrides
  sglang_restomod/            SGLang plugin package (`SGLANG_PLUGINS=restomod`)
v100_plus/                    vendored V100 baseline (overlay + manifest)
  core-patches.json           every edited upstream file and the reason
  provenance.json             upstream pin, third-party origins, hashes
  check-core-diff.py          fails on unrecorded edits under python/sglang
python/sglang/                upstream tree; edit only through recorded patches
```

Enable both plugins:

```bash
export SGLANG_PLUGINS=v100_plus,restomod
```

## Upstream sync rules

1. A periodic `upstream-watch` job checks the newest `v*` tag and opens a
   `bump/upstream-<tag>` PR with a merge dry-run report.
2. A human merges the tag into `main` and resolves conflicts. Prefer resolving
   at the small, recorded registration points; never resolve by dropping an
   upstream change wholesale.
3. Update `UPSTREAM_PIN.toml` and `v100_plus/provenance.json` together
   (`scripts/upstream_bump.py` does both).
4. Run `scripts/check_pin.py` and `v100_plus/check-core-diff.py --ref HEAD`.
   CI runs both on every PR.
5. Tag the new pin as `upstream-<tag>` via `pin-sync`.

## Core patch policy

- Additive files go under `restomod/` or new upstream directories and are free.
- Edits to `python/sglang/**` must be listed in `v100_plus/core-patches.json`
  with a one-line reason. `delta-guard` rejects unrecorded paths.
- A patch that can be upstreamed should be extracted into a clean topic branch
  and sent to `sgl-project/sglang`; the fork then drops its copy.
- Every patch needs an owner and an upstream status. The census may shrink; it
  must not grow silently.

## Contributing upstream

This repository is private, so it cannot be the head of a PR against the public
upstream. Create a public launchpad fork under the same account when needed
(`gh repo fork sgl-project/sglang --clone=false`), push a clean topic branch
containing only the upstreamable change, and open the PR from there. Never push
`main` or `restomod/` to the launchpad.

## Status

- Baseline: `heislera763/sglang-v100-plus@595b5f01` (Apache-2.0), which
  integrates upstream through `0d6b283d`.
- The distribution name in `pyproject.toml` is still `sglang-v100-plus`;
  renaming it to `sglang-restomod` is a separate, lockfile-touching change.
- Build and launch instructions live in the vendored baseline README and
  `v100_plus/setup.sh`. For V100, start with the baseline; our overlay is
  initially a no-op seam.
