# Architecture

## The model

`main` is a layer, not a fork:

```
main
├── pyproject.toml        sglang-restomod; depends on the pinned sglang
├── sglang_restomod/      plugin and selected components
├── patches/              only the upstream files we choose to change
├── recipes/              validated launch profiles
└── scripts/              fetch / patch / build / check / bump
```

The upstream tree is **not** committed to `main`. `scripts/fetch_upstream.sh`
fetches it at the pin into `.build/sglang` (gitignored) when a build needs the
source. `scripts/build.sh` installs the pinned wheel when the patch queue is
empty, or the patched source when it is not.

The `upstream` branch in this repository is a pristine reference tree at the pin.
It exists for inspection and for CI; `main` never merges it.

## Two ways to add a component

1. **Additive (preferred).** Put the code under `sglang_restomod/` and register
   through SGLang's public plugin entry point (`sglang.srt.plugins`). No patch.
2. **Patch (only when unavoidable).** Add a minimal `patches/NNNN-*.patch` and
   a manifest entry. Registration-point edits are preferred over rewriting
   upstream logic.

Every component PR records its source repository and commit, license, touched
paths, and a test or recipe. Do not copy a whole project's overlay; select the
piece you need.

## Component selection

Start from a concrete need, for example "V100 attention backend for
Qwen3.8-27B". Then:

1. Find the smallest implementation among the candidate sources
   (`heislera763/sglang-v100-plus`, `haohervchb/sglang-V100`,
   `dg1kjd/sglang-sxm2`, and their dependencies).
2. Split it into new files and registration points.
3. Add the new files under `sglang_restomod/`, with a provenance header naming
   the source and commit.
4. If a registration point must change, add a patch and record it.
5. Add a test or a recipe, and update `COMPATIBILITY.md` only with a recorded
   run.

## Patch lifecycle

Statuses: `pending`, `accepted`, `local-only`, `rejected`.

A patch is retired **on the pin bump that first contains the upstream change**,
not when the upstream PR merges. The pin is the source of truth; deleting a
patch earlier would silently remove a fix that the pinned source does not have.

1. When upstream merges the change, set `upstream_status: accepted` and record
   `upstream_pr` / `accepted_in` in `patches/manifest.json`.
2. On the next bump, `scripts/retire_patches.py` runs
   `git apply --check --reverse` for each accepted patch against the new source.
   A successful reverse check means upstream already contains the change; the
   patch file and manifest entry are deleted in the bump PR.
3. A patch that does not reverse-apply is still needed; keep it, rebase it, and
   leave it `accepted` until a later pin.
4. `restomod-patch-guard` fails if an accepted patch is still present and
   already reverse-applies, so a missed retirement cannot sit silently.

`local-only` patches never auto-retire; they need an explicit decision.

## Upstream sync

`restomod-upstream-watch` checks the newest upstream `v*` tag weekly:

- pin at/after the tag: no-op;
- tag ahead: open a `bump/upstream-<tag>` PR that updates the pin, reports
  patch applicability, and retires accepted patches that are now redundant;
- divergent history: open an issue, do not force a merge.

`restomod-pin-sync` moves the `upstream` reference branch to the pin after each
`main` push and tags it `upstream-<tag>`.

## Contributing upstream

This repository is private, so it cannot be the head of a PR against the public
upstream. Create a public launchpad fork when needed
(`gh repo fork sgl-project/sglang --clone=false`), push a clean topic branch
with only the upstreamable change, and open the PR from there. Never push
`main` or `sglang_restomod/` to the launchpad.

## CI

| Workflow | Purpose |
|---|---|
| `restomod-ci` | compile, pin/manifest checks, build + twine check |
| `restomod-patch-guard` | fetch the pin, apply/check patches, enforce retirement |
| `restomod-upstream-watch` | detect new tags, open bump PRs, retire redundant patches |
| `restomod-pin-sync` | move the reference `upstream` branch |
| `restomod-release` | GitHub release on `restomod-v*` tags |
| `restomod-publish` | TestPyPI on dispatch; PyPI on `restomod-v*` tags |

## Publishing

Trusted Publishing (OIDC); no API tokens are stored.

Register a pending publisher in both PyPI and TestPyPI:

| Field | Value |
|---|---|
| PyPI project name | `sglang-restomod` |
| Owner | `Oaklight` |
| Repository | `sglang-restomod` |
| Workflow name | `restomod-publish.yml` |
| Environment | `testpypi` for TestPyPI, `pypi` for PyPI |

- **TestPyPI**: Actions → `restomod-publish` → Run workflow → `testpypi`.
  `skip-existing` makes re-runs safe. Set the optional `version` input to
  publish a test version without a commit, for example `0.0.0a0`.
- **PyPI**: bump `version`, commit, tag `restomod-vX.Y.Z`, push the tag.
  `restomod-release` creates the GitHub release; `restomod-publish` uploads
  the wheel and sdist. The manual `target=pypi` or `target=both` path with a
  `version` override is for test uploads only.
- Example test publish to both channels:

  ```bash
  gh workflow run restomod-publish --ref main \
    -f target=both -f version=0.0.0a0
  ```
- The `testpypi` and `pypi` GitHub environments are already created; they hold
  no secrets.
- If the registry project name differs, change `name` in `pyproject.toml` and
  the publisher settings together.

Upstream SGLang's own workflow tree is not carried here: `main` has no
`.github/workflows` from upstream, and the reference branch is never checked
out for Actions. This keeps a private repository's Actions usage bounded.

## Branch rules

See [BRANCHES.md](BRANCHES.md). While the repository is private on GitHub Free,
rulesets are unavailable; the CI checks and the `upstream` convention apply.
Once public (or with Pro), run `scripts/apply_rulesets.sh` once.
