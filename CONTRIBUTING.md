# Contributing

Thanks for helping. This repository is a **lightweight layer**, not a fork. It
adds components and a patch queue on top of a pinned SGLang, so the most
valuable contribution is a small, well-sourced one.

## Where to read first

- [`README.md`](README.md) — status, measured numbers, and layout.
- [`RESTMOD.md`](RESTMOD.md) — architecture, component selection, patch lifecycle.
- [`BRANCHES.md`](BRANCHES.md) — branch and tag rules.
- [`patches/README.md`](patches/README.md) — adding and retiring a patch.
- [`AGENTS.md`](AGENTS.md) — instructions for coding agents.

## Design decisions

Standing choices that should not be reversed without a new decision record:

- **`main` carries the layer, not SGLang.** The upstream tree is fetched into
  `.build/sglang` only when a patch exists. Never vendor it or merge the
  `upstream` branch into `main`.
- **Additive code goes in `sglang_restomod/` and registers through
  `sglang.srt.plugins`.** An upstream edit is the exception, and it always
  comes with a patch.
- **Every upstream edit needs a `patches/manifest.json` entry** that lists every
  touched path and a one-line reason. Patch status is one of `pending`,
  `accepted`, `local-only`, or `rejected`.
- **Unset or empty `SGLANG_RESTOMOD_COMPONENTS` means baseline.** The plugin
  loads and does nothing.
- **A patch is retired on the pin bump that contains it**, by reverse-apply
  check. Do not delete it when the upstream PR merely merges.

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install -e . --no-deps
```

`--no-deps` keeps SGLang out of the development environment. Install
`sglang==0.5.22` separately if you need to run the server.

## Before you start

Open an issue describing:

- the need (model, hardware, workload);
- the source project, commit, and license;
- the minimal files and registration points;
- how you will test it.

## Checks

Run these before opening a PR:

```bash
cmp README.md README_en.md
python scripts/check_pin.py
python scripts/check_patches.py --skip-apply
python -m compileall -q sglang_restomod scripts
```

When the change touches `patches/`:

```bash
scripts/fetch_upstream.sh
python scripts/check_patches.py
scripts/apply_patches.sh
```

CI runs the same commands in `restomod-ci` and `restomod-patch-guard`.

## Adding a component

1. **Prefer additive code.** Put it under `sglang_restomod/` and register
   through `sglang.srt.plugins`. No upstream edit.
2. **Patch only when unavoidable.** Keep one logical change per patch, touch
   the smallest set of paths, and add a `patches/manifest.json` entry with the
   paths, reason, source, and `upstream_status`.
3. **Record provenance.** Note the source repository and commit in the file
   header or manifest.
4. **Test it.** Add a unit test or a recipe, and update `COMPATIBILITY.md` only
   with a recorded run.
5. **Do not import a whole overlay.** Select the files and registration points
   you need.
6. **Keep import cheap.** No torch, no sglang, and no kernel loading at module
   import; `register()` does no heavy work either. Load on first use.

## Patch lifecycle

Statuses: `pending`, `accepted`, `local-only`, `rejected`.

- When upstream accepts a patch, set `upstream_status: accepted` and record
  `upstream_pr` / `accepted_in`.
- Do **not** delete it yet. On the next pin bump, `scripts/retire_patches.py`
  checks whether the new upstream source already contains it via
  `git apply --check --reverse`; it deletes only patches that reverse-apply.
- `restomod-patch-guard` fails on an accepted patch that is still present and
  already reverse-applies.

## Pin bump

1. Re-pin `UPSTREAM_PIN.toml`, run `scripts/fetch_upstream.sh`, then
   `scripts/apply_patches.sh`. `bump/upstream-*` is opened by CI, not by hand.
2. Run `scripts/retire_patches.py` for accepted patches, and
   `scripts/sync_kernels.py --check` after re-pinning `[kernels].commit`.
3. Re-check the vendored QSA backend against the pin's own file and re-apply
   the SM70 routes on top of it. It has been pin-based rather than fork-based
   since #243. The step recurs because a bump adopts upstream QSA commits
   (#45752 and #45791 were two) whose effect on our routes is not visible in
   the bump's own diff: the file we re-apply onto changes, and nothing fails
   until a route reads text that moved. Details:
   [docs/qsa-sm70-audit.md](docs/qsa-sm70-audit.md).

## Branches and commits

- Branch from `main`: `feat/...`, `fix/...`, or `patch/...`.
- Keep commit messages concise and focused on *why*. One logical change per
  commit.
- PR to `main`; `restomod-ci` and `restomod-patch-guard` must pass.
- Sign off commits (`git commit -s`) when DCO applies.
- Do not force-push `main`; do not move `upstream` manually.

## AI-assisted contributions

Using AI tools (for example Claude, Cursor, or Copilot) to assist is welcome.
These rules are not optional:

- **No AI or agentic co-author tags in commits.** Do not add `Co-authored-by`
  lines for AI tools or agentic systems to a commit message. This keeps the
  history readable and authorship truthful.
- **Disclose in the PR description.** If AI tools did a significant part of the
  work, add one line, for example "AI was used to assist with implementation".
- **You own the code.** Contributors are responsible for everything they
  submit, including AI-generated code: review it, test it, and understand it
  before you commit it.

## Upstream contributions

This repository is not a GitHub fork of SGLang, so it cannot be the head of a
PR against `sgl-project/sglang`. Use a public launchpad fork:

```bash
gh repo fork sgl-project/sglang --clone=false
git checkout -b fix/foo upstream/main
git commit -s
git push git@github.com:<you>/sglang.git fix/foo
gh pr create --repo sgl-project/sglang --head <you>:fix/foo --base main
```

Keep upstream-only commits free of restomod-specific code.

## Release and publishing

- `__version__` in `sglang_restomod/__init__.py` is the only version source;
  `pyproject.toml` reads it dynamically. Bump it in the PR that prepares the
  release.
- TestPyPI: Actions → `test-release` → `version=<x>`.
- PyPI: Actions → `release` → `version=<x>`. It publishes to PyPI, creates
  `restomod-v<version>`, and creates the GitHub Release.
- Both workflows verify the requested version against `__init__.py`; they never
  override it.

## License

By contributing you agree that your contribution is licensed under
Apache-2.0, the same license as this repository.
