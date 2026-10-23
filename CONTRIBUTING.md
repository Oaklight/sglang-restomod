# Contributing

Thanks for helping. This repository is a **lightweight layer**, not a fork. The
most valuable contribution is a small, well-sourced component.

## Before you start

Open an issue describing:

- the need (model, hardware, workload);
- the source project, commit, and license;
- the minimal files and registration points;
- how you will test it.

## Development setup

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install -e . --no-deps
```

`--no-deps` keeps SGLang out of the development environment. Install
`sglang==0.5.22` separately if you need to run the server.

## Checks

Run these before opening a PR:

```bash
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

## Patch lifecycle

Statuses: `pending`, `accepted`, `local-only`, `rejected`.

- When upstream accepts a patch, set `upstream_status: accepted` and record
  `upstream_pr` / `accepted_in`.
- Do **not** delete it yet. On the next pin bump, `scripts/retire_patches.py`
  checks whether the new upstream source already contains it via
  `git apply --check --reverse`; it deletes only patches that reverse-apply.
- `restomod-patch-guard` fails on an accepted patch that is still present and
  already reverse-applies.

## Branches and commits

- Branch from `main`: `feat/...`, `fix/...`, or `patch/...`.
- PR to `main`; `restomod-ci` and `restomod-patch-guard` must pass.
- Sign off commits (`git commit -s`) when DCO applies.
- Do not force-push `main`; do not move `upstream` manually.

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
