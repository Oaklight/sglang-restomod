# restomod overlay

This directory is ours. It is deliberately separate from the vendored
`v100_plus/` baseline so upstream syncs do not touch it.

## Rules

1. **Additive first.** New backends, kernels, quant methods, model hooks, and
   launch helpers go here, not into `python/sglang/`.
2. **Register, don't monkey-patch.** Use SGLang's public registries and plugin
   hooks. `sglang.srt.plugins` entry points are the supported seam.
3. **Core edits are recorded.** If an upstream file must change, add it to
   `v100_plus/core-patches.json` with a one-line reason. `delta-guard` fails on
   unrecorded paths under `python/sglang/`.
4. **Name our knobs `RESTMOD_*`.** Do not add new `SGLANG_*` variables unless
   they are being upstreamed.
5. **Remove patches when upstream takes them.** A shrinking census is the goal.

## Plugin entry point

`pyproject.toml` registers:

```toml
[project.entry-points."sglang.srt.plugins"]
v100_plus = "sglang_v100_plus:register"
restomod = "sglang_restomod:register"
```

Load both with:

```bash
export SGLANG_PLUGINS=v100_plus,restomod
```

`register()` is currently a no-op. Put real installations in
`sglang_restomod/overrides.py:install()` and call it from `register()` once the
first override lands.

## When this becomes a separate package

The overlay can be split into `restomod-sglang` later without touching the
engine: it already has its own package directory, entry point, and namespace.
Until then, keep it in this monorepo to avoid version skew with `main`.
