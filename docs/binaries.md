# Out-of-repo artefacts

The serving stack depends on files the repository does not contain: prebuilt
libraries, a third-party flashinfer fork, hand-tuned MoE tiles, a container
image. `BINARIES.toml` pins them; `restomod doctor` checks a host against the
pin; `scripts/fetch_binaries.py` and `scripts/build_binaries.py` produce them.

The manifest is **path-free**. Paths live in a site-local file named by
`RESTOMOD_SITE_CONFIG`; `restomod.site.example.toml` shows the shape.

## Why this exists

Each row records what silently breaks without the artefact. The important
ones are the silent failures, not the missing files:

| Missing | What you see |
|---|---|
| `_sm70_turbomind_v100.so` or its path | `fp8_dense`, `fp8_lm_head`, `fp8_dense_quant` register and do nothing; dense runs bf16; every number is a different arm |
| flashinfer commit drift | different kernels; no log line |
| tuned V100 MoE tiles | `Using default MoE kernel config`; slower draft MoE |
| QPN8 source | `fp8_dense_quant` stays on the older decode path; no failure |
| checkout `dist-info` | **no `sglang.srt.plugins` entry point**, so no component loads at all, and the symptom is `--attention-backend: invalid choice` |
| a component's module fails to import | recorded as `inactive` with `import failed: ...`; without that it looks "loaded" and the engine serves its own backend |

## `restomod doctor`

```bash
restomod doctor --site "$RESTOMOD_SITE_CONFIG"
restomod doctor --deep      # also hash the container image
restomod doctor --json
```

It checks:

1. the engine imports and does not resolve to a `/var/tmp` checkout;
2. the `restomod` entry point resolves;
3. no selected component is `inactive`, `failed`, or `missing`, after loading
   the plugin set;
4. the tuned V100 MoE tiles resolve **in the tree `sglang.__file__` points at**;
5. every artefact that has a site path matches its `sha256`;
6. the container image exists, and with `--deep` matches its hash;
7. the versions of torch, triton, tilelang, sglang-kernel, flashinfer, and
   transformers are printed;
8. the restomod tree records the commit it was built from (git `HEAD` or a
   `COMMIT` file) and matches the declared `[engine].commit` or
   `RESTOMOD_EXPECT_COMMIT`.

It exits non-zero with a named reason. It imports nothing heavy until a check
needs it, and a missing site config is a note, not a crash: the wheel must stay
importable on a host that is not ours.

An **import failure** is recorded as `inactive`, not `failed`: the module never
loaded, but the server is healthy and would otherwise serve its own backend
under the same component list. The doctor prints `import failed: ...` as the
reason and exits non-zero.

## Commit pinning

A deployed tree must name the code it contains: either it is a git checkout
(`git rev-parse HEAD`) or it carries a `COMMIT` file written when it was built.
The site config declares what the run expects:

```toml
[engine]
commit = "<40-hex-commit>"
```

`restomod doctor` fails when the two differ or when the tree records nothing.
A stale tree is an error, not a configuration choice: a site tree that predated
`sglang_restomod/_sm70_paths.py` served bf16 dense linears under the full
component list, and every gate available at the time passed.

## Fetch and build

```bash
python scripts/fetch_binaries.py --check            # verify what is present
python scripts/fetch_binaries.py --run --only flashinfer
python scripts/build_binaries.py --print            # show build commands
python scripts/build_binaries.py --run --only marlin-dense
```

Build commands and source directories come from the site config's `[build]`
table. A new host is `doctor` → `fetch`/`build` → `doctor`.

## Licence rule

Nothing is promoted to `ship` until a licence text is in hand. `license =
"unverified"` means exactly that. The TurboMind library and both marlin
libraries are fetched or built, not redistributed.

## Tracked rows

Two rows are ours and travel with the wheel:

- `sglang_restomod/configs/moe/triton_3_5_1/`: the hand-tuned V100 tiles for
  E=512, N=160. Deployments copy them into the engine's
  `.../moe_runner/triton_utils/configs/triton_3_5_1/` directory.
- `sglang_restomod/quant/sm70_fp8_dense/csrc/sm70_fp8_qpn8_decode.cu`: the
  QPN8 decode source. `SGLANG_SM70_QPN8_SRC` remains an override; the packaged
  copy is the default.

## Checkout metadata

A checkout placed on `PYTHONPATH` does not have distribution metadata, and
SGLang discovers plugins through `importlib.metadata.entry_points`. A
checkout-based deployment therefore needs:

- `sglang-<version>.dist-info` next to the pin tree's `python/` directory;
- `sglang_restomod-<version>.dist-info` next to the restomod checkout, with

```ini
[sglang.srt.plugins]
restomod = sglang_restomod:register
```

Without them `SGLANG_PLUGINS` has nothing to whitelist and a component's
backend name is rejected before any component code runs.
