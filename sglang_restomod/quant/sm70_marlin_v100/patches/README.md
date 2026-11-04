# marlin_v100 patches

These patches apply to the `zhinianqin/marlin_v100` source tree, not to the
pinned sglang source. They therefore live here, next to the component that
loads the built `_C` extension, and not in the top-level `patches/` queue,
which `scripts/apply_patches.sh` applies to `.build/sglang`.

Apply before building the extension:

```bash
git -C /path/to/marlin_v100 apply \
  /path/to/sglang_restomod/quant/sm70_marlin_v100/patches/0001-splitk-from-occupancy.patch
```

## 0001-splitk-from-occupancy.patch

### What it changes

`sm70_marlin_dense_auto_params` chose a launch geometry from `size_n` alone and
left `requested_split_k = 1`. One CTA computes one output tile, so a tall-K
projection with few output tiles — `down_proj`, present in every transformer —
runs its whole K loop on 20 of the 80 SMs of a V100.

The patch derives `split_k` from the tile count:

```
tiles = ceil(size_m / cta_m) * ceil(size_n / cta_n)
split = clamp(pow2_floor(192 / tiles), 1, 8)
```

and keeps the K-divisibility fallback (`size_k % (split * cta_k) == 0`).
`192` is two waves of 80 SMs plus slack; the measured crossover sits there.

The explicit environment knob `SM70_MARLIN_DENSE_SPLIT_K` still wins. The
shared env helper and the MoE selector keep `split_k = 1`: they do not carry
the dense problem shape, and the rule was not measured for MoE.

### Measured

One V100-SXM2-32GB, CUDA 12.8, block-FP8 W8A16, correct scale layout
(`relerr` 0.000-0.008 against the TurboMind reference). Marlin microseconds,
`SM70_MARLIN_DENSE_SPLIT_K` swept by hand to find the surface:

| shape (N, K) | M | split_k=1 | best split_k | best | TurboMind |
|---|---:|---:|---:|---:|---:|
| 5120, 17408 | 1 | 601.8 | 8 | 138.8 | 124.7 |
| 5120, 17408 | 6 | 601.7 | 8 | 154.8 | 125.5 |
| 5120, 17408 | 16 | 603.3 | 8 | 168.4 | 134.6 |
| 5120, 17408 | 64 | 606.0 | 4 | 271.2 | 332.2 |
| 5120, 17408 | 256 | 848.3 | 1 | 848.3 | 642.5 |
| 5120, 17408 | 1024 | 3398.1 | 1 | 3398.1 | 2676.8 |
| 17408, 5120 | 1 | 196.3 | 2 | 154.1 | 116.4 |
| 17408, 5120 | 6 | 196.5 | 2 | 167.1 | 116.7 |
| 17408, 5120 | 16 | 199.1 | 2 | 179.3 | 123.1 |
| 17408, 5120 | 64 | 257.9 | 1 | 257.9 | 250.6 |
| 5120, 5120 | 1 | 181.8 | 8 | 83.2 | 42.6 |
| 5120, 5120 | 256 | 246.4 | 1 | 246.4 | 207.7 |
| 6144, 5120 | 1 | 182.8 | 8 | 88.1 | 50.9 |
| 6144, 5120 | 256 | 461.3 | 1 | 461.3 | 290.7 |

The rule picks the measured best at 23 of the 24 cells above; the one miss is
`N=6144, M=64`, where it chooses 4 and the best is 1 (221 us against 185 us,
20% slower). Correcting an earlier estimate in issue #180, which said 19 of 24.

### Verified default path

Built from the patched tree and run with **no** `SM70_MARLIN_DENSE_SPLIT_K` set,
the default reproduces the sweep's best column. Marlin microseconds:

| shape (N, K) | M=1 | M=6 | M=16 | M=64 | M=256 | M=1024 |
|---|---:|---:|---:|---:|---:|---:|
| 5120, 17408 | 139.1 | 154.1 | 168.0 | 269.1 | 854.9 | 3403.8 |
| 17408, 5120 | 153.6 | 167.4 | 179.0 | 256.7 | 954.8 | 3515.6 |
| 5120, 5120 | 66.9 | 72.5 | 87.0 | 123.7 | 246.4 | 1002.7 |
| 6144, 5120 | 87.7 | 115.1 | 133.6 | 199.7 | 458.9 | 1215.9 |

`relerr` against TurboMind stays at 0.000-0.006, so the selector change does not
touch the numerics. One cell now beats TurboMind: `5120, 17408, M=64`, 269.1 us
against 331.8 us (1.23x), which is the prefill regime rather than decode.

The patch does not change the deployment decision. In the decode regime Marlin
stays behind TurboMind on every 27B shape: 0.81x at `5120, 17408, M=6` and
0.59x at `5120, 5120, M=6`. It wins only at `M=64` on the tall-K projection
(1.23x), which is prefill, and `fp8_dense` already covers prefill with
TurboMind. `fp8_dense` stays on
TurboMind. The patch matters because the default was 4.3x off the time the
kernel can reach, and any future Marlin-backed path pays it.

### Status

`upstream_status: pending`. The account has pull-only access to
`zhinianqin/marlin_v100`, so the change is kept here until a fork or an
upstream PR is agreed.

Verified against a fresh clone of the upstream repository: `git apply --check`
passes at `3f16d44`, the commit `quant/sm70_fp4_moe/PROVENANCE.md` pins. It also
passes in reverse against the tree the extension under test was built from.

Trace: restomod issue #180, lab note `notes/marlin-dense-fp8-sm70.md`.
