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

### Measured: the full 24-cell grid

One V100-SXM2-32GB, CUDA 12.8, block-FP8 W8A16, correct scale layout
(`relerr` 0.000-0.008 against the TurboMind reference). Marlin microseconds,
`SM70_MARLIN_DENSE_SPLIT_K` swept by hand. `rule` is what the patch picks;
`-` means that split was never run for that cell.

| shape (N, K) | M | rule | sk=1 | sk=2 | sk=4 | sk=8 | best measured | rule is best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 5120, 17408 | 1 | 8 | 601.8 | 320.5 | 179.7 | **138.8** | 138.8 | yes |
| 5120, 17408 | 6 | 8 | 601.7 | 333.0 | 195.5 | **154.8** | 154.8 | yes |
| 5120, 17408 | 16 | 8 | 603.3 | 342.2 | 206.9 | **168.4** | 168.4 | yes |
| 5120, 17408 | 64 | 4 | 606.0 | 353.2 | **271.2** | 313.2 | 271.2 | yes |
| 5120, 17408 | 256 | 1 | **848.3** | 938.5 | 1033.2 | 1202.3 | 848.3 | yes |
| 5120, 17408 | 1024 | 1 | **3398.1** | 3705.3 | 4052.2 | 4692.0 | 3398.1 | yes |
| 17408, 5120 | 1 | 2 | 196.3 | **154.1** | 160.0 | 177.5 | 154.1 | yes |
| 17408, 5120 | 6 | 2 | 196.5 | **167.1** | 188.0 | 235.4 | 167.1 | yes |
| 17408, 5120 | 16 | 2 | 199.1 | **179.3** | 211.8 | 277.1 | 179.3 | yes |
| 17408, 5120 | 64 | 1 | **257.9** | 338.7 | 423.2 | 534.3 | 257.9 | yes |
| 17408, 5120 | 256 | 1 | **953.6** | 1143.1 | 1442.1 | 1985.2 | 953.6 | yes |
| 17408, 5120 | 1024 | 1 | **3526.8** | 4609.3 | 5652.7 | 7645.6 | 3526.8 | yes |
| 5120, 5120 | 1 | 8 | 181.8 | - | - | **83.2** | 83.2 | yes |
| 5120, 5120 | 6 | 8 | 182.2 | - | - | **84.2** | 84.2 | yes |
| 5120, 5120 | 16 | 8 | 183.3 | - | - | **87.8** | 87.8 | yes |
| 5120, 5120 | 64 | 4 | 184.3 | - | - | 167.6 | 167.6 | **unswept at 4** |
| 5120, 5120 | 256 | 1 | **246.4** | - | - | 583.5 | 246.4 | yes |
| 5120, 5120 | 1024 | 1 | **1004.7** | - | - | 2381.5 | 1004.7 | yes |
| 6144, 5120 | 1 | 8 | 182.8 | - | - | **88.1** | 88.1 | yes |
| 6144, 5120 | 6 | 8 | 182.4 | - | - | **115.0** | 115.0 | yes |
| 6144, 5120 | 16 | 8 | 183.2 | - | - | **133.3** | 133.3 | yes |
| 6144, 5120 | 64 | 4 | 184.6 | - | - | 221.0 | 184.6 | **unswept at 4** |
| 6144, 5120 | 256 | 1 | **461.3** | - | - | 713.5 | 461.3 | yes |
| 6144, 5120 | 1024 | 1 | **1226.5** | - | - | 2746.9 | 1226.5 | yes |

Count, stated the way the table can be audited: the rule's choice was measured
in **22 of the 24 cells**, and it is the best measured value in all 22. The two
exceptions are the `M=64` row of the two narrow-N shapes, where the rule picks
`split_k=4` and `4` was never run; for those, the best measured is `8` on
`5120,5120` (167.6) and `1` on `6144,5120` (184.6).

An earlier revision of this file claimed "23 of 24, one miss at 6144x5120 M=64".
That was wrong: `split_k=4` was never measured there, so the cell is unswept,
not a miss. Corrected after review.

### Speedup, measured pairwise

Single-process runs of the same build, 8 repeats each, alternating the knob:

| cell | `split_k=1` median (range) | default median (range) | ratio |
|---|---|---|---:|
| 5120, 17408, M=1 | 666.3 us (635.5-686.3) | 150.7 us (150.4-151.7) | 4.4x |
| 5120, 17408, M=6 | 603.1 us (602.7-603.5) | 163.5 us (158.6-164.5) | 3.7x |
| 5120, 5120, M=1 | 182.4 us (182.3-183.1) | 66.9 us (66.6-70.1) | 2.7x |

So "4.3x" belongs to the `M=1` tall-K cell (4.4x on medians). At `M=6` it is
3.7x, and on the square shape at `M=1` it is 2.7x. The earlier text attached
4.3x to the `M=6` numbers, which was a mismatch.

### Why two tables gave 66.9 and 83.2 for the same cell

`5120, 5120, M=1` reads 83.2 us in the sweep and 66.9 us in the default-path
table, both at `split_k=8`. It is process-to-process variance, not the code
path:

- inside one process the cell is stable: 8 repeats give 66.6-70.1 us;
- separate process runs of the *same* configuration span 67-86 us on this cell
  (26 MB of weights), including 83.8 us with `split_k=8` forced by the knob on
  the same build.

The larger shapes are steadier: `5120, 17408, M=1` repeats at 150.4-151.7 us and
`17408, 5120, M=1` at 155.9-158.1 us.

### Verified default path

Built from the patched tree and run with **no** `SM70_MARLIN_DENSE_SPLIT_K` set.
Marlin microseconds, one run per cell:

| shape (N, K) | M=1 | M=6 | M=16 | M=64 | M=256 | M=1024 |
|---|---:|---:|---:|---:|---:|---:|
| 5120, 17408 | 139.1 | 154.1 | 168.0 | 269.1 | 854.9 | 3403.8 |
| 17408, 5120 | 153.6 | 167.4 | 179.0 | 256.7 | 954.8 | 3515.6 |
| 5120, 5120 | 66.9 | 72.5 | 87.0 | 123.7 | 246.4 | 1002.7 |
| 6144, 5120 | 87.7 | 115.1 | 133.6 | 199.7 | 458.9 | 1215.9 |

Split-K changes the reduction order, so bitwise identity with the unpatched
kernel is not claimed. What is claimed is the measured accuracy: `relerr`
against TurboMind stays at 0.000-0.006, the same band as before the patch.

One cell beats TurboMind: `5120, 17408, M=64`, 269.1 us against 331.8 us (1.23x),
which is the prefill regime rather than decode.

The patch does not change the deployment decision. In the decode regime Marlin
stays behind TurboMind on every 27B shape: 0.81x at `5120, 17408, M=6`
(154.1 against 125.4 us) and 0.58x at `5120, 5120, M=6` (73.7 against 42.9 us,
both measured in the same run). TurboMind's time is nearly flat across M on the
square shape, so the same-batch ratio is what the table shows. `fp8_dense` stays
on TurboMind; the patch matters because the default was 2.7-4.4x off the time
the kernel can reach, and any future Marlin-backed path pays it.

### Guard

`scripts/check_component_patches.py` reads `manifest.json` next to this file,
clones `zhinianqin/marlin_v100` at the pinned commit, and requires the patch to
apply, then reverse-apply. `restomod-patch-guard` runs it on every PR, which
closes the gap the review noted: the top-level `patches/` check never sees a
component patch, so without this it could rot unnoticed.

### Status

`upstream_status: pending`. The account has pull-only access to
`zhinianqin/marlin_v100`, so the change is kept here until a fork or an
upstream PR is agreed.

Verified against a fresh clone of the upstream repository: `git apply --check`
passes at `3f16d44`, the commit `quant/sm70_fp4_moe/PROVENANCE.md` pins. It also
passes in reverse against the tree the extension under test was built from.

Trace: restomod issue #180, lab note `notes/marlin-dense-fp8-sm70.md`.

## 0002-0009: QPN2 prefill reader (issue #224)

Series exported from the local `feat/qpn2-prefill-reader` branch of
`marlin_v100-qpn2` (commits `ef791ba`..`714f002`, base `3f16d44`, the same base
as 0001). The upstream repository is pull-only, so the work lives here.

What it contains:

- the FP16-staging producer measurement harness (`qpn2_producer_bench.cu`) and
  the layout/repack model it shares with the step-1 tests. `0002` also creates
  `csrc/quantization/marlin/qpn2_layout.cuh` and the host-side model files
  (`tests/qpn2_repack_model.h`, `tests/qpn2_layout_host_test.cpp`,
  `tests/qpn2_transpose_probe.cu`): the commit that added them was not
  exported, so the creation is folded into `0002` and the series applies in
  order from `3f16d44`;
- `qpn2_layout.cuh`, `sm70_qpn2_iterator.cuh` (prefill B iterator, stock
  `PitchLinearWarpRakedThreadMap<..., PitchLinearShape<1,32>, 8>`) and
  `sm70_qpn2_mma_core.cuh` (MmaCore alias that keeps the Volta B-congruous
  layout);
- `bool Qpn2 = false` on `Sm70MarlinGemmTraits` and the dense nvfp4 kernel, so
  an unset launch keeps Marlin, plus `sm70_qpn2_select.h`, which fails closed
  when the flag and the layout pointer disagree;
- the M>1 GEMM test and the gate-3 timing source.

State when exported: the producer gate passes (0.79x / 0.88x Marlin at
2 CTAs/SM, value-checked), the iterator fragment matches the layout decode, the
SASS store is two `STS.128`, and the dense TU compiles. The M>1 GEMM test is
**failing** at runtime: the pipeline's B prefetch was fixed with the mask
(`0008`), and a 16-byte shared write past the B stage remains, attributed to the
thread map's warp ordering. Gate 3 has not run. Do not treat the QPN2 path as
usable until that test passes.

Accepted plan and status: issue #224 comments (design accepted after the
producer-cost gates; the patch-series plan is recorded there).
