# 2026-11-15 — The dense-GEMM dive: where the time is, and where it is not

Status: **measured, and the GEMV lane is closed.** The decode dense cluster is
byte-bound once it is on fp8 and launch-bound whatever its dtype; neither a
Triton nor a TileLang per-call GEMV clears the control on the shape family that
bounds the time; the exact-dense prefill dispatch was worth 2.1% once every
listed shape reached it, and nothing more than that was claimed for it.

Everything here is one box (lambda4, 4x V100-32GB, TP4), one arm per
configuration, no repeats unless stated. The prototype scripts for the GEMV
numbers were scratch files and are gone; the numbers are what those runs
printed, not a re-runnable harness. The two server arms in the last section are
`bench.py` on the site tree and are reproducible.

## 1. The decode dense cluster is byte-bound, and then launch-bound

Before any of this, the dense (non-expert) linears were the largest single
item in a decode step: **11.27 ms of a 28.56 ms step (39.5%)** on the bf16 arm.

The byte floor says what that could be worth. Those weights are 1.91 ms of HBM
traffic at ~900 GB/s on this part, so the cluster was running at **~17% of its
floor** — the earlier figure of 8.5% that circulated for a day was computed
against the fp8 byte count while the arm was bf16. The 2x there is the dtype,
not the kernel.

Moving the cluster to block FP8 brought it to **9.53 ms** (kernel total 26.54
ms), a 15% cut. Effective bandwidth *fell* from 152 to ~120 GB/s across the
same move. That direction is the discriminator: a kernel win raises achieved
bandwidth on the same bytes; a bytes win lowers it. The gain came from reading
half as many bytes, not from a better kernel.

What is left, **6.42 ms**, is the 97 HC linears per step, and they are
launch-bound: each is already ~24 us, close to what `torch.mv` costs on the
same tensors, so the arithmetic is not the problem. Counting launches, not
bytes, is the way to attack them.

## 2. A per-call GEMV kernel is the wrong fix, in Triton

The obvious move was to write the HC linears as one fused GEMV. Measured on
the same tensors:

| implementation | time |
|---|---:|
| `torch.mv` (the served path, already ~24 us in-engine) | 28-32 us |
| Triton split-K GEMV, shape 1 | 92.7 us |
| Triton split-K GEMV, shape 2 | 172.7 us |
| Triton split-K GEMV, `[10240, 320]` | 685.9 us |

Every Triton variant is slower, by 3x to 24x, and the wide-N/few-K shape
collapses outright. The reason is the one this repository keeps rediscovering:
**Triton 3.5.1 emits no tensor-core operations on sm70**, so a hand-written
Triton kernel is scalar FMA against a cuBLAS path that is already at the
launch-cost floor.

Verdict for Triton: no per-call kernel change is justified on this
evidence. The Triton numbers say nothing about TileLang, which does emit
`mma.sync` on sm70 — so that question was tested separately, below.

### The TileLang rounds

Standalone microbenchmark, lambda4 GPU4, TileLang 0.1.8 / torch 2.9.1, M=4.
Control is `min(torch.mv, torch.matmul)` on the same tensors; the control itself
runs at 314-348 GB/s on the two 6.55 MB shapes, not the ~218 GB/s quoted
earlier, so the bar is higher than it looked.

**Round one: a hand-mapped reduction, 25 GB/s.** The kernel mapped
`(block_N, M)` per block with a serial K loop and read one fp16 per thread per
step. It ran at 25 GB/s — 10x slower than the control. **This number says the
mapping was wrong, not that TileLang is slow:** it is listed here so it does not
later read as a language verdict.

**Round two: tensor-core tiles with M padded 4 to 16.** Padding costs 4x compute
on a bandwidth-bound operation, which is free; the MMA path is the one sm70
actually has.

| shape (N,K) | control us | round two us | ratio |
|---|---:|---:|---:|
| `(320, 10240)` | 19.0 | 148.9 | 0.13x |
| `(10240, 320)` | 20.9 | 14.5 | 1.45x |
| `(4, 10240)` | 22.4 | 101.8 | 0.22x |
| `(320, 2560)` | 23.6 | 27.1 | 0.87x |
| `(2560, 160)` | 19.0 | 4.8 | 4.00x |

**The split is structural.** The grid was over N only, so `(320, 10240)` became
5-10 blocks on an 80-SM part with 160 sequential K chunks, and `(4, 10240)`
became one block. Every loss is a small-N shape; the two large-N shapes won.

**Round three: split-K**, grid over (N, ksplit), fp32 partials, a second kernel
summing them, both inside the timed region.

| shape (N,K) | control us | split-K us | best ksplit | ratio |
|---|---:|---:|---:|---:|
| `(320, 10240)` | 18.8 | 32.4 | 8 | **0.58x** |
| `(10240, 320)` | 17.2 | 14.2 | 1 | **1.21x** |
| `(4, 10240)` | 17.8 | 24.3 | 8 | 0.73x |
| `(320, 2560)` | 15.2 | 9.4 | 8 | **1.61x** |
| `(2560, 160)` | 11.8 | 5.3 | 1 | **2.22x** |

Split-K moved `(320, 10240)` from 148.9 to 32.4 us and `(320, 2560)` from 0.87x
to 1.61x, so the diagnosis was right. It did not close `(320, 10240)`: five
N-blocks times eight splits is 40 blocks against 80 SMs, and the reduction
kernel's cost is charged on top — at 6.55 MB that second kernel is a real
fraction of the total, though the two are timed together here and their shares
are not separated. `(10240, 320)` won at ksplit=1; it never needed the split.

**The lane is closed.** The gate was: beat the control on both 6.55 MB shapes.
`(320, 10240)` loses at 0.58x with the parallelism fixed, so no further rounds
were run. Four of five shapes are at or above the control, and that is still not
a pass, because the gate was the two big shapes.

What would reopen it: a design that keeps N-blocks and K-splits in one kernel
(atomics, or a fused reduction), which is a different kernel, not another round
of this one.

### Why not dispatch per shape, and use TileLang only where it wins

The obvious objection: four of five shapes are at or above the control, so route
just those to a TileLang kernel. The arithmetic does not support it.

The two big shapes are the whole family's bytes — 97 launches each, 6.55 MB
each, about 1.27 GB per step — and one of them, `(320, 10240)`, is a loss. What
the wins are worth, from this run's timings:

| shape | saves per launch | launches per step | saved |
|---|---:|---:|---:|
| `(10240, 320)` 1.21x | 3.0 us | 97 | 0.29 ms |
| `(320, 2560)` 1.61x | 5.8 us | 48 | 0.28 ms |
| `(2560, 160)` 2.22x | 6.5 us | 48 | 0.31 ms |
| **total** | | | **~0.88 ms of a 25.6 ms step, ~3%** |

Those are standalone savings, and in situ the served path already averages ~24 us
per launch, so the real number is at best this and probably less. The price is a
new kernel, a new dispatch gate and a per-shape table on a family that has
already cost +2.7 ms/step once through a Python-level hook: the HC modules call
`F.linear` on the raw weight and bypass `forward`, so any hook has to intercept
where the current path dispatches.

A 3% ceiling for a permanent dispatch table is not the trade this repository
should make, and the closure is argued rather than asserted.

### The TileLang question is answered, not abandoned

TileLang is competitive or faster at large N without split-K (`(10240, 320)`
1.21x, `(2560, 160)` 2.22x) and with split-K at `(320, 2560)` (1.61x). It cannot
beat cuBLAS's gemv on `(320, 10240)` even after the parallelism defect is fixed.
That is a result about where the design loses, and it should stop the next
person spending a window on the same kernel.

Two properties of the measurement worth carrying:

- `(4, 10240)` is 0.08 MB against a 17.8 us control. That is launch latency, not
  bandwidth, so no kernel of this family wins it while the launch dominates —
  a property of the measurement, not of the language.
- Building these kernels has two traps with no useful error message. TVMScript
  will not see constants from a closure (`NameError: KS/KP`), and it parses with
  `inspect.getsource`, so a source string passed to `exec` fails with
  `OSError: could not get source code`. Write each generated kernel to a module
  file and import it.

## 3. A landmine: un-compiled small GEMMs run at ~4 GB/s

While measuring, two shapes ran outside any compiled region and came in at
**~4 GB/s** — roughly 50x slower than `torch.mv` on the same data:

- `[320, 10240]` at M=4
- `[4, 10240]` at M=4

Inside a CUDA graph these cost what they cost; eagerly, at model load or in any
non-captured path, they are catastrophic and look like a different defect. Two
consequences:

1. Any timing of decode or load that reports these shapes outside a captured
   region should be read as measuring this, not the model.
2. Any future change that moves a linear out of the compiled region inherits
   this cost silently. It is worth a test of its own, not a note.

## 4. The exact-dense prefill dispatch: no difference where it reached

The `fp8_prefill_dispatch` kernel is a scheduling variant of the TurboMind
`fp8_gemm` for TP4 dense projections at M >= 3920. Before timing it, it was
checked numerically: same prepared weight, same scales, same input, M=4096,
all six Flash-Next shapes, **max relative difference 0.000e+00**. So this pair
is a speed question only.

4121-token prompts, three cold runs each, nonce per run, every run
`cached=0`, one launch per arm:

| arm | admitted / rank | prefill | decode step |
|---|---:|---|---|
| base, no shape table | 0 | 1053 ms / 3912 tok/s (spread 1.5%) | 24.74 ms, accept 3.12 |
| table + env var | 36 | 1059 ms / 3893 tok/s (spread 1.5%) | 25.39 ms, accept 3.05 |

**Null.** Decode cannot be affected by construction — the dispatch requires
M >= 3920 and decode runs at M <= 8 — so the 2.6% decode difference is
arm-to-arm noise.

The null is narrow, and the log says how narrow. Of 416 candidates on
Flash-Next:

- **144 carry a `prefix`** and reached a name match; 36 per rank matched, all
  of them `linear_attn.in_proj_qkvz (2560, 4096)`.
- **272 carry no `prefix`** at all, so their suffix is empty and a
  name-matched gate cannot admit them. They hold the shapes the table lists:
  196 x `(1536, 2560)` — the o_proj/out_proj entry, **49 per rank, the largest
  group in the model** — plus 52 x `(2560, 3584)`, 8 x `(2560, 2560)`, 8 x the
  LM head, and two singletons.

So the statement for that pair is narrow: **36 of the ~68 layers per rank
that the table can cover took the dispatch and showed no difference.** The
49-per-rank `(1536, 2560)` group never took it.

### The follow-up, with the shape fallback in place

The later measurement was run the same evening, on the head that admits by
shape when the name is absent (#299), same protocol, one launch:

| arm | admitted / rank | prefill |
|---|---:|---|
| base, site tree, no table | 0 | 1053 ms / 3912 tok/s, spread 1.5% |
| name gate only | 36 | 1059 ms / 3893 tok/s, spread 1.5% |
| head, shape fallback | 100 | **1031 ms / 3996 tok/s, spread 0.8%** |

**2.1% faster than base**, and the separation is per run rather than on the
median: the three head runs (1039, 1031, 1031) are each faster than the
fastest base run (1051). Admitted per rank: 49 x `(1536, 2560)`, 36 x
`(2560, 4096)`, 13 x `(2560, 3584)`, 2 x `(2560, 2560)`; three stay on
`fp8_gemm`, all outside both tables.

Two limits on that number. It is one launch per arm, so it is a small positive
on per-run separation and not a repeated-measures result. And the finiteness is
worth naming: the dispatch is now reached for every dense projection the table
lists, so a 2% prefill gain at 4121 tokens is what this kernel is worth here —
it is not the missing 8%, and the prefill question remains open beyond it.

The candidate census also killed one entry in the table: `qkv_proj` is
`(2560, 3584)` per rank, not the `(2560, 3328)` derived from the config's head
counts.

## 5. What this does not say

- It does not say the dispatch is slower. Bit-identical arithmetic with a
  different schedule, measured at two points 0.6% apart inside a 1.5% spread.
- It does not say the prefill regression has a kernel cause. That question is
  still open, and the largest dense group was never in the comparison.
- It does not say TileLang cannot help the HC linears. Only Triton was tried.

## The gate itself was dead, and that is a separate note

Both exact-dense tables admitted zero layers on every site until today, because
their guard read an attribute no linear module carries. The cause, the
evidence and the fix are in [#298](https://github.com/Oaklight/sglang-restomod/pull/298)
with the code in [#299](https://github.com/Oaklight/sglang-restomod/pull/299);
they are not repeated here. This note records what the dense kernels cost and
what was tried, not why the switch was off.
