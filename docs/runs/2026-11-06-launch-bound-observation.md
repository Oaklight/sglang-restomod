# 2026-11-06 — Launch-bound steps in a hybrid attention model

Status: **observation from a sibling project, no run here.** Recorded because
the measurement distinction may transfer to serving; it is not a request to
port a patch. Recorded from issue #203.

## The finding

On a hybrid attention model (gated linear attention plus full attention), a
training step that looked expensive for GPU reasons was bound by shape changes
and kernel launches, not by arithmetic:

| | |
|---|---|
| summed kernel time per step | 13–21 ms |
| wall time per step, before | 2,111 ms |
| wall time per step, after | 994 ms |
| end-to-end rate, before → after | 0.72 → 16 items/s, about 22x |

Two structural causes, both about the shape of the work rather than its amount:

1. A forward pass whose output was discarded was still paid for.
2. The combined sequence used its real length, so nearly every item took a new
   shape and paid a kernel load. Fixed-shape buckets, already used on the
   frozen path, were not used on the adapted path.

## What probably transfers to serving

- The measurement first. In single-stream or small-batch serving even more of
  the wall time is launch overhead, and the profiler distinction between kernel
  time and step time says whether optimisation belongs in kernels or in
  scheduling and shapes.
- Fixed-shape bucketing of a per-request suffix, if the serving path builds
  sequences of varying length per request.
- Prefix-cache behaviour: on the training side, keeping the prefix cache under
  `no_grad` instead of recomputing halved a 64-item evaluation. The serving
  analogue is whether the radix or prefix cache is genuinely reused per
  request, or silently rebuilt because a key or a shape does not match.

## What does not transfer

- The discarded-forward fix: that pass exists only on the gradient path.
- The backward-path workaround: a training-only in-place version check in the
  chunked delta-rule backward.
- Seed control.

## What to be careful about

The original measurement was on a Hopper-class GPU with bf16. A V100 target
differs in at least three ways: fp16 rather than bf16, no Hopper-specific
kernels, and different memory bandwidth relative to launch overhead, which
shifts the balance further toward launch-bound. Bucket sizes must be
re-measured, not assumed.

## Proposed first step

One measurement on the target hardware, with the same profiler: kernel time
versus step time for a representative request shape, at batch 1 and at the
intended serving batch. If kernel time is a small fraction of wall time, the
rest of the observation is worth pursuing; if not, close it.
