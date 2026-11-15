## Coverage gap after #230

`#230` merged as `38048fb2`. The admission predicate now has CPU coverage and
none at the GPU level:

- **CPU.** `tests/test_sm70_fp8_dense_quant.py` runs in `restomod-ci` and
  asserts the predicate on synthetic layers: a `RowParallelLinear` without
  `output_size_per_partition` is admitted, a `ReplicatedLinear` is refused as
  `replicated`, an expert layer and the LM head are refused.
- **GPU.** The signed receipt `evidence-7a0f619` covers the standard 14-test
  suite. None of those tests asserts that the admitted set is what a real
  launch converts, so a regression that only shows up on a loaded model would
  still pass the GPU gate.

A cheap test that closes most of the gap, without a server: construct the real
`LinearBase` subclasses (or minimal stand-ins) for every class and shape the
details log distinguishes, run admission over them, and assert the exact
admitted name/shape set plus the refusal reason for each shape — `864x1152`,
`1076x1152`, `1152x288`, `1152x1076`, `2560x160`, `320x2560`, `24x2560`,
`1x2560`, `320x10240`, `10240x320`, `4x10240`. That is CPU-only and sits next
to the current tests.

What it cannot cover is whether the running model's modules really are those
classes; that part needs either a one-shot launch assertion against
`SGLANG_SM70_FP8_DENSE_QUANT_DETAILS` or a test that imports the model's
constructors. Recorded here so the next change to this predicate inherits both
halves.
