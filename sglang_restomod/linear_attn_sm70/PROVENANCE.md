# Provenance: linear_attn_sm70

## Source

| Field | Value |
|---|---|
| Repository | `dg1kjd/sglang-sxm2` |
| Commit | `73abfe1dd7e7aae7a62b222a201e48f532fb298e` |
| File | `python/sglang/srt/layers/attention/linear/kernels/gdn_chunked_tilelang.py` |
| License | Apache-2.0 |
| Related | `.../kernels/gdn_tilelang.py` (dispatcher, not vendored) |

## What is selected

`gdn_chunked_tilelang.py` only: the SM70 TileLang chunked gated-delta-rule
forward, its KKT inverse, the chunk-forward kernel and the recurrent fallback.
The fork's `gdn_tilelang.py` is not vendored; the component binds
`chunked_gdn_sm70` directly from `gdn_tilelang.py` here, because the pin's GDN
backend calls `extend(q, k, v, g, beta, ...)`, which is the chunked entry
point. The fork's `packed_extend` and persistent decode kernel are not needed:
decode and verify keep their pinned Triton kernels.

The file imports only `sglang.kernels.ops.attention.fla.{chunk_fwd,cumsum,index,l2norm}`,
all present at the pin, and `tilelang` (0.1.8 in the pinned environment).

## Integration

No engine patch. The component wraps `TritonGDNKernel.extend` and answers the
call when the tensors match the kernel's contract (`[1, tokens, heads, dim]`,
FP16, `K == V == 128`, at least 256 tokens, SM70). Everything else falls through
to Triton.

The pin's `GDNAttnBackend.forward_extend` reads the mamba track mask from
`forward_metadata` and then calls `extend` without it, so the kernel cannot tell
whether its third return value (the per-chunk states `h`) will be read. The
component wraps `forward_extend` to record that decision per call and turns on
the TileLang kernel's `store_checkpoints` when it is set. That wrapper is
installed on the first prefill, not at register time: the package entry point
runs components before it installs the SM70 overrides, and importing the model
runner that early binds the upstream `sm75` capability gate for the rest of the
process.

## Measured

Qwen3.8-27B-FP8, TP2, 2x V100-SXM2-32GB, chunked prefill 4096, prompt built at
an exact token count, fresh prompts, greedy, lambda12, 2026-11-02.

| context | Triton GDN (tok/s) | TileLang GDN (tok/s) | TTFT |
|---|---|---|---|
| 8,041 | 719.9 | 1,933.6 / 1,924.9 | 11.4 s -> 4.16 s |
| 31,158 | 684.6 | 1,658.3 / 1,655.1 | 46.4 s -> 18.79 s |

Greedy continuations are byte-identical to the Triton baseline at 8,192 and
31,744 prompt tokens, including the second send of the same prompt (radix cache
hit) and the 32k path that requests mamba state checkpoints.

The TileLang rows are the signed run behind `evidence-55cb6e7-gdn-tilelang`
(`scripts/gpu_suite.sh` on lambda12 at `55cb6e7`, `versions.txt` = sglang
0.5.22, the pin); the Triton rows are the earlier `bench.py` runs on the same
box with the same launcher.

Decode does not go through this component: `packed_decode` and the fork's
persistent decode kernel are not vendored. A same-commit A/B at 31,744 prompt
tokens, 128 output tokens, 4 reps per arm, one server restart apart:

| arm | prefill tok/s | decode tok/s |
|---|---|---|
| `gdn_tilelang_sm70` | 1,648.4 / 1,645.1 / 1,639.0 / 1,637.4 | 45.74 / 48.14 / 42.69 / 41.75 |
| Triton only | 684.7 / 684.2 / 684.4 / 684.3 | 47.88 / 38.09 / 49.21 / 46.84 |

Prefill is 2.40x with a spread under 1% on both arms. The decode ranges overlap
and the Triton arm alone spans 38.09 to 49.21, so no difference is attributable
to the component in either direction; two-rep decode numbers on this box are
not a stable estimate.

## Dependency note

`tilelang` is not declared in `pyproject.toml`. It arrives with the pinned
engine environment (0.1.8 was the version measured here), and the kernel import
is guarded: if it is absent, the component logs once and leaves every call on
Triton.
