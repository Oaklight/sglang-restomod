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
an exact token count, fresh prompts, greedy, node-b, 2026-11-02.

| context | Triton GDN (tok/s) | TileLang GDN (tok/s) | TTFT |
|---|---|---|---|
| 8,041 | 719.9 | 1,933.6 / 1,924.9 | 11.4 s -> 4.16 s |
| 31,158 | 684.6 | 1,658.3 / 1,655.1 | 46.4 s -> 18.79 s |

Greedy continuations are byte-identical to the Triton baseline at 8,192 and
31,744 prompt tokens, including the second send of the same prompt (radix cache
hit) and the 32k path that requests mamba state checkpoints.

The TileLang rows are the signed run behind `evidence-55cb6e7-gdn-tilelang`
(`scripts/gpu_suite.sh` on node-b at `55cb6e7`, `versions.txt` = sglang
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

## Kernel pair (`gdn_pair`, `gdn_pair_tilelang`)

| Field | Value |
|---|---|
| Repository | `Oaklight/sm70-kernels` |
| Commit | `7f7467a46eb4ddf79ea9be9dd0d0e9a61d2cb402` |
| Files | `sm70_kernels/gdn_chunk_o/_chunk_o.py`, `sm70_kernels/gdn_chunk_h/_chunk_h.py` |
| License | Apache-2.0. The repository carries `LICENSE` and a `NOTICE` naming its upstream attributions at this commit; these two kernels are original to it, which the NOTICE states. |
| Vendored as | `_gdn_chunk_o.py`, `_gdn_chunk_h.py` -- byte copies, synced by `scripts/sync_kernels.py` from the `[[kernels.files]]` pin entries |

### What is selected

The two kernels only: the chunk output `chunk_o` and the sequential state pass
`chunk_h`. The package's `__init__.py` getters, benchmarks and tests are not
vendored. `scripts/sync_kernels.py --check` fails on any drift between the pin,
the submodule and these copies.

### Integration

No engine patch. `gdn_pair.install()` rebinds two functions in their source
modules -- `fla.chunk_o.chunk_fwd_o` and
`fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h` -- and in every module that
already imported them, then asserts the replacements are reachable from
`fla.chunk`. That last check is the point: `fla.chunk` binds both names at
import, so patching the source modules alone leaves the engine calling Triton.

Each call is gated. Anything the kernel cannot compute exactly runs the
original Triton function with its arguments untouched:

| refused | why |
|---|---|
| `cu_seqlens` / `chunk_indices` | the kernels take one contiguous sequence, not chunk indices |
| `T % 64 != 0`, `chunk_size != 64` | the state pass steps one chunk at a time; the reference masks the tail |
| `g is None`, `gk`, `use_exp2` | no gate tensor, or a per-channel gate the kernel does not take |
| `track_state` / `track_chunk_idx` | no checkpoint snapshots |
| `save_new_value=False` | the kernel always writes `v_new` |
| `scale != K**-0.5` | the kernel hardcodes `K**-0.5` before the fp16 store |
| non-fp16 activations, non-fp32 gate, bad head dims, non-SM70, not CUDA | outside the kernel's types and tiles |

A launch exception also falls back, once per exception type, so a serving
process cannot be taken down by the fast path.

`chunk_fwd_o` returns a fresh tensor, as the reference does. `chunk_h` writes
the final state back into `initial_state` at the caller's slot indices when
`inplace_update` is set, as the reference does; a negative slot is the pin's
"no state" sentinel and starts at zero without being written back.

### Measured

Kernel time, V100-SXM2, B=1, T=4096, H=12, Hg=4, K=V=128 (per TP rank), from
`benchmarks/gdn_chunk_o.py` and `benchmarks/gdn_chunk_h.py` at `d75987a`:
chunk_o 24.94 ms Triton against 0.27 ms TileLang (108x), chunk_h 15.4 ms
against 0.47 ms (33x). Numerics: 0.41-1.28 fp16 ULP against the Triton
reference.

End-to-end, the pair accounted for 1390 ms of a 3045 ms prefill on the
FlashNext 8xV100 box. That delta is kernel-time and config-independent; the
tok/s multiple measured on that contended box with unequal memory fractions is
a reference, not a target. A clean alternating A/B on lambda4 (Triton 1633
prompt tok/s, isolation confirmed) is the number this component is sized
against, and it has not been taken yet.
