# SM70 FP8 MoE GEMM

Binds the block-FP8 expert GEMM that ships inside the prebuilt TurboMind
library (`_sm70_turbomind_v100.so`) but is never registered with torch.

Status: **spike validated on 2026-10-25** (lambda4, 4x V100-32GB). Relative
error against an FP16-dequantized reference is 7e-4 for a gated, multi-expert
case. Not yet wired into `Fp8MoEMethod`; that is #20.

## Why a shim

- `nm` shows `fp8_moe_gemm_sm70_out` exported by the prebuilt library.
- `sm70_turbomind_bindings.cpp` registers `fp8_prepare`, `f16_moe_gemm`,
  `build_ptrs` and the permute helpers, but not this symbol.
- The kernel sources are not in the SGLang tree, so the library cannot be
  rebuilt here.

The shim resolves the exported symbol with `dlopen`/`dlsym` and registers
`sglang_sm70_fp8_moe::fp8_moe_gemm` with the same schema as the origin
implementation (`1CatAI/1Cat-vLLM`, `csrc/ops.h` and
`csrc/torch_bindings.cpp`).

## Contract

Per expert, from a block-FP8 checkpoint weight `W[N, K]` (logical, e4m3) and
its scales `S[N/128, K/128]` (fp32):

```python
tm_w, tm_s, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
    W[e].contiguous(), S[e].contiguous(), 128, True /* interleave_gated_silu */
)
# stack over experts
ptr_w, ptr_s = torch.ops.sglang_sm70_turbomind.build_ptrs(
    tm_w_stack, tm_s_stack, int(meta[0]), int(meta[1]), num_experts
)
torch.ops.sglang_sm70_fp8_moe.fp8_moe_gemm(
    out,        # [tokens, N // 2] fp16, sorted order
    x,          # [tokens, K] fp16, sorted order
    offsets,    # [num_experts + 1] int32
    ptr_w, ptr_s, num_experts, K, N, 128, True,
)
```

Notes from the validation:

- Scales are `[N/128, K/128]`, not per channel. `fp8_prepare` rejects anything
  else with "output scale block mismatch".
- `interleave_gated_silu=True` is required for gated MLPs. With it the kernel
  computes `silu(y[:, :N/2]) * y[:, N//2:]` after the projection, matching the
  checkpoint's fused gate/up layout once prepared.
- `build_ptrs` is the AWQ pointer builder; it is dtype-agnostic and is what the
  origin implementation uses for FP8 experts too.

## Files

- `shim.cpp` — the binding. Build with `torch.utils.cpp_extension.load(...,
  is_python_module=False)` after loading the TurboMind library and setting
  `SM70_TURBOMIND_SO` to its path.
- `selftest.py` — gated, multi-expert numeric check against a dequantized
  reference.

## Provenance

- Symbol and signature: `1CatAI/1Cat-vLLM`, `csrc/ops.h` (`fp8_moe_gemm_sm70_out`),
  `csrc/torch_bindings.cpp`, `vllm/_sm70/linear.py` (`fp8_sm70_prepare` with
  `interleave_gated_silu`), `vllm/model_executor/layers/quantization/fp8_sm70_moe.py`
  (prepare/build_ptrs/gemm call sequence).
- License: Apache-2.0 (same as the origin and SGLang).
