"""Numeric self-test for the SM70 FP8 MoE binding.

Run on a V100 with the SGLang TurboMind library available:

    python -m sglang_restomod.quant.sm70_fp8_moe.selftest

It checks a gated, multi-expert case against an FP16-dequantized reference and
returns the relative error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import apply, ensure_loaded, prepare_experts


def run() -> float:
    ensure_loaded()

    experts, n, k, per_expert = 4, 256, 128, 2
    tokens = experts * per_expert
    torch.manual_seed(3)

    w = (torch.randn(experts, n, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    scales = torch.ones(experts, n // 128, k // 128, device="cuda", dtype=torch.float32)
    prepared = prepare_experts(w, scales)

    x = torch.randn(tokens, k, device="cuda", dtype=torch.float16)
    offsets = torch.tensor(
        [i * per_expert for i in range(experts + 1)], device="cuda", dtype=torch.int32
    )
    out = torch.empty(tokens, n // 2, device="cuda", dtype=torch.float16)
    apply(out, x, offsets, prepared)
    torch.cuda.synchronize()

    ref = torch.empty_like(out, dtype=torch.float32)
    for e in range(experts):
        lo, hi = e * per_expert, (e + 1) * per_expert
        y = x[lo:hi].float() @ w[e].float().T
        ref[lo:hi] = F.silu(y[:, : n // 2]) * y[:, n // 2 :]

    err = (out.float() - ref).abs()
    return (err.max() / ref.abs().max()).item()


def main() -> None:
    rel = run()
    print(f"rel={rel:.5f}")
    assert rel < 5e-3, "FP8 MoE GEMM does not match the reference"


if __name__ == "__main__":
    main()
