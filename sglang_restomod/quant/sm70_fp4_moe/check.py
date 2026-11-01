"""End-to-end check for the SM70 NVFP4 expert GEMM (GPU).

Builds a small NVFP4 expert bank in the ModelOpt packed layout, runs the SM70
conversion and the Marlin GEMM, and compares against a dequantized FP16
reference:

    python -m sglang_restomod.quant.sm70_fp4_moe.check

The packing assumption is the GPTQ one: byte ``j`` along K holds ``k=2j`` in
the low nibble and ``k=2j+1`` in the high nibble. A real checkpoint is checked
by serving a model; this test pins the kernel, the repack, and the scale
encoding against each other.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

# E2M1 magnitude grid: one sign, two exponent, one mantissa bit.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _quantize(w: torch.Tensor):
    """[N, K] fp16 -> packed uint8 [N, K/2], e4m3 scales [N, K/16], codes."""
    n, k = w.shape
    groups = w.reshape(n, k // 16, 16)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-8) / 6.0
    scales = amax.to(torch.float8_e4m3fn)
    q = groups.float() / scales.to(torch.float32)[..., None]
    grid = _E2M1.to(q.device)
    idx = (q.abs()[..., None] - grid).abs().argmin(dim=-1)
    code = torch.where(q < 0, -idx, idx)

    nibble = code.abs().to(torch.uint8)
    sign = (code < 0).to(torch.uint8)
    value = nibble | (sign << 3)
    value = value.reshape(n, k // 2, 2)
    packed = (value[..., 0] | (value[..., 1] << 4)).contiguous()
    return packed, scales, code


def _dequant(packed, scales):
    """Packed [N, K/2] + scales [N, K/16] -> dense fp16 [N, K]."""
    n, k = packed.shape[0], packed.shape[1] * 2
    nib = packed.to(torch.int32)
    codes = torch.stack([nib & 0xF, nib >> 4], dim=-1).reshape(n, k)
    grid = _E2M1.to(packed.device)
    sign = (codes & 8) != 0
    mag = grid[codes & 7]
    values = torch.where(sign, -mag, mag)
    scale = scales.to(torch.float16).repeat_interleave(16, dim=1)
    return (values.to(torch.float16) * scale).contiguous()


def run() -> float:
    """Return the max relative error of the kernel against the reference."""
    from sgl_kernel.scalar_type import scalar_types

    from sglang_restomod.quant.sm70_fp4_moe.marlin import ensure_loaded
    from sglang_restomod.quant.sm70_fp4_moe.method import prepare_moe_nvfp4

    torch.manual_seed(0)
    device = "cuda"
    experts, n, k = 4, 128, 256
    hidden = k
    inter = 128
    tokens, block = 8, 8

    packed, scales, dense = [], [], []
    for _ in range(experts):
        w = torch.randn(n, k, device=device, dtype=torch.float16)
        p, s, _ = _quantize(w)
        packed.append(p)
        scales.append(s)
        dense.append(_dequant(p, s))

    layer = SimpleNamespace(
        params_dtype=torch.float16,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_config=SimpleNamespace(group_size=16),
        w13_bias=None,
        w2_bias=None,
        # Checkpoint layout: [E, N, K/2] uint8 and [E, N, K/16] e4m3.
        w13_weight=torch.stack(packed),
        w13_weight_scale=torch.stack(scales),
        w13_weight_scale_2=torch.ones(experts, device=device),
        w2_weight=torch.zeros(experts, hidden, inter // 2, dtype=torch.uint8, device=device),
        w2_weight_scale=torch.zeros(
            experts, hidden, inter // 16, dtype=torch.float8_e4m3fn, device=device
        ),
        w2_weight_scale_2=torch.ones(experts, device=device),
    )
    prepare_moe_nvfp4(layer)

    x = torch.randn(tokens, k, device=device, dtype=torch.float16)
    sorted_ids = torch.arange(tokens, dtype=torch.int32, device=device)
    expert_ids = torch.zeros(1, dtype=torch.int32, device=device)
    padded = torch.tensor([tokens], dtype=torch.int32, device=device)
    topk_weights = torch.ones(tokens, 1, device=device)

    out = torch.empty(tokens, n, dtype=torch.float16, device=device)
    ensure_loaded().gemm(
        x,
        out,
        layer.w13_weight,
        None,
        layer.w13_weight_scale,
        None,
        layer.w13_weight_scale_2,
        None,
        None,
        None,
        torch.empty(0, dtype=torch.int32, device=device),
        sorted_ids,
        expert_ids,
        padded,
        topk_weights,
        block,
        1,
        False,
        scalar_types.float4_e2m1f.id,
        tokens,
        n,
        k,
        True,
        False,
        False,
        False,
        -1,
        -1,
        -1,
    )
    ref = x @ dense[0].T
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    return float((diff / denom).max())


def main() -> None:
    error = run()
    print(f"max relative error: {error:.3e}")
    raise SystemExit(0 if error < 5e-2 else 1)


if __name__ == "__main__":
    main()
