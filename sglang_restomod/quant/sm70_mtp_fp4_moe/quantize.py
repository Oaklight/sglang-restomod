"""NVFP4 quantization of unquantized expert banks to the checkpoint layout.

The draft model's experts arrive as FP16 because the checkpoint's
``quantization_config.ignore`` lists ``mtp.*``. This module produces the triple
the loader would have produced for an NVFP4 checkpoint, so that the existing
``prepare_moe_nvfp4`` / Marlin path reads a draft layer exactly as it reads a
target layer.

Convention, taken from the engine's own reader rather than assumed:
``dequantization.dequantize_nvfp4`` computes

    w = fp4_value * block_scale_e4m3 * global_scale

with ``block_scale`` repeated 16 times along the contraction axis. This module
inverts that:

    global_scale = amax_expert / (448 * 6)
    block_scale  = amax_block * 448 / amax_expert      (rounded to e4m3)
    fp4_value    = round_e2m1(w / (block_scale * global_scale))

so ``block_scale * global_scale = amax_block / 6`` and the quotient lands in
[-6, 6]. One global scale per expert keeps the gate and up halves on the same
scale, which Marlin requires; ``ModelOptNvFp4FusedMoEMethod`` collapses a
per-half pair for the same reason.

Packing is two codes per byte, low nibble = even index, matching
``dequantize_nvfp4``'s default ``high_nibble_first=False``.

Experts are quantized one at a time. The full bank in one shot needs three
FP32-scale temporaries plus an int64 code tensor, which is several times the
bank itself; at 128 local experts x 320 x 2560 that does not fit beside a
serving model at ``mem_fraction_static 0.82``.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK = 16
MAX_E4M3 = 448.0
MAX_E2M1 = 6.0
#: Smallest e4m3 subnormal; a block scale of 0 would erase its whole block.
MIN_BLOCK_SCALE = 2.0**-9

#: Midpoints between the E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass(frozen=True)
class QuantizedExperts:
    """One expert bank in the checkpoint layout."""

    weight: object  # uint8 [E, N, K/2], two e2m1 codes per byte
    scales: object  # fp8_e4m3 [E, N, K/16]
    global_scale: object  # fp32 [E]


def refusal(weight) -> str | None:
    """Return why ``weight`` cannot be quantized, or ``None`` when it can."""
    import torch

    if weight.ndim != 3:
        return f"expected [experts, N, K], got {tuple(weight.shape)}"
    if not weight.is_floating_point():
        return f"integer weights ({weight.dtype}) are already packed"
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"unsupported dtype {weight.dtype}"
    k = weight.shape[-1]
    if k % BLOCK:
        return f"K={k} is not a multiple of the {BLOCK}-element NVFP4 block"
    if bool((~torch.isfinite(weight)).any()):
        return "weights contain inf or nan"
    return None


def _codes(x, boundaries):
    """Nearest E2M1 magnitude code (0..7) with the sign bit in bit 3."""
    import torch

    magnitude = (x.abs().clamp(max=MAX_E2M1)).to(torch.float32)
    # bucketize returns the index of the first boundary >= the value, which is
    # the nearest magnitude because the boundaries are the midpoints.
    codes = torch.bucketize(magnitude, boundaries).to(torch.uint8)
    return codes | ((x < 0).to(torch.uint8) << 3)


def quantize_experts(weight):
    """Quantize a ``[E, N, K]`` expert bank to the NVFP4 checkpoint layout.

    Raises ``ValueError`` for a shape or dtype :func:`refusal` rejects; callers
    that convert as they load should call :func:`refusal` first and leave the
    layer alone instead.
    """
    import torch

    reason = refusal(weight)
    if reason is not None:
        raise ValueError(reason)

    experts, size_n, size_k = weight.shape
    blocks = size_k // BLOCK
    device = weight.device
    boundaries = torch.tensor(_MIDPOINTS, dtype=torch.float32, device=device)

    packed = torch.empty(experts, size_n, size_k // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(experts, size_n, blocks, dtype=torch.float8_e4m3fn, device=device)
    global_scales = torch.ones(experts, dtype=torch.float32, device=device)

    for expert in range(experts):
        bank = weight[expert].to(torch.float32)
        amax_expert = bank.abs().amax()
        if amax_expert <= 0:
            packed[expert] = 0
            scales[expert] = 0
            continue
        global_scale = amax_expert / (MAX_E4M3 * MAX_E2M1)
        block_scale = (
            bank.view(size_n, blocks, BLOCK).abs().amax(dim=-1) * (MAX_E4M3 / amax_expert)
        ).clamp_min(MIN_BLOCK_SCALE)
        block_scale = block_scale.to(torch.float8_e4m3fn)
        effective = block_scale.to(torch.float32) * global_scale
        codes = _codes(bank / effective.repeat_interleave(BLOCK, dim=-1), boundaries)
        low = codes[:, 0::2]
        high = codes[:, 1::2]
        packed[expert] = low | (high << 4)
        scales[expert] = block_scale
        global_scales[expert] = global_scale

    return QuantizedExperts(
        weight=packed.contiguous(),
        scales=scales.contiguous(),
        global_scale=global_scales.contiguous(),
    )
