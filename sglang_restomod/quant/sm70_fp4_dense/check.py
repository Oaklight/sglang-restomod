"""End-to-end check for the SM70 dense NVFP4 linear GEMM (GPU).

Builds a synthetic ModelOpt NVFP4 linear in the checkpoint layout, runs the
engine's own preparation and ``apply_fp4_marlin_linear`` through the patched
``marlin_utils_fp4``, and compares against a dequantized FP16 reference:

    python -m sglang_restomod.quant.sm70_fp4_dense.check

Three shapes are exercised: one already aligned to the Marlin tiles, one whose
``N`` the preparation pads (100 -> 128), and one whose ``K`` it pads
(320 -> 384). A wrong tile layout or a padding mistake changes the answer on
one of them.

The packing assumption is the GPTQ one: byte ``j`` along K holds ``k=2j`` in
the low nibble and ``k=2j+1`` in the high nibble. ``nvfp4_marlin_process_scales``
and the global-scale factor are the engine's own, so this check pins the
bridge and the kernel against each other; a real checkpoint is checked by
serving a model.

The global scale is deliberately non-unit. The reference multiplies by it:
``weight_scale`` is the per-block amax/6, and ModelOpt's ``weight_scale_2``
(``weight_global_scale`` here) is the per-tensor factor the block scales were
divided by when the checkpoint was quantized, so dequantization multiplies it
back. The kernel agrees to the last bit on these shapes, so this pins the
convention as well as the layout.

The extension directory must hold the build that carries the dense CUDA
kernel. The prebuilt ``_sm70_marlin_v100_dense.abi3.so`` (symlinked as
``_C.so``, see ``PROVENANCE.md``) has it; a smaller schema-only ``_C.abi3.so``
loads and exports the name but has no CUDA implementation, and the call fails
with "Could not run '_C::marlin_gemm' with arguments from the 'CUDA' backend".
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

# E2M1 magnitude grid: one sign, two exponent, one mantissa bit.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# Shapes: aligned, N-padded, K-padded.
_SHAPES = ((256, 256), (100, 256), (192, 320))

_TOKENS = 8
_GLOBAL_SCALE = 3.0


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


def _dequant(packed, scales, global_scale: float):
    """Packed [N, K/2] + scales [N, K/16] -> dense fp16 [N, K]."""
    n, k = packed.shape[0], packed.shape[1] * 2
    nib = packed.to(torch.int32)
    codes = torch.stack([nib & 0xF, nib >> 4], dim=-1).reshape(n, k)
    grid = _E2M1.to(packed.device)
    sign = (codes & 8) != 0
    mag = grid[codes & 7]
    values = torch.where(sign, -mag, mag)
    scale = scales.to(torch.float16).repeat_interleave(16, dim=1)
    return (values.to(torch.float16) * scale * global_scale).contiguous()


def _relative_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    return float((diff / denom).max())


def _install():
    """Route ``marlin_utils_fp4`` to the extension; idempotent."""
    import sglang_restomod.quant.sm70_fp4_dense as comp

    comp.install()
    from sglang.srt.layers.quantization import marlin_utils_fp4

    assert getattr(
        marlin_utils_fp4.gptq_marlin_gemm, "_restomod_sm70", False
    ), "the component did not patch marlin_utils_fp4"
    return marlin_utils_fp4


def _case(size_n: int, size_k: int, seed: int = 0):
    """Build one checkpoint-layout layer and its dequantized reference."""
    marlin_utils_fp4 = _install()

    torch.manual_seed(seed)
    device = "cuda"
    w = torch.randn(size_n, size_k, device=device, dtype=torch.float16)
    packed, scales, _ = _quantize(w)
    dense = _dequant(packed, scales, _GLOBAL_SCALE)

    layer = SimpleNamespace(
        # Checkpoint layout: uint8 [N, K/2], e4m3 [N, K/16], scalar.
        weight=torch.nn.Parameter(packed, requires_grad=False),
        weight_scale=torch.nn.Parameter(scales, requires_grad=False),
        weight_global_scale=torch.tensor(_GLOBAL_SCALE, device=device),
        output_size_per_partition=size_n,
        input_size_per_partition=size_k,
        params_dtype=torch.float16,
        quant_config=SimpleNamespace(group_size=16),
    )
    marlin_utils_fp4.prepare_nvfp4_layer_for_marlin(layer)

    x = torch.randn(_TOKENS, size_k, device=device, dtype=torch.float16)
    out = marlin_utils_fp4.apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=size_n,
        size_k=size_k,
        bias=None,
    )
    return _relative_error(out, x @ dense.T)


def run() -> float:
    """Return the worst relative error over the three shapes."""
    return max(
        _case(size_n, size_k, seed=i)
        for i, (size_n, size_k) in enumerate(_SHAPES)
    )


def declines() -> dict[str, str]:
    """Exercise the refusals the dense path makes; return case -> message.

    The preparation pads a misaligned N or K instead of declining it, so shape
    is not a refusal. What is refused: a pack width other than 4 (the SM70 tile
    layout is 4-bit only) and a group size the Marlin layout does not have.
    """
    import sglang_restomod.quant.sm70_fp4_dense as comp

    comp.install()
    from sglang.srt.layers.quantization import marlin_utils_fp4

    cases: dict[str, str] = {}
    device = "cuda"
    qweight = torch.zeros((16, 8), dtype=torch.int32, device=device)
    perm = torch.empty(0, dtype=torch.int32, device=device)
    try:
        marlin_utils_fp4.gptq_marlin_repack(qweight, perm, 256, 256, num_bits=8)
    except ValueError as exc:
        cases["pack-width"] = str(exc)
    else:
        raise AssertionError("a width the SM70 tile cannot read was packed")

    layer = SimpleNamespace(quant_config=SimpleNamespace(group_size=32))
    try:
        marlin_utils_fp4.prepare_nvfp4_layer_for_marlin(layer)
    except ValueError as exc:
        cases["group-size"] = str(exc)
    else:
        raise AssertionError("group_size=32 was prepared for the Marlin layout")

    return cases


def main() -> None:
    error = run()
    print(f"max relative error: {error:.3e}")
    raise SystemExit(0 if error < 5e-2 else 1)


if __name__ == "__main__":
    main()
