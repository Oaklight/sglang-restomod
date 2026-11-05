"""GPU test for the decode-shaped MoE GEMV and its dispatch.

Two things have to hold before the kernel can replace the Marlin call:

1. it agrees with Marlin on a shape it accepts, and
2. it declines the shapes it cannot read, instead of silently using the wrong
   repack ordering.

The check builds a real quantized expert bank, so the weights, the encoded
block scales and the routing all come from the production path.

Skips when `SM70_MARLIN_DIR` is unset: the Marlin extension is built outside
this repository (see `quant/sm70_fp4_moe/PROVENANCE.md`).
"""

import os

import pytest

pytestmark = pytest.mark.gpu

# gate_up at TP4: 320 columns, which is N/64 == 5 (odd) and therefore decoded.
GATE_UP_N = 320
# The same call at TP8: 160 columns, N/64 == 2.5, not a repack tile.
TP8_N = 160


def _skip_unless_usable():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    if not os.environ.get("SM70_MARLIN_DIR"):
        pytest.skip("SM70_MARLIN_DIR is unset; the marlin_v100 extension is external")
    return torch


def _build_case(torch, size_n: int, size_k: int, experts: int, tokens: int, top_k: int):
    """Quantize an expert bank through the production path and route tokens."""
    from types import SimpleNamespace

    from sglang_restomod.quant.sm70_fp4_moe import check as chk
    from sglang_restomod.quant.sm70_fp4_moe.method import prepare_moe_nvfp4

    device = "cuda"
    packed, scales, dense = [], [], []
    for _ in range(experts):
        w = torch.randn(size_n, size_k, device=device, dtype=torch.float16)
        p, s, _ = chk._quantize(w)
        packed.append(p)
        scales.append(s)
        dense.append(chk._dequant(p, s))

    layer = SimpleNamespace(
        params_dtype=torch.float16,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_config=SimpleNamespace(group_size=16),
        w13_bias=None,
        w2_bias=None,
        w13_weight=torch.stack(packed),
        w13_weight_scale=torch.stack(scales),
        w13_weight_scale_2=torch.ones(experts, device=device),
        w2_weight=torch.stack(packed),
        w2_weight_scale=torch.stack(scales),
        w2_weight_scale_2=torch.ones(experts, device=device),
    )
    prepare_moe_nvfp4(layer)

    rows = tokens * top_k
    block = 8
    sorted_ids = torch.full((rows * block,), rows, dtype=torch.int32, device=device)
    for r in range(rows):
        sorted_ids[r * block] = r
    expert_ids = torch.tensor([r % experts for r in range(rows)],
                              dtype=torch.int32, device=device)
    return dict(
        torch=torch,
        layer=layer,
        dense=torch.stack(dense),
        x=torch.randn(tokens, size_k, device=device, dtype=torch.float16),
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        padded=torch.tensor([rows * block], dtype=torch.int32, device=device),
        topk_weights=torch.full((rows,), 1.0, dtype=torch.float32, device=device),
        rows=rows,
        top_k=top_k,
        block=block,
    )


def _run(torch, case, size_n: int, size_k: int, force_marlin: bool):
    """Call the patched entry point; ``force_marlin`` disables the new path."""
    from sglang_restomod.quant.sm70_fp4_moe.marlin import ensure_loaded
    from sglang_restomod.quant.sm70_fp4_moe.method import moe_wna16_marlin_gemm
    from sgl_kernel.scalar_type import scalar_types

    previous = os.environ.get("SM70_MOE_DECODE_GEMV")
    if force_marlin:
        os.environ["SM70_MOE_DECODE_GEMV"] = "0"
    try:
        out = torch.empty(case["rows"], size_n, dtype=torch.float16, device="cuda")
        moe_wna16_marlin_gemm(
            case["x"], out, case["layer"].w13_weight, None,
            case["layer"].w13_weight_scale, case["layer"].w13_weight_scale_2,
            None, None, None, torch.empty(0, dtype=torch.int32, device="cuda"),
            case["sorted_ids"], case["expert_ids"], case["padded"],
            case["topk_weights"], case["block"], case["top_k"], False, False,
            scalar_types.float4_e2m1f, case["rows"] // case["top_k"],
            size_n, size_k,
        )
        return out
    finally:
        if previous is None:
            os.environ.pop("SM70_MOE_DECODE_GEMV", None)
        else:
            os.environ["SM70_MOE_DECODE_GEMV"] = previous


def test_decode_gemv_matches_marlin():
    torch = _skip_unless_usable()
    ensure = pytest.importorskip("sglang_restomod.quant.sm70_fp4_moe.decode_gemv")
    if ensure._load() is None:
        pytest.skip("decode GEMV extension did not build")

    case = _build_case(torch, GATE_UP_N, size_k=2560, experts=4, tokens=4, top_k=2)
    assert ensure.supports(4, GATE_UP_N, 2560, 2, 8)

    gemv = _run(torch, case, GATE_UP_N, 2560, force_marlin=False)
    marlin = _run(torch, case, GATE_UP_N, 2560, force_marlin=True)
    torch.cuda.synchronize()

    scale = marlin.float().abs().max().item()
    diff = (gemv.float() - marlin.float()).abs().max().item()
    assert diff <= 1e-3 * scale, f"max |gemv - marlin| = {diff:.3e} on |out| <= {scale:.3f}"


def test_decode_gemv_declines_an_even_tile_count():
    """N=128 has two 64-column tiles; that repack order is not decoded."""
    torch = _skip_unless_usable()
    from sglang_restomod.quant.sm70_fp4_moe import decode_gemv as gemv

    assert not gemv.supports(4, 128, 2560, 2, 8)
    assert gemv.supports(4, GATE_UP_N, 2560, 2, 8)
    assert not gemv.supports(4, GATE_UP_N, 2560, 2, 16), "route tile is 8 rows"
