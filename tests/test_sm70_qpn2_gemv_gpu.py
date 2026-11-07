"""GPU guard: QPN2 decode GEMV against the shipping path on a small bank (#224 P2).

Both layouts are built through ``prepare_moe_nvfp4`` from the same
checkpoint-format tensors, and each arm is served through the shipping entry
point ``moe_wna16_marlin_gemm``. That makes the comparison layout vs layout
rather than a hand-written call vs a kernel, and the Marlin arm is the same
consumer the server uses.

Shapes are the TP4 per-rank ones: ``w13`` N=320 K=2560 and ``w2`` N=2560 K=160,
8 experts, 4 tokens x top_k 2 = 8 route rows, one route per 8-slot block.

The Marlin arm runs with ``SM70_MOE_DECODE_GEMV=0`` so the reference is the
Marlin kernel itself, not the other custom kernel; the QPN2 arm is selected by
the uint8 weight buffer, not by an env flag.

Skips when ``SM70_MARLIN_DIR`` is unset: the Marlin extension is built outside
this repository (see ``quant/sm70_fp4_moe/PROVENANCE.md``).
"""

import os

import pytest

pytestmark = pytest.mark.gpu

EXPERTS = 8
SHAPES = {"w13": (320, 2560), "w2": (2560, 160)}
TOKENS, TOP_K, BLOCK = 4, 2, 8
ROWS = TOKENS * TOP_K
GLOBAL_SCALE = 0.0123
TOLERANCE = 1e-3
LAYOUT_ENV = "SGLANG_SM70_FP4_MOE_LAYOUT"
TENSORS_ENV = "SGLANG_SM70_QPN2_TENSORS"


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


def _build_bank(torch):
    """Checkpoint-format tensors for both projections, one RNG for all arms."""
    from sglang_restomod.quant.sm70_fp4_moe import check as chk

    torch.manual_seed(0)
    bank = {}
    for prefix, (size_n, size_k) in SHAPES.items():
        packed, scales, codes, dense = [], [], [], []
        for _ in range(EXPERTS):
            w = torch.randn(size_n, size_k, device="cuda", dtype=torch.float16)
            p, s, c = chk._quantize(w)
            packed.append(p)
            scales.append(s)
            codes.append(c.reshape(size_n, size_k))
            dense.append(chk._dequant(p, s).float())
        bank[prefix] = dict(
            weight=torch.stack(packed),
            scales=torch.stack(scales),
            codes=torch.stack(codes),
            dense=torch.stack(dense),
        )
    return bank


def _make_layer(torch, bank, device="cuda"):
    from types import SimpleNamespace

    layer = SimpleNamespace(
        params_dtype=torch.float16,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_config=SimpleNamespace(group_size=16),
        w13_bias=None,
        w2_bias=None,
    )
    for prefix, tensors in bank.items():
        setattr(layer, prefix + "_weight", tensors["weight"].clone())
        setattr(layer, prefix + "_weight_scale", tensors["scales"].clone())
        setattr(
            layer,
            prefix + "_weight_scale_2",
            torch.full((EXPERTS,), GLOBAL_SCALE, dtype=torch.float32, device=device),
        )
    return layer


def _prepare(torch, bank, layout, tensors="both"):
    from sglang_restomod.quant.sm70_fp4_moe.method import prepare_moe_nvfp4

    layer = _make_layer(torch, bank)
    previous = (os.environ.get(LAYOUT_ENV), os.environ.get(TENSORS_ENV))
    os.environ[LAYOUT_ENV] = layout
    os.environ[TENSORS_ENV] = tensors
    try:
        prepare_moe_nvfp4(layer)
    finally:
        for key, value in zip((LAYOUT_ENV, TENSORS_ENV), previous):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return layer


def _routing(torch, device="cuda"):
    sorted_ids = torch.full((ROWS * BLOCK,), ROWS, dtype=torch.int32, device=device)
    for row in range(ROWS):
        sorted_ids[row * BLOCK] = row
    return dict(
        sorted_token_ids=sorted_ids,
        expert_ids=torch.arange(ROWS, dtype=torch.int32, device=device) % EXPERTS,
        padded=torch.tensor([ROWS * BLOCK], dtype=torch.int32, device=device),
        topk_weights=torch.full((ROWS,), 0.1, dtype=torch.float32, device=device),
    )


def _run(torch, layer, routing, prefix, x, size_n, size_k, force_marlin):
    from sgl_kernel.scalar_type import scalar_types

    from sglang_restomod.quant.sm70_fp4_moe.method import (
        MARLIN_SPLIT_K,
        moe_wna16_marlin_gemm,
    )

    previous = (os.environ.get("SM70_MOE_DECODE_GEMV"), os.environ.get(MARLIN_SPLIT_K))
    if force_marlin:
        os.environ["SM70_MOE_DECODE_GEMV"] = "0"
    try:
        out = torch.empty(ROWS, size_n, dtype=torch.float16, device="cuda")
        moe_wna16_marlin_gemm(
            x, out, getattr(layer, prefix + "_weight"), None,
            getattr(layer, prefix + "_weight_scale"),
            getattr(layer, prefix + "_weight_scale_2"),
            None, None, None, torch.empty(0, dtype=torch.int32, device="cuda"),
            routing["sorted_token_ids"], routing["expert_ids"], routing["padded"],
            routing["topk_weights"], BLOCK, TOP_K, False, False,
            scalar_types.float4_e2m1f, TOKENS, size_n, size_k,
        )
        return out
    finally:
        for key, value in zip(("SM70_MOE_DECODE_GEMV", MARLIN_SPLIT_K), previous):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _exact(torch, bank, prefix, x, size_n, size_k):
    """FP32 reference from the source codes and scales, per route row."""
    grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cuda"
    )
    codes = bank[prefix]["codes"]
    scales = bank[prefix]["scales"].to(torch.float32)
    magnitude = grid[codes.abs()]
    values = torch.where(codes < 0, -magnitude, magnitude)
    dense = values * scales.repeat_interleave(16, dim=2) * GLOBAL_SCALE
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.empty(ROWS, size_n, dtype=torch.float32, device="cuda")
        for row in range(ROWS):
            # The kernel recovers the token as route / top_k, matching the
            # align kernel's token * top_k + slot route index.
            expert = row % EXPERTS
            token = row // TOP_K
            out[row] = torch.mv(dense[expert], x[token].float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return out


@pytest.mark.parametrize("tensors", ["both", "w13", "w2"])
def test_qpn2_matches_marlin(tensors):
    torch = _skip_unless_usable()
    from sglang_restomod.quant.sm70_fp4_moe import decode_qpn2

    if decode_qpn2._load() is None:
        pytest.skip("QPN2 GEMV extension did not build")

    bank = _build_bank(torch)
    marlin = _prepare(torch, bank, "marlin")
    qpn2_layer = _prepare(torch, bank, "qpn2", tensors)
    routing = _routing(torch)

    for prefix, (size_n, size_k) in SHAPES.items():
        x = torch.randn(TOKENS, size_k, device="cuda", dtype=torch.float16)
        want = _exact(torch, bank, prefix, x, size_n, size_k)
        got_marlin = _run(torch, marlin, routing, prefix, x, size_n, size_k, True).float()
        got_qpn2 = _run(torch, qpn2_layer, routing, prefix, x, size_n, size_k, False).float()
        torch.cuda.synchronize()

        scale = want.abs().max().item()
        err_marlin = (got_marlin - want).abs().max().item() / scale
        err_qpn2 = (got_qpn2 - want).abs().max().item() / scale
        pairwise = (got_qpn2 - got_marlin).abs().max().item() / scale
        print(
            f"  {prefix} ({tensors}): vs exact marlin={err_marlin:.3e} "
            f"qpn2={err_qpn2:.3e}; pairwise={pairwise:.3e}"
        )
        assert err_qpn2 <= TOLERANCE, (
            f"{prefix}: QPN2 vs exact {err_qpn2:.3e} > {TOLERANCE:.0e}"
        )
        assert pairwise <= TOLERANCE, (
            f"{prefix}: QPN2 vs Marlin {pairwise:.3e} > {TOLERANCE:.0e}"
        )


def test_down_projection_at_the_server_decode_shape():
    """The down GEMM's real decode call: 1 token x top_k 10, size_m = 10.

    ``fused_marlin_moe`` passes ``size_m = M`` for the first GEMM and
    ``size_m = M * topk`` with ``top_k=1`` for the second, so gating the QPN2
    kernel on ``size_m <= 8`` rejected the down projection of an ordinary
    decode batch and sent it to the Marlin fallback on every step. This runs
    that exact call for both layouts and compares them.
    """
    torch = _skip_unless_usable()
    from sgl_kernel.scalar_type import scalar_types

    from sglang_restomod.quant.sm70_fp4_moe import decode_qpn2
    from sglang_restomod.quant.sm70_fp4_moe.method import (
        MARLIN_SPLIT_K,
        moe_wna16_marlin_gemm,
    )

    if decode_qpn2._load() is None:
        pytest.skip("QPN2 GEMV extension did not build")

    tokens, top_k, size_m = 1, 10, 10
    rows = tokens * top_k
    size_n, size_k = SHAPES["w2"]
    bank = _build_bank(torch)
    marlin = _prepare(torch, bank, "marlin")
    qpn2_layer = _prepare(torch, bank, "qpn2", "w2")

    sorted_ids = torch.full((rows * BLOCK,), rows, dtype=torch.int32, device="cuda")
    for row in range(rows):
        sorted_ids[row * BLOCK] = row
    routing = dict(
        sorted_token_ids=sorted_ids,
        expert_ids=torch.arange(rows, dtype=torch.int32, device="cuda") % EXPERTS,
        padded=torch.tensor([rows * BLOCK], dtype=torch.int32, device="cuda"),
        topk_weights=torch.full((rows,), 0.1, dtype=torch.float32, device="cuda"),
    )
    # The second GEMM's activation is the intermediate cache, which has one row
    # per route, not per token; upstream passes the same count as size_m.
    x = torch.randn(rows, size_k, device="cuda", dtype=torch.float16)

    def call(layer, force_marlin):
        previous = (os.environ.get("SM70_MOE_DECODE_GEMV"), os.environ.get(MARLIN_SPLIT_K))
        if force_marlin:
            os.environ["SM70_MOE_DECODE_GEMV"] = "0"
        try:
            out = torch.empty(rows, size_n, dtype=torch.float16, device="cuda")
            moe_wna16_marlin_gemm(
                x, out, layer.w2_weight, None, layer.w2_weight_scale,
                layer.w2_weight_scale_2, None, None, None,
                torch.empty(0, dtype=torch.int32, device="cuda"),
                routing["sorted_token_ids"], routing["expert_ids"], routing["padded"],
                routing["topk_weights"], BLOCK, 1, False, False,
                scalar_types.float4_e2m1f, size_m, size_n, size_k,
            )
            return out
        finally:
            for key, value in zip(("SM70_MOE_DECODE_GEMV", MARLIN_SPLIT_K), previous):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    got_marlin = call(marlin, True).float()
    got_qpn2 = call(qpn2_layer, False).float()
    torch.cuda.synchronize()

    scale = got_marlin.abs().max().item()
    pairwise = (got_qpn2 - got_marlin).abs().max().item() / scale
    print(f"  w2 at rows={rows} size_m={size_m}: pairwise={pairwise:.3e}")
    assert pairwise < TOLERANCE
    assert decode_qpn2.supports(rows, size_n, size_k, 1, BLOCK)
