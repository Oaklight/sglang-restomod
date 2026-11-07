"""GPU guard: the QPN2 prefill shim must reproduce the Marlin preparation.

The shipping Marlin layout is the reference every non-QPN2 arm runs, so the
shim that stands in for it during prefill has to produce the same three
buffers bit for bit: the ``[E, K/16, 2N]`` int32 words, the encoded
``[E, K/16, N]`` scale bytes and the encoded global scale.

The existing repack guard compares the fused kernel against a Python reference
written for it. This one compares it against ``prepare_moe_nvfp4`` on the
Marlin layout for the same weights, which is the path the control arm actually
uses. Shapes and expert count are the production ones (TP4 of
Qwen3.8-Flash-Next-NVFP4: 128 experts, ``w13`` N=320 K=2560, ``w2``
N=2560 K=160), not the 4-expert bank ``check.run`` builds: a coverage or
indexing mistake that only appears at the real expert count is invisible to a
4-expert bank.

Two layers are prepared and called in prepare order (A.w13, A.w2, B.w13,
B.w2) so that B.w13 arrives through the shim's prefetch path -- the side-stream
repack issued while A.w2's GEMM runs -- rather than the synchronous first-use
path. Both are checked.

Skips when ``SM70_MARLIN_DIR`` is unset: the Marlin extension is built outside
this repository (see ``quant/sm70_fp4_moe/PROVENANCE.md``).
"""

import os

import pytest

pytestmark = pytest.mark.gpu

EXPERTS = 128
SHAPES = {"w13": (320, 2560), "w2": (2560, 160)}
GLOBAL_SCALE = 4.53e-4  # the served checkpoint's range, measured from the shards
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


def _bank(torch, seed):
    """Checkpoint-format tensors for both projections, one generator."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bank = {}
    for prefix, (size_n, size_k) in SHAPES.items():
        packed = torch.randint(
            0, 256, (EXPERTS, size_n, size_k // 2),
            dtype=torch.uint8, device="cuda", generator=gen,
        )
        # Plausible group scales: e4m3 values in [1e-4, 0.2], no NaN encodings,
        # so a bitwise comparison is well defined.
        raw = torch.rand(
            (EXPERTS, size_n, size_k // 16), device="cuda", generator=gen,
        )
        scales = (raw * 0.1999 + 1e-4).to(torch.float8_e4m3fn)
        bank[prefix] = (packed, scales)
    return bank


def _make_layer(torch, bank):
    from types import SimpleNamespace

    layer = SimpleNamespace(
        params_dtype=torch.float16,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_config=SimpleNamespace(group_size=16),
        w13_bias=None,
        w2_bias=None,
    )
    for prefix, (packed, scales) in bank.items():
        setattr(layer, prefix + "_weight", packed.clone())
        setattr(layer, prefix + "_weight_scale", scales.clone())
        setattr(
            layer,
            prefix + "_weight_scale_2",
            torch.full((EXPERTS,), GLOBAL_SCALE, dtype=torch.float32, device="cuda"),
        )
    return layer


def _prepare(torch, bank, layout):
    from sglang_restomod.quant.sm70_fp4_moe.method import prepare_moe_nvfp4

    layer = _make_layer(torch, bank)
    previous = os.environ.get(LAYOUT_ENV), os.environ.get(TENSORS_ENV)
    os.environ[LAYOUT_ENV] = layout
    os.environ[TENSORS_ENV] = "both"
    try:
        prepare_moe_nvfp4(layer)
    finally:
        for key, value in zip((LAYOUT_ENV, TENSORS_ENV), previous):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return layer


def _reset_shim():
    from sglang_restomod.quant.sm70_fp4_moe import qpn2_repack

    qpn2_repack.reset()
    qpn2_repack._SCRATCH.clear()
    qpn2_repack._OWNER.clear()
    qpn2_repack._LOGGED.clear()


def _first_difference(torch, got, want):
    """Index and values of the first element that differs, or None."""
    if got.shape != want.shape:
        return f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    bad = (got != want).reshape(-1)
    if not bool(bad.any()):
        return None
    flat = bad.nonzero()[0].item()
    return (
        f"{int(bad.sum())} of {bad.numel()} elements differ; first at flat "
        f"{flat}: got {got.reshape(-1)[flat].item()} want {want.reshape(-1)[flat].item()}"
    )


def test_the_shim_reproduces_the_marlin_preparation_bit_for_bit():
    torch = _skip_unless_usable()
    from sglang_restomod.quant.sm70_fp4_moe import qpn2_repack

    banks = [_bank(torch, seed) for seed in (0, 1)]
    references = [_prepare(torch, bank, "marlin") for bank in banks]
    _reset_shim()
    subjects = [_prepare(torch, bank, "qpn2") for bank in banks]

    # Prepare order, then one call per prefix: B.w13 is served from the
    # prefetch issued during A.w2's shim call.
    for index, (subject, reference) in enumerate(zip(subjects, references)):
        for prefix, (size_n, size_k) in SHAPES.items():
            marlin = qpn2_repack.prefill_shim(
                getattr(subject, prefix + "_weight"),
                getattr(subject, prefix + "_weight_scale_2"),
                size_n,
                size_k,
            )
            assert marlin is not None, "the shim declined a production shape"
            weight, scales, global_scale = marlin

            tag = f"layer {index} {prefix} N={size_n} K={size_k}"
            assert torch.equal(weight, getattr(reference, prefix + "_weight")), (
                f"{tag}: repacked words differ from the Marlin preparation: "
                + str(_first_difference(torch, weight, getattr(reference, prefix + "_weight")))
            )
            ref_scales = getattr(reference, prefix + "_weight_scale")
            assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8)), (
                f"{tag}: encoded scales differ from the Marlin preparation: "
                + str(_first_difference(
                    torch, scales.view(torch.uint8), ref_scales.view(torch.uint8)))
            )
            ref_global = getattr(reference, prefix + "_weight_scale_2")
            assert torch.equal(global_scale, ref_global), (
                f"{tag}: encoded global scale differs: "
                + str(_first_difference(torch, global_scale, ref_global))
            )
