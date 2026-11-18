"""GPU test: the fused HC prenorm against the shim's own reference arm.

Agreement, not speed. The kernel widens x and fn to fp32 in registers, so it
should meet the reference to fp32-accumulation order; the tolerance below is a
few times that, tight enough that a wrong reduction fails.

Determinism is checked as a separate property: two runs must be bit-identical,
which is what the no-atomics split-K design promises.
"""

import pytest

pytestmark = pytest.mark.gpu

MIX, HC_MULT, HIDDEN = 24, 4, 5120
HC = HC_MULT * HIDDEN


def _skip_unless_usable():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    pytest.importorskip("triton")
    return torch


def _reference(torch, x, fn, out, sqrsum):
    xf = x.float()
    torch.mm(xf, fn.float().t(), out=out)
    torch.sum(xf * xf, dim=-1, out=sqrsum)


@pytest.mark.parametrize("t", [1, 4, 32, 256, 1024, 4096])
def test_matches_the_reference(t):
    torch = _skip_unless_usable()
    from sglang_restomod.kernels import hc_prenorm_sm70

    torch.manual_seed(0)
    x = (torch.randn(t, HC, dtype=torch.float32, device="cuda") * 0.5).to(torch.bfloat16)
    fn = torch.randn(MIX, HC, dtype=torch.float32, device="cuda") * 0.02

    want_out = torch.empty(t, MIX, dtype=torch.float32, device="cuda")
    want_ss = torch.empty(t, dtype=torch.float32, device="cuda")
    _reference(torch, x, fn, want_out, want_ss)

    out = torch.empty(t, MIX, dtype=torch.float32, device="cuda")
    ss = torch.empty(t, dtype=torch.float32, device="cuda")
    hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)
    torch.cuda.synchronize()

    scale = float(want_out.abs().max())
    assert float((out - want_out).abs().max()) <= 2e-4 * scale
    ss_scale = float(want_ss.abs().max())
    assert float((ss - want_ss).abs().max()) <= 2e-4 * ss_scale


def test_two_runs_are_bit_identical():
    torch = _skip_unless_usable()
    from sglang_restomod.kernels import hc_prenorm_sm70

    t = 512
    x = (torch.randn(t, HC, device="cuda") * 0.5).to(torch.bfloat16)
    fn = torch.randn(MIX, HC, device="cuda") * 0.02
    a_out = torch.empty(t, MIX, dtype=torch.float32, device="cuda")
    a_ss = torch.empty(t, dtype=torch.float32, device="cuda")
    b_out = torch.empty(t, MIX, dtype=torch.float32, device="cuda")
    b_ss = torch.empty(t, dtype=torch.float32, device="cuda")

    hc_prenorm_sm70.hc_prenorm(x, fn, a_out, a_ss)
    hc_prenorm_sm70.hc_prenorm(x, fn, b_out, b_ss)
    torch.cuda.synchronize()

    assert torch.equal(a_out, b_out)
    assert torch.equal(a_ss, b_ss)


def test_registered_op_is_what_the_shim_calls(monkeypatch):
    """The switch, end to end: ``fused`` reaches this kernel, not a fallback."""
    torch = _skip_unless_usable()
    from sglang_restomod.kernels import deep_gemm_sm70 as shim
    from sglang_restomod.kernels import hc_prenorm_sm70

    monkeypatch.setenv(shim.HC_PRENORM_ENV, "fused")
    hc_prenorm_sm70.register()

    t = 32
    x = (torch.randn(t, HC, device="cuda") * 0.5).to(torch.bfloat16)
    fn = torch.randn(MIX, HC, device="cuda") * 0.02
    want_out = torch.empty(t, MIX, dtype=torch.float32, device="cuda")
    want_ss = torch.empty(t, dtype=torch.float32, device="cuda")
    _reference(torch, x, fn, want_out, want_ss)

    out = torch.empty(t, MIX, dtype=torch.float32, device="cuda")
    ss = torch.empty(t, dtype=torch.float32, device="cuda")
    shim.tf32_hc_prenorm_gemm(x, fn, out, ss)
    torch.cuda.synchronize()
    # An absolute floor matters: a logit near zero has no relative tolerance.
    assert torch.allclose(out, want_out, rtol=2e-4, atol=2e-4 * float(want_out.abs().max()))
