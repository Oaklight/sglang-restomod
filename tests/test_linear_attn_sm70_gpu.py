"""GPU test: the TileLang GDN pair against the pinned Triton kernels.

The pair replaces two functions whose contracts include a state pool with
per-sequence slots and an in-place final-state write, so agreeing on the math
is not enough: the state has to land in the same slot the Triton kernel would
have written. The check runs both implementations on the same inputs and
compares the outputs, the per-chunk states and the final pool.

Skips off SM70 and when TileLang or the pinned engine is not importable.
"""

import pytest

pytestmark = pytest.mark.gpu

B, T, HG, H, K, V = 1, 256, 2, 4, 64, 64


def _skip_unless_usable():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    pytest.importorskip("tilelang")
    try:
        import sglang.kernels.ops.attention.fla.chunk  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the pinned engine is not importable: {exc}")
    from sglang_restomod.linear_attn_sm70 import gdn_pair

    return torch, gdn_pair


def _assert_close(got, ref, what: str):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    big = ref.abs() >= 1e-2
    rel = (err[big] / ref.abs()[big]).max().item() if bool(big.any()) else 0.0
    small = err[~big].max().item() if bool((~big).any()) else 0.0
    assert rel < 2e-2, f"{what}: max relative error {rel:.3e}"
    assert small < 1e-2, f"{what}: max error below the relative band {small:.3e}"


def _chunk_cumsum(torch, decay):
    """Chunk-local cumsum of the log decay, for both a full and a partial tail."""
    t_len = decay.shape[1]
    chunks = -(-t_len // 64)
    pad = chunks * 64 - t_len
    padded = torch.nn.functional.pad(decay, (0, 0, 0, pad), value=0.0)
    cum = padded.view(B, chunks, 64, H).cumsum(dim=2)
    return cum.reshape(B, chunks * 64, H)[:, :t_len].contiguous()


def _chunk_o_inputs(torch, t_len=T):
    torch.manual_seed(0)
    q = torch.randn(B, t_len, HG, K, device="cuda", dtype=torch.float16)
    k = torch.nn.functional.normalize(
        torch.randn(B, t_len, HG, K, device="cuda", dtype=torch.float16).float(), dim=-1
    ).half()
    v = torch.randn(B, t_len, H, V, device="cuda", dtype=torch.float16)
    h = torch.randn(B, -(-t_len // 64), H, V, K, device="cuda", dtype=torch.float16) * 0.05
    decay = -torch.rand(B, t_len, H, device="cuda", dtype=torch.float32) * 0.05
    g = _chunk_cumsum(torch, decay)
    return q, k, v, h, g


def _chunk_h_inputs(torch, t_len=T):
    torch.manual_seed(1)
    k = torch.nn.functional.normalize(
        torch.randn(B, t_len, HG, K, device="cuda", dtype=torch.float16).float(), dim=-1
    ).half()
    w = torch.randn(B, t_len, H, K, device="cuda", dtype=torch.float16) * (K**-0.5)
    u = torch.randn(B, t_len, H, V, device="cuda", dtype=torch.float16) * 0.5
    decay = -torch.nn.functional.softplus(
        torch.randn(B, t_len, H, device="cuda", dtype=torch.float32) * 1.5
    )
    g = _chunk_cumsum(torch, decay)
    pool = torch.randn(3, H, V, K, device="cuda", dtype=torch.float16) * 0.05
    return k, w, u, g, pool


def test_chunk_o_matches_triton():
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o as triton_chunk_o

    q, k, v, h, g = _chunk_o_inputs(torch)
    ref = triton_chunk_o(q, k, v, h, g, scale=K**-0.5)
    gdn_pair._ORIGINALS["chunk_fwd_o"] = triton_chunk_o
    got = gdn_pair.chunk_fwd_o(q, k, v, h, g=g, scale=K**-0.5)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    _assert_close(got, ref, "chunk_o")


def test_chunk_h_matches_triton_and_writes_the_same_pool_slot():
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as triton_fwd_h,
    )

    k, w, u, g, pool = _chunk_h_inputs(torch)
    # Slot 2 is live, slot 0 is the pin's "no state" sentinel.
    indices = torch.tensor([2], device="cuda", dtype=torch.int64)
    ref_pool = pool.clone()
    got_pool = pool.clone()

    ref_h, ref_v = triton_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=ref_pool, initial_state_indices=indices
    )
    gdn_pair._ORIGINALS["chunk_gated_delta_rule_fwd_h"] = triton_fwd_h
    got_h, got_v = gdn_pair.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=got_pool, initial_state_indices=indices
    )

    assert got_h.shape == ref_h.shape and got_v.shape == ref_v.shape
    _assert_close(got_h, ref_h, "chunk_h states")
    _assert_close(got_v, ref_v, "chunk_h v_new")
    _assert_close(got_pool[2], ref_pool[2], "chunk_h final state")
    for slot in (0, 1):
        torch.testing.assert_close(got_pool[slot], pool[slot])


def test_chunk_o_matches_triton_with_a_single_span():
    """Serving always passes cu_seqlens; the single-span call must agree too."""
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o as triton_chunk_o

    q, k, v, h, g = _chunk_o_inputs(torch)
    spans = torch.tensor([0, T], device="cuda", dtype=torch.int64)
    ref = triton_chunk_o(q, k, v, h, g, scale=K**-0.5, cu_seqlens=spans)
    gdn_pair._ORIGINALS["chunk_fwd_o"] = triton_chunk_o
    got = gdn_pair.chunk_fwd_o(q, k, v, h, g=g, scale=K**-0.5, cu_seqlens=spans)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    _assert_close(got, ref, "chunk_o single span")


def test_chunk_h_matches_triton_with_a_single_span():
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as triton_fwd_h,
    )

    k, w, u, g, pool = _chunk_h_inputs(torch)
    spans = torch.tensor([0, T], device="cuda", dtype=torch.int64)
    indices = torch.tensor([2], device="cuda", dtype=torch.int64)
    ref_pool, got_pool = pool.clone(), pool.clone()

    ref_h, ref_v = triton_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=ref_pool,
        initial_state_indices=indices,
        cu_seqlens=spans,
    )
    gdn_pair._ORIGINALS["chunk_gated_delta_rule_fwd_h"] = triton_fwd_h
    got_h, got_v = gdn_pair.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=got_pool,
        initial_state_indices=indices,
        cu_seqlens=spans,
    )

    assert got_h.shape == ref_h.shape and got_v.shape == ref_v.shape
    _assert_close(got_h, ref_h, "chunk_h single span states")
    _assert_close(got_v, ref_v, "chunk_h single span v_new")
    _assert_close(got_pool[2], ref_pool[2], "chunk_h single span final state")


def test_chunk_o_matches_triton_on_a_partial_tail():
    """A chunked prefill ends on a partial chunk; the padded call must still agree."""
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o as triton_chunk_o

    q, k, v, h, g = _chunk_o_inputs(torch, 300)
    ref = triton_chunk_o(q, k, v, h, g, scale=K**-0.5)
    gdn_pair._ORIGINALS["chunk_fwd_o"] = triton_chunk_o
    got = gdn_pair.chunk_fwd_o(q, k, v, h, g=g, scale=K**-0.5)

    assert got.shape == ref.shape
    _assert_close(got, ref, "chunk_o partial tail")


def test_chunk_h_matches_triton_on_a_partial_tail():
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as triton_fwd_h,
    )

    k, w, u, g, pool = _chunk_h_inputs(torch, 300)
    indices = torch.tensor([2], device="cuda", dtype=torch.int64)
    ref_pool, got_pool = pool.clone(), pool.clone()

    ref_h, ref_v = triton_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=ref_pool, initial_state_indices=indices
    )
    gdn_pair._ORIGINALS["chunk_gated_delta_rule_fwd_h"] = triton_fwd_h
    got_h, got_v = gdn_pair.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=got_pool, initial_state_indices=indices
    )

    assert got_h.shape == ref_h.shape and got_v.shape == ref_v.shape
    _assert_close(got_h, ref_h, "chunk_h partial tail states")
    _assert_close(got_v, ref_v, "chunk_h partial tail v_new")
    _assert_close(got_pool[2], ref_pool[2], "chunk_h partial tail final state")


def test_install_reaches_the_engine_caller():
    torch, gdn_pair = _skip_unless_usable()
    from sglang.kernels.ops.attention.fla import chunk as fla_chunk
    from sglang.kernels.ops.attention.fla import chunk_delta_h, chunk_o

    if gdn_pair._INSTALLED:
        pytest.skip("the component was already installed by the plugin loader")
    assert gdn_pair.install() is True
    assert fla_chunk.chunk_fwd_o is gdn_pair.chunk_fwd_o
    assert fla_chunk.chunk_gated_delta_rule_fwd_h is gdn_pair.chunk_gated_delta_rule_fwd_h
    assert chunk_o.chunk_fwd_o is gdn_pair.chunk_fwd_o
    assert chunk_delta_h.chunk_gated_delta_rule_fwd_h is gdn_pair.chunk_gated_delta_rule_fwd_h
