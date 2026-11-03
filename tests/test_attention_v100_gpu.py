"""GPU tests for the attention_v100 paged extend kernel.

The CPU tests cover the dispatch (which backend a forward mode selects). These
cover what the selected kernel computes: a paged, causal extend against
`torch.nn.functional.scaled_dot_product_attention` on the same gathered KV.

They run from `scripts/gpu_suite.sh`, which records `gpu_tests.txt`, and skip
unless the device is SM70.
"""

import math

import pytest

pytestmark = pytest.mark.gpu


def _require_sm70():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    return torch


def _scaled_dot_product_reference(torch, q, k, v, prefix):
    """Attend ``q`` over ``k``/``v`` causally, with ``q`` starting at ``prefix``.

    Shapes are the kernel's: ``q`` is ``[extend, heads, dim]`` and ``k``/``v``
    are ``[prefix + extend, heads_kv, dim]``.
    """
    extend, heads, dim = q.shape
    total = k.shape[0]
    rep = heads // k.shape[1]
    kk = k.transpose(0, 1).unsqueeze(0).repeat_interleave(rep, dim=1)
    vv = v.transpose(0, 1).unsqueeze(0).repeat_interleave(rep, dim=1)
    qq = q.transpose(0, 1).unsqueeze(0)
    query_pos = torch.arange(prefix, total, device=q.device)[:, None]
    key_pos = torch.arange(total, device=q.device)[None, :]
    mask = (key_pos <= query_pos).unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(
        qq, kk, vv, attn_mask=mask, scale=1.0 / math.sqrt(dim)
    )
    return out.squeeze(0).transpose(0, 1)


def _paged_extend(torch, prefix, extend, heads=4, heads_kv=2, dim=256, block=16):
    from sglang_restomod.attention.v100.tilelang import paged_forward

    torch.manual_seed(0)
    dev = "cuda"
    total = prefix + extend
    blocks = (total + block - 1) // block
    k_cache = torch.randn(blocks, block, heads_kv, dim, dtype=torch.float16, device=dev)
    v_cache = torch.randn(blocks, block, heads_kv, dim, dtype=torch.float16, device=dev)
    q = torch.randn(extend, heads, dim, dtype=torch.float16, device=dev)
    block_table = torch.arange(blocks, dtype=torch.int32, device=dev).unsqueeze(0)
    seq_lens = torch.tensor([total], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, extend], dtype=torch.int32, device=dev)
    prefix_kv_lens = torch.tensor([prefix], dtype=torch.int32, device=dev)

    out = torch.empty_like(q)
    paged_forward(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        prefix_kv_lens,
        out=out,
        block_size=block,
        softmax_scale=1.0 / math.sqrt(dim),
        causal=True,
        num_kv_heads=heads_kv,
    )

    # The reference reads the same cache through the trivial page table.
    k = k_cache.reshape(blocks * block, heads_kv, dim)[:total]
    v = v_cache.reshape(blocks * block, heads_kv, dim)[:total]
    return out, _scaled_dot_product_reference(torch, q, k, v, prefix)


def test_extend_matches_reference_with_a_prefix():
    torch = _require_sm70()
    out, ref = _paged_extend(torch, prefix=64, extend=32)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_extend_matches_reference_without_a_prefix():
    torch = _require_sm70()
    out, ref = _paged_extend(torch, prefix=0, extend=48)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
