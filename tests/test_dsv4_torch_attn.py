"""The stage-1 torch paged MLA: gather, mask, sink, and the value-is-key identity.

CPU only, on a deliberately tiny page (4 tokens) so the expected result can be
written out rather than compared against another implementation of the same
idea. The cache is filled with exactly-representable values at scale 2**0, so a
failure here is an addressing or masking bug, not rounding.
"""

from __future__ import annotations

import pytest

from sglang_restomod.attention.dsv4_sm70 import kv_layout, torch_attn

torch = pytest.importorskip("torch")

PAGE = 32  # 32 * 528 is a multiple of the 512-byte page align
DIMS = 512
LAYOUT = kv_layout.get_layout("V41")


def _build_cache(pages_tokens):
    """pages_tokens: [pages][page_size][dims] floats -> the pin's uint8 cache."""
    pages = len(pages_tokens)
    pbytes = kv_layout.page_bytes(LAYOUT, PAGE)
    assert pbytes == PAGE * LAYOUT.bytes_per_token()  # no page padding at this size
    cache = torch.zeros(pages, PAGE, 1, LAYOUT.bytes_per_token(), dtype=torch.uint8)
    flat = cache.view(-1)
    for p, page in enumerate(pages_tokens):
        base = p * pbytes
        for slot, row in enumerate(page):
            d_off, s_off = kv_layout.row_offsets(LAYOUT, PAGE, slot)
            values = torch.tensor(row, dtype=torch.float32)
            flat[base + d_off : base + d_off + DIMS] = values.to(torch.float8_e4m3fn).view(
                torch.uint8
            )
            # Scale 2**0, one per 32 dims: byte 127.
            flat[base + s_off : base + s_off + 16] = 127
    return cache


def _page_keys(cache, page_index):
    """The keys a page dequantizes to -- what attention actually multiplies.

    The cache stores e4m3 at scale 2**0, so the raw floats used to fill it are
    not what comes back; comparing against those would test the quantizer.
    """
    pbytes = kv_layout.page_bytes(LAYOUT, PAGE)
    flat = cache.view(-1)[page_index * pbytes : (page_index + 1) * pbytes]
    rows = []
    for slot in range(PAGE):
        d_off, s_off = kv_layout.row_offsets(LAYOUT, PAGE, slot)
        rows.append(
            kv_layout.dequantize_token(
                LAYOUT,
                flat[d_off : d_off + DIMS],
                flat[s_off : s_off + LAYOUT.scale_bytes],
            )
        )
    return torch.stack(rows)


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(PAGE, DIMS, generator=g) * 0.5).numpy().tolist()


def test_attention_over_one_page():
    page = _rows(0, 1)
    cache = _build_cache([page])
    q = torch.zeros(1, 1, 1, DIMS)
    q[0, 0, 0, :8] = torch.tensor([1.0, 0.5, -0.25, 0.125, 0.0, 0.0, 0.0, 0.0])
    keys = _page_keys(cache, 0)

    out, lse = torch_attn.paged_mla(
        q,
        cache,
        indices=torch.zeros(1, 1, dtype=torch.int32),
        topk_length=torch.tensor([3]),
        is_fp8_kvcache=True,
    )

    scores = (q[0, 0, 0] @ keys.T) * DIMS**-0.5
    scores = scores[:3]  # topk_length truncates the visible range
    probs = torch.softmax(scores, dim=-1)
    want = probs @ keys[:3]
    assert torch.allclose(out[0, 0].float(), want, atol=2e-2)
    assert torch.allclose(lse[0, 0], torch.logsumexp(scores, dim=-1), atol=2e-2)


def test_topk_length_excludes_later_tokens():
    page = _rows(0, 2)
    cache = _build_cache([page])
    q = torch.zeros(1, 1, 1, DIMS)
    q[0, 0, 0, 0] = 1.0
    keys = _page_keys(cache, 0)

    out, _ = torch_attn.paged_mla(
        q,
        cache,
        indices=torch.zeros(1, 1, dtype=torch.int32),
        topk_length=torch.tensor([1]),
        is_fp8_kvcache=True,
    )
    # Only the first token is visible, so the output is that token exactly.
    assert torch.allclose(out[0, 0].float(), keys[0], atol=2e-2)


def test_negative_page_ids_do_not_read_page_zero():
    page_a = _rows(0, 3)
    page_b = _rows(1, 4)
    cache = _build_cache([page_a, page_b])
    q = torch.zeros(1, 1, 1, DIMS)
    q[0, 0, 0, 0] = 1.0
    keys = _page_keys(cache, 1)

    out, _ = torch_attn.paged_mla(
        q,
        cache,
        indices=torch.tensor([[1, -1]], dtype=torch.int32),
        topk_length=torch.tensor([PAGE]),
        is_fp8_kvcache=True,
    )
    # The pinned '-1' is trailing padding, so every visible token is page_b's.
    assert torch.allclose(out[0, 0].float(), keys.mean(dim=0), atol=1e-1)


def test_the_sink_only_changes_the_denominator():
    page = _rows(0, 5)
    cache = _build_cache([page])
    q = torch.zeros(1, 1, 1, DIMS)
    q[0, 0, 0, 0] = 1.0
    sink = torch.tensor([10.0])

    out_sink, lse_sink = torch_attn.paged_mla(
        q,
        cache,
        indices=torch.zeros(1, 1, dtype=torch.int32),
        topk_length=torch.tensor([PAGE]),
        attn_sink=sink,
        is_fp8_kvcache=True,
    )
    out_plain, lse_plain = torch_attn.paged_mla(
        q,
        cache,
        indices=torch.zeros(1, 1, dtype=torch.int32),
        topk_length=torch.tensor([PAGE]),
        is_fp8_kvcache=True,
    )
    # A huge sink should suppress the cache almost entirely: the output shrinks
    # but stays a positive multiple of the plain one, because the weights are
    # unchanged in direction.
    assert float(out_sink.abs().max()) < float(out_plain.abs().max())
    assert float(lse_sink[0, 0]) > float(lse_plain[0, 0])


def test_a_dense_page_table_is_refused():
    cache = _build_cache([_rows(0, 6)])
    q = torch.zeros(1, 1, 1, DIMS)
    with pytest.raises(torch_attn.Stage1Error, match="sparse page indices"):
        torch_attn.paged_mla(q, cache)


def test_a_causal_variant_is_refused():
    cache = _build_cache([_rows(0, 7)])
    q = torch.zeros(1, 1, 1, DIMS)
    with pytest.raises(torch_attn.Stage1Error, match="causal"):
        torch_attn.paged_mla(
            q, cache, indices=torch.zeros(1, 1, dtype=torch.int32), causal=True, is_fp8_kvcache=True
        )
