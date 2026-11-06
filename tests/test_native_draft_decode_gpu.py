"""GPU tests for the native draft-decode page-table path.

The CPU tests cover the conversion from a step's flat index list to a page
table. These run the native split-KV decode kernel on that table and compare
it with an explicit gather plus scaled dot-product attention.

They run from ``scripts/gpu_suite.sh``, which records ``gpu_tests.txt``, and
skip unless the device is SM70.
"""

import math

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
# Installs Logger.info_once, which the vendored CUDA decode loader calls.
import sglang.srt.utils.common  # noqa: E402,F401


def _require_sm70():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    return torch


def _row_tokens(pages, page_size, length):
    tokens = []
    for page in pages:
        take = min(page_size, length - len(tokens))
        tokens.extend(page * page_size + offset for offset in range(take))
    assert len(tokens) == length
    return tokens


def _reference(torch, q, k_cache, v_cache, index_rows, scale):
    """Per-head reference with the kernel's head mapping: head // group -> KV head."""
    rows, heads, dim = q.shape
    heads_kv = k_cache.shape[2]
    group = heads // heads_kv
    out = torch.empty_like(q)
    for row, token_ids in enumerate(index_rows):
        ids = torch.tensor(token_ids, dtype=torch.int64, device=q.device)
        k_all = k_cache.reshape(-1, heads_kv, dim)[ids]
        v_all = v_cache.reshape(-1, heads_kv, dim)[ids]
        for head in range(heads):
            kv = head // group
            out[row, head] = torch.nn.functional.scaled_dot_product_attention(
                q[row, head].reshape(1, 1, 1, dim),
                k_all[:, kv, :].reshape(1, 1, -1, dim),
                v_all[:, kv, :].reshape(1, 1, -1, dim),
                scale=scale,
            ).squeeze()
    return out


def _case(torch, lengths, dtype=torch.float16, heads=12, heads_kv=2, dim=256, page_size=16):
    from sglang_restomod.attention.v100 import draft_decode_plan as plan
    from sglang_restomod.attention.v100.tilelang import grouped_decode_forward

    dev = "cuda"
    torch.manual_seed(0)
    pages_per_row = [(length + page_size - 1) // page_size for length in lengths]
    total_pages = sum(pages_per_row)
    k_cache = torch.randn(
        total_pages, page_size, heads_kv, dim, dtype=torch.float16, device=dev
    ).to(dtype)
    v_cache = k_cache.clone()
    q = torch.randn(len(lengths), heads, dim, dtype=torch.float16, device=dev)

    # Assign disjoint physical pages to rows, in reverse logical order, so a
    # page-table mistake cannot pass by accident.
    index_rows = []
    next_page = 0
    for length, count in zip(lengths, pages_per_row):
        pages = list(range(next_page, next_page + count))[::-1]
        index_rows.append(_row_tokens(pages, page_size, length))
        next_page += count

    kv_indices = torch.tensor(
        [token for row in index_rows for token in row], dtype=torch.int64, device=dev
    )
    lengths_t = torch.tensor(
        [len(row) for row in index_rows], dtype=torch.int32, device=dev
    )
    kv_indptr = torch.zeros((len(lengths) + 1,), dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(lengths_t, dim=0)

    max_pages = max(pages_per_row)
    page_table = torch.zeros((len(lengths), max_pages), dtype=torch.int32, device=dev)
    seq_lens = torch.zeros((len(lengths),), dtype=torch.int32, device=dev)
    rows = plan.page_table_from_indices(
        kv_indptr, kv_indices, page_table, seq_lens, page_size
    )
    assert rows == len(lengths)
    assert seq_lens.tolist() == lengths
    want_pages = [
        [row[i * page_size] // page_size for i in range(count)]
        + [0] * (max_pages - count)
        for row, count in zip(index_rows, pages_per_row)
    ]
    assert page_table.tolist() == want_pages

    scale = 1.0 / math.sqrt(dim)
    got = grouped_decode_forward(
        q, k_cache, v_cache, page_table, seq_lens, softmax_scale=scale
    )
    want = _reference(
        torch, q, k_cache.to(torch.float16), v_cache.to(torch.float16), index_rows, scale
    )
    return got, want


def test_native_decode_matches_a_gathered_reference():
    torch = _require_sm70()
    got, want = _case(torch, [37, 53])
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_native_decode_matches_a_single_row_reference():
    torch = _require_sm70()
    got, want = _case(torch, [67])
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_native_decode_matches_an_e5m2_reference():
    torch = _require_sm70()
    got, want = _case(torch, [41, 67], dtype=torch.float8_e5m2)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
