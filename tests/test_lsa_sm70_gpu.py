"""GPU tests for the LSA SM70 index and read.

Three claims are checked on the device:

- the bucket build gives the CPU oracle's buckets;
- the vectorized selection gives the CPU oracle's lists;
- the Triton read gives the same attention as a dense gather over the same slots.

They run from `scripts/gpu_suite.sh` and skip unless the device is SM70.
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


def _unit_rows(torch, rows, dim, seed, device="cuda"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.randn(rows, dim, generator=generator)
    keys = keys / keys.norm(dim=1, keepdim=True)
    return keys.to(device)


def _dense_read(torch, q, k_buffer, v_buffer, lists, lengths, scale):
    """Attend each row over the slots its list names, in plain torch."""
    rows, heads, dim = q.shape
    kv_heads = k_buffer.shape[1]
    group = heads // kv_heads
    out = torch.zeros(rows, heads, v_buffer.shape[-1], dtype=torch.float32, device=q.device)
    for row in range(rows):
        count = int(lengths[row])
        slots = lists[row, :count].long()
        keys = k_buffer[slots].float()
        values = v_buffer[slots].float()
        query = q[row].float().view(kv_heads, group, dim)
        logits = torch.einsum("kgd,jkd->jkg", query, keys) * scale
        probs = torch.softmax(logits, dim=0)
        merged = torch.einsum("jkg,jkd->kgd", probs, values)
        out[row] = merged.reshape(heads, -1)
    return out


def test_bucket_build_on_cuda_matches_the_cpu_oracle():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.reference import build_buckets

    keys = _unit_rows(torch, 192, 16, 0)
    sizes = dict(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=64)

    on_cpu = build_buckets(keys.cpu(), **sizes)
    on_cuda = build_buckets(keys, **sizes)

    assert on_cuda.bucket_centroids.is_cuda
    assert on_cuda.bucket_members.is_cuda
    assert torch.equal(on_cuda.bucket_members.cpu(), on_cpu.bucket_members)
    assert torch.allclose(on_cuda.bucket_centroids.cpu(), on_cpu.bucket_centroids, atol=1e-5)


def test_selection_on_cuda_matches_the_cpu_oracle():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.reference import build_buckets, read
    from sglang_restomod.attention.lsa_sm70.selection import select_lists

    keys = _unit_rows(torch, 192, 16, 1)
    queries = _unit_rows(torch, 192, 16, 2)
    positions = torch.arange(192, dtype=torch.int32)
    sizes = dict(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=64)
    buckets = build_buckets(keys, **sizes)

    selected, lengths = select_lists(
        queries,
        positions.cuda(),
        buckets,
        top_p=16,
        top_k=32,
        sink_size=2,
        window_size=64,
    )
    expected, expected_lengths = read(
        queries.cpu(),
        positions,
        build_buckets(keys.cpu(), **sizes),
        top_p=16,
        top_k=32,
        sink_size=2,
        window_size=64,
    )

    assert torch.equal(lengths.cpu(), expected_lengths)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(selected[row, :length].cpu(), expected[row, :length])


def test_sparse_read_matches_a_dense_gather():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.read import lsa_sparse_read

    torch.manual_seed(0)
    slots, kv_heads, head_dim, group, rows = 512, 2, 64, 2, 8
    heads = kv_heads * group
    kv = torch.randn(slots, kv_heads, head_dim, dtype=torch.float16, device="cuda")
    q = torch.randn(rows, heads, head_dim, dtype=torch.float16, device="cuda")
    width = 96
    lists = torch.randint(0, slots, (rows, width), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, width + 1, (rows,), dtype=torch.int32, device="cuda")
    lists = torch.where(
        torch.arange(width, device="cuda")[None, :] < lengths[:, None], lists, -1
    ).to(torch.int32)
    scale = 1.0 / math.sqrt(head_dim)

    out = lsa_sparse_read(q, kv, kv, lists, lengths, scale)
    expected = _dense_read(torch, q, kv, kv, lists, lengths, scale)

    assert torch.allclose(out.float(), expected, atol=2e-2, rtol=2e-2)


def test_selected_lists_read_matches_a_dense_gather():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.read import lsa_sparse_read
    from sglang_restomod.attention.lsa_sm70.reference import build_buckets
    from sglang_restomod.attention.lsa_sm70.selection import select_lists

    torch.manual_seed(1)
    slots, kv_heads, head_dim, group, rows = 192, 2, 64, 2, 16
    heads = kv_heads * group
    keys = _unit_rows(torch, slots, 16, 3)
    queries = _unit_rows(torch, rows, 16, 4)
    positions = torch.arange(160, 160 + rows, dtype=torch.int64, device="cuda")
    sizes = dict(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=64)
    buckets = build_buckets(keys, **sizes)
    selected, lengths = select_lists(
        queries, positions, buckets, top_p=16, top_k=32, sink_size=2, window_size=64
    )

    kv = torch.randn(slots, kv_heads, head_dim, dtype=torch.float16, device="cuda")
    q = torch.randn(rows, heads, head_dim, dtype=torch.float16, device="cuda")
    scale = 1.0 / math.sqrt(head_dim)

    out = lsa_sparse_read(q, kv, kv, selected, lengths, scale)
    expected = _dense_read(torch, q, kv, kv, selected, lengths, scale)

    assert torch.allclose(out.float(), expected, atol=2e-2, rtol=2e-2)


def test_triton_selection_matches_the_cpu_oracle_at_the_serving_shape():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.kernels import select_lists_triton
    from sglang_restomod.attention.lsa_sm70.reference import build_buckets, read

    sizes = dict(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=512)
    keys = _unit_rows(torch, 2048, 64, 5)
    queries = _unit_rows(torch, 8, 64, 6)
    positions = torch.tensor([0, 1, 511, 512, 1023, 1024, 1500, 2047], dtype=torch.int64)
    buckets = build_buckets(keys, **sizes)

    selected, lengths = select_lists_triton(
        queries, positions.cuda(), buckets, top_p=512, top_k=2048, sink_size=4, window_size=512
    )
    expected, expected_lengths = read(
        queries.cpu(),
        positions,
        build_buckets(keys.cpu(), **sizes),
        top_p=512,
        top_k=2048,
        sink_size=4,
        window_size=512,
    )

    assert torch.equal(lengths.cpu(), expected_lengths)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(selected[row, :length].cpu(), expected[row, :length])


def test_triton_selection_matches_the_torch_path_when_top_p_exceeds_the_buckets():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.kernels import select_lists_triton
    from sglang_restomod.attention.lsa_sm70.reference import build_buckets
    from sglang_restomod.attention.lsa_sm70.selection import select_lists

    sizes = dict(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=64)
    keys = _unit_rows(torch, 192, 64, 7)
    queries = _unit_rows(torch, 16, 64, 8)
    positions = torch.arange(0, 192, 12, dtype=torch.int64, device="cuda")
    buckets = build_buckets(keys, **sizes)

    on_kernels = select_lists_triton(
        queries, positions, buckets, top_p=16, top_k=32, sink_size=2, window_size=64
    )
    on_torch = select_lists(
        queries,
        positions,
        buckets,
        top_p=16,
        top_k=32,
        sink_size=2,
        window_size=64,
        use_kernels=False,
    )

    assert torch.equal(on_kernels[1], on_torch[1])
    for row, length in enumerate(on_kernels[1].tolist()):
        assert torch.equal(on_kernels[0][row, :length], on_torch[0][row, :length])


def test_triton_selection_with_no_closed_blocks():
    torch = _require_sm70()
    from sglang_restomod.attention.lsa_sm70.kernels import select_lists_triton
    from sglang_restomod.attention.lsa_sm70.reference import Buckets, read

    empty = Buckets(
        torch.empty(0, 64, device="cuda"),
        torch.empty(0, 16, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        64,
        512,
    )
    queries = _unit_rows(torch, 4, 64, 9)
    positions = torch.arange(4, dtype=torch.int64, device="cuda")

    selected, lengths = select_lists_triton(
        queries, positions, empty, top_p=16, top_k=64, sink_size=2, window_size=32
    )
    expected, expected_lengths = read(
        queries.cpu(),
        positions.cpu(),
        Buckets(
            torch.empty(0, 64),
            torch.empty(0, 16, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            64,
            512,
        ),
        top_p=16,
        top_k=64,
        sink_size=2,
        window_size=32,
    )

    assert torch.equal(lengths.cpu(), expected_lengths)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(selected[row, :length].cpu(), expected[row, :length])
