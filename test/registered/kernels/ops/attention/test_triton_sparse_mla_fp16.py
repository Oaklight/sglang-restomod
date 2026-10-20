"""FP16 sparse MLA parity, including GLM's zero-dimensional RoPE tail."""

import math

import pytest
import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _triton_sparse_mla_fwd_splitk,
    triton_sparse_mla_fwd,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    _get_splitk_bufs,
    triton_sparse_mla_decode_splitk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA FP16 kernel coverage",
)


@pytest.mark.parametrize("tail_dim", [0, 64])
@pytest.mark.parametrize(
    "mode,seq,heads,topk",
    [
        ("prefill", 3, 16, 63),
        ("prefill", 3, 16, 2048),
        ("prefill", 129, 8, 63),
        ("prefill", 129, 8, 2051),
        ("decode-fused", 3, 16, 193),
        ("decode-split", 3, 16, 193),
    ],
)
def test_fp16_sparse_mla_matches_reference(mode, seq, heads, topk, tail_dim):
    # FP32 CPU math independently checks the indexed softmax/value contract.
    generator = torch.Generator().manual_seed(731)
    value_dim = 512
    q = torch.randn(seq, heads, value_dim + tail_dim, generator=generator).half()
    kv = torch.randn(4099, value_dim + tail_dim, generator=generator).half()
    indices = torch.stack(
        [torch.randperm(kv.shape[0], generator=generator)[:topk] for _ in range(seq)]
    ).int()
    indices[:, ::7] = -1
    indices[-1] = -1
    scale = (value_dim + tail_dim) ** -0.5
    expected = torch.zeros(seq, heads, value_dim)
    for row in range(seq - 1):
        selected = kv[indices[row][indices[row] >= 0].long()].float()
        probabilities = (q[row].float() @ selected.T * scale).softmax(-1)
        expected[row] = probabilities @ selected[:, :value_dim]

    q_gpu = q.cuda()
    args = (
        q_gpu[:, :, :value_dim],
        q_gpu[:, :, value_dim:],
        kv.cuda().unsqueeze(1),
        indices.cuda().unsqueeze(1),
        scale,
        value_dim,
    )
    if mode == "prefill":
        actual = triton_sparse_mla_fwd(*args)
    else:
        actual = triton_sparse_mla_decode_splitk(
            *args, kv_splits=1 if mode == "decode-fused" else 3
        )
    assert actual.dtype == torch.float16
    assert actual.shape == (1, seq, heads, value_dim)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0].float().cpu(), expected, rtol=0.01, atol=0.001)
    assert torch.count_nonzero(actual[0, -1]) == 0
    print(
        mode,
        seq,
        heads,
        topk,
        tail_dim,
        "max_abs_error",
        (actual[0].float().cpu() - expected).abs().max().item(),
    )


@pytest.mark.parametrize("kv_splits", [None, 1, 3])
@pytest.mark.parametrize(
    "mode,seq,heads",
    [("decode", 1, 8), ("decode", 1, 16), ("decode", 4, 16), ("verify", 4, 16)],
)
def test_glm_decode_graph_replays_sparse_and_empty_indices(kv_splits, mode, seq, heads):
    """Tile changes must preserve padded/strided queries and live graph indices."""
    generator = torch.Generator().manual_seed(735)
    dim, topk = 512, 2051
    # A larger packed row leaves noncontiguous head strides, as model views do.
    packed = torch.randn(seq, heads, dim + 64, generator=generator).half()
    kv = torch.randn(4099, dim, generator=generator).half()
    q = packed.cuda()
    cache = kv.cuda().unsqueeze(1)
    indices = torch.full((seq, 1, topk), -1, dtype=torch.int32, device="cuda")
    scale = dim**-0.5
    workspace = []

    def run():
        args = (q[:, :, :dim], q[:, :, dim:dim], cache, indices, scale, dim)
        if mode == "verify":
            if kv_splits is None:
                return triton_sparse_mla_fwd(*args)
            return _triton_sparse_mla_fwd_splitk(*args, kv_splits=kv_splits)
        return triton_sparse_mla_decode_splitk(
            *args, kv_splits=kv_splits, workspace=workspace
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        for count in (0, 1, 65, 128, 2048, 2051, 0):
            ids = torch.full((seq, topk), -1, dtype=torch.int32)
            for row in range(seq):
                valid_count = max(0, count - row * 16)
                positions = torch.randperm(topk, generator=generator)[:valid_count]
                ids[row, positions] = torch.randperm(kv.shape[0], generator=generator)[
                    :valid_count
                ].int()
            packed.normal_(generator=generator)
            q.copy_(packed)
            indices.copy_(ids.unsqueeze(1))
            graph.replay()
            stream.synchronize()
            expected = torch.zeros(seq, heads, dim)
            for row in range(seq):
                selected = kv[ids[row][ids[row] >= 0].long()].float()
                if selected.numel():
                    probabilities = (
                        packed[row, :, :dim].float() @ selected.T * scale
                    ).softmax(-1)
                    expected[row] = probabilities @ selected
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual[0].float().cpu(), expected, rtol=0.01, atol=0.001
            )
            if count == 0:
                assert torch.count_nonzero(actual) == 0


def test_splitk_workspace_preserves_dtype_and_old_allocations():
    workspace = []
    args = (3, 3, 16, 512, torch.device("cuda:0"), workspace)
    _, bf16 = _get_splitk_bufs(*args)
    _, fp16 = _get_splitk_bufs(*args, dtype=torch.float16)
    assert bf16.dtype == torch.bfloat16
    assert fp16.dtype == torch.float16
    assert bf16.data_ptr() != fp16.data_ptr()
    assert len(workspace) == 2
    _, reused = _get_splitk_bufs(*args, dtype=torch.float16)
    assert reused.data_ptr() == fp16.data_ptr()
    assert len(workspace) == 2


def test_empty_tile_prefill_replays_without_changing_valid_tile_math():
    """Skipping padding must retain holes, pool4 tails and graph input refresh."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta prefill padding dispatch")
    torch.manual_seed(738)
    packed = torch.randn(129, 8, 576, device="cuda", dtype=torch.float16)
    q, rope = packed[:, :, :512], packed[:, :, 512:512]
    kv = torch.randn(4099, 1, 512, device="cuda", dtype=torch.float16)
    indices = torch.full((129, 1, 2051), -1, device="cuda", dtype=torch.int32)
    args = (q, rope, kv, indices, 256**-0.5, 512)
    for _ in range(3):
        triton_sparse_mla_fwd(*args, skip_empty_tiles=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = triton_sparse_mla_fwd(*args, skip_empty_tiles=True)
    for count in (0, 1, 33, 128, 2048, 2051, 0):
        indices.fill_(-1)
        if count:
            indices[:, :, :count] = torch.arange(count, device="cuda")
            if count < 2048:
                indices[:, :, 16:32] = -1  # Empty interior tile/partial masks.
                indices[:, :, 2048:] = torch.tensor([7, -1, 31], device="cuda")
        packed.normal_()
        kv.normal_()
        reference = triton_sparse_mla_fwd(*args)
        graph.replay()
        # The independent softmax fixtures above establish the arithmetic;
        # this pins its reduction/rounding order across the padding branch.
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("pages,seq,topk", [(257, 33, 67), (262273, 129, 2051)])
def test_sm70_tensorcore_preserves_support_ragged_tail_and_graph_replay(
    heads, pages, seq, topk
):
    """Gather/softmax MMA tiles must preserve independent sparse query support."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse prefill")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(740)
    # Head-strided input, odd query count and non-power-of-two pool4 width.
    packed = torch.randn(seq, heads, 576, device="cuda", dtype=torch.float16)
    q = packed[:, :, :512]
    kv = torch.randn(pages, 1, 512, device="cuda", dtype=torch.float16)
    indices = torch.full((seq, 1, topk), -1, device="cuda", dtype=torch.int32)
    scale = 256**-0.5
    for _ in range(3):
        sparse_attention_v100(q, kv, indices, scale).unsqueeze(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sparse_attention_v100(q, kv, indices, scale).unsqueeze(0)
    for update in range(3):
        packed.normal_()
        kv.normal_()
        indices.fill_(-1)
        for row in range(seq):
            if (row + update) % 5 == 0:
                continue
            valid = torch.randperm(pages, device="cuda")[: min(topk, row * 3 + 1)]
            columns = torch.randperm(topk, device="cuda")[: valid.numel()]
            indices[row, 0, columns] = valid.int()
        # A populated first query with an entirely empty second query guards
        # the all-masked initial tile: a finite sentinel would leak KV values.
        indices[0].fill_(-1)
        indices[0, 0, :3] = torch.tensor([5, 17, 31], device="cuda")
        indices[1].fill_(-1)
        if topk == 2051:
            indices.fill_(-1)
            for row, count in enumerate((1, 65, 128, 2048, 2051)):
                indices[row + 2, 0, :count] = torch.arange(
                    pages - count, pages, device="cuda", dtype=torch.int32
                )
            indices[4, 0, 16:32] = -1
        before = indices.clone()
        graph.replay()
        eager = sparse_attention_v100(q, kv, indices, scale).unsqueeze(0)
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row in range(seq):
            ids = indices[row, 0]
            selected = kv[ids[ids >= 0].long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        for result in (actual, eager):
            assert torch.isfinite(result).all()
            torch.testing.assert_close(
                result[0].double(), expected, rtol=0.01, atol=0.001
            )
            assert torch.count_nonzero(result[0, 1]) == 0
        assert torch.equal(indices, before)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("topk", [67, 2051])
def test_sm70_tensorcore_graph_refreshes_explicit_length_bounds(heads, topk):
    """Shared-tile reuse must retain live bounds, masks and duplicate support."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse prefill")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(742)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.float16)
    kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.float16)
    indices = (
        torch.arange(topk, dtype=torch.int32, device="cuda").remainder(257).repeat(3, 1)
    )
    indices[:, ::11] = -1
    indices[:, ::17] = 257  # Out-of-pool padding.
    indices[:, 33:41] = 256  # Duplicate keys keep their original multiplicity.
    indices[:, 64:128] = -1  # Empty tile before later populated tiles.
    indices[:, 0] = 7
    lengths = torch.tensor([0, 1, topk], dtype=torch.int32, device="cuda")
    scale = 512**-0.5

    def run():
        return sparse_attention_v100(q, kv, indices, scale, lengths).unsqueeze(0)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for bounds in ((0, 1, topk), (topk + 33, -1, 65), (1, topk, 0)):
        lengths.copy_(torch.tensor(bounds, dtype=torch.int32, device="cuda"))
        q.normal_()
        kv.normal_()
        indices.copy_(indices.roll(1, dims=1))
        graph.replay()
        eager = run()
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row, count in enumerate(bounds):
            ids = indices[row, : min(topk, max(0, count))]
            selected = kv[ids[(ids >= 0) & (ids < kv.shape[0])].long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual[0].double(), expected, rtol=0.01, atol=0.001)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))


@pytest.mark.parametrize("topk", [67, 2051, 4096])
def test_sparse_length_scan_handles_ragged_width_and_empty_rows(topk):
    """The first backscan block may cross column zero, never its row boundary."""
    from sglang.kernels.ops.kvcache.cache_ops import q8kv8_topk_length_from_indices

    indices = torch.full((4, topk), -1, dtype=torch.int32, device="cuda")
    indices[1, 0] = 7
    indices[2, topk - 1] = 11
    indices[3, min(65, topk - 1)] = 19
    expected = torch.tensor(
        [1, 1, topk, min(65, topk - 1) + 1], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(q8kv8_topk_length_from_indices(indices), expected)


@pytest.mark.parametrize(
    "rows,heads,topk,splits,use_lengths,owned_workspace",
    [
        (1, 8, 2051, None, False, True),
        (1, 16, 2051, None, False, True),
        (4, 16, 2051, None, False, False),
        (3, 16, 67, 3, True, True),
        (2, 8, 67, 1, True, False),
        (4, 16, 2051, 3, True, True),
    ],
)
def test_sm70_tensorcore_decode_reference_graph_and_workspace(
    rows, heads, topk, splits, use_lengths, owned_workspace
):
    """Split partials must retain sparse support, stride and live graph inputs."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse decode")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(743)
    pages = 262273 if topk == 2051 else 257
    packed = torch.randn(rows, heads, 576, device="cuda", dtype=torch.float16)
    if rows == 4 and not use_lengths:
        # BMM-derived queries can be head-major after a transpose, not merely
        # a head-strided view of a token-major tensor.
        packed = packed.transpose(0, 1).contiguous().transpose(0, 1)
    q = packed[:, :, :512]
    kv = torch.randn(pages, 1, 512, device="cuda", dtype=torch.float16)
    ids = torch.full((rows, 1, topk), -1, device="cuda", dtype=torch.int32)
    lengths = (
        torch.zeros(rows, device="cuda", dtype=torch.int32) if use_lengths else None
    )
    workspace = {} if owned_workspace else None
    scale = 256**-0.5

    def run():
        return sparse_attention_v100(
            q, kv, ids, scale, lengths, workspace, splits
        ).unsqueeze(0)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    old_buffers = [
        (lse.data_ptr(), acc.data_ptr()) for lse, acc in (workspace or {}).values()
    ]
    for update, count in enumerate((0, 1, 65, 2048, 2051, 0)):
        packed.normal_()
        kv.normal_()
        ids.fill_(-1)
        for row in range(rows):
            valid = min(topk, max(0, count - row * 16))
            if valid:
                selected = torch.randint(
                    pages, (valid,), device="cuda", dtype=torch.int32
                )
                columns = torch.randperm(topk, device="cuda")[:valid]
                ids[row, 0, columns] = selected
                ids[row, 0, -1] = pages - 1
                # Invalid upper slot plus duplicate selection and an interior hole.
                ids[row, 0, :3] = torch.tensor(
                    [pages, pages - 1, pages - 1], device="cuda"
                )
                ids[row, 0, 16:32] = -1
        if use_lengths:
            lengths.copy_(
                torch.tensor(
                    [
                        -1 if count == 0 else min(topk + 7, count - row * 16)
                        for row in range(rows)
                    ],
                    device="cuda",
                    dtype=torch.int32,
                )
            )
        before = ids.clone()
        graph.replay()
        eager = run()
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row in range(rows):
            limit = topk if lengths is None else min(topk, max(0, int(lengths[row])))
            selected_ids = ids[row, 0, :limit]
            selected_ids = selected_ids[(selected_ids >= 0) & (selected_ids < pages)]
            selected = kv[selected_ids.long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        for output in (actual, eager):
            assert output.dtype == torch.float16 and output.shape == (
                1,
                rows,
                heads,
                512,
            )
            assert torch.isfinite(output).all()
            torch.testing.assert_close(
                output[0].double(), expected, rtol=0.01, atol=0.001
            )
            if count == 0:
                assert torch.count_nonzero(output) == 0
        assert torch.equal(ids, before)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))
        assert old_buffers == [
            (lse.data_ptr(), acc.data_ptr()) for lse, acc in (workspace or {}).values()
        ]


@pytest.mark.parametrize("misaligned", ["q", "kv"])
def test_sm70_tensorcore_decode_rejects_unaligned_storage_views(misaligned):
    """A contiguous view can still violate the vector-load pointer alignment."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse decode")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.float16)
    kv = torch.zeros(65, 512, device="cuda", dtype=torch.float16)
    ids = torch.zeros(1, 1, 67, device="cuda", dtype=torch.int32)
    if misaligned == "q":
        q = torch.empty(q.numel() + 1, device="cuda", dtype=q.dtype)[1:].view_as(q)
        q.zero_()
    else:
        kv = torch.empty(kv.numel() + 1, device="cuda", dtype=kv.dtype)[1:].view_as(kv)
    if misaligned == "kv":
        with pytest.raises(RuntimeError, match="alignment"):
            sparse_attention_v100(q, kv, ids, 0.0625)
    else:
        assert torch.count_nonzero(sparse_attention_v100(q, kv, ids, 0.0625)) == 0


@pytest.mark.parametrize("heads", [8, 16])
def test_sm70_sparse_prefill_realigns_contiguous_offset_views(heads):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 sparse FA")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(743)
    q = torch.randn(3 * heads * 512 + 1, device="cuda", dtype=torch.float16)[1:]
    q = q.view(3, heads, 512)
    kv = torch.randn(67 * 512 + 1, device="cuda", dtype=torch.float16)[1:]
    kv = kv.view(67, 512)
    assert q.is_contiguous() and kv.is_contiguous()
    assert q.data_ptr() % 16 and kv.data_ptr() % 16
    ids = torch.tensor(
        [[1, 1, -1], [-1, -1, -1], [5, 31, 66]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([3, 3, 3], device="cuda", dtype=torch.int32)
    before = ids.clone()
    # The shared API realigns Q; the cache must already meet its storage contract.
    with pytest.raises(RuntimeError, match="alignment"):
        sparse_attention_v100(q, kv, ids, 0.0625, lengths)
    actual = sparse_attention_v100(q, kv.clone(), ids, 0.0625, lengths)
    expected = torch.zeros_like(q, dtype=torch.float64)
    for row in (0, 2):
        selected = kv[ids[row][ids[row] >= 0].long()].double()
        expected[row] = (q[row].double() @ selected.T * 0.0625).softmax(-1) @ selected
    torch.testing.assert_close(actual.double(), expected, rtol=0.01, atol=0.001)
    assert torch.equal(ids, before)


@pytest.mark.parametrize("invalid", ["dtype", "lengths", "heads", "alignment"])
def test_sm70_sparse_fa_binding_rejects_unsafe_operands(invalid):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 sparse FA binding")
    from sglang_fa_sm70 import ensure_loaded

    ensure_loaded()
    q = torch.zeros(2, 8, 512, device="cuda", dtype=torch.float16)
    kv = torch.zeros(67, 512, device="cuda", dtype=torch.float16)
    ids = torch.zeros(2, 3, device="cuda", dtype=torch.int32)
    lengths = torch.full((2,), 3, device="cuda", dtype=torch.int32)
    if invalid == "dtype":
        q = q.float()
    elif invalid == "lengths":
        lengths = lengths[:1]
    elif invalid == "heads":
        q = q[:, :4].contiguous()
    else:
        q = torch.empty(q.numel() + 1, device="cuda", dtype=q.dtype)[1:].view_as(q)
    with pytest.raises(RuntimeError, match="SM70 sparse MLA"):
        torch.ops.sglang_fa_sm70.prefill(q, kv, ids, lengths, 0.0625)


@pytest.mark.parametrize("rows", [1, 4, 5, 129])
def test_sm70_dcp_normalization_and_graph_replay(rows):
    """Two latent stripes must reconstruct sparse attention, including empty owners.

    Pooled-index selection stays global and replicated; normalizers must be
    base-2, with -inf for empty rows, before the upstream FP32 DCP correction.
    """
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta FA partial normalization")
    from sglang.kernels.ops.attention.dcp_kernels import correct_attn_out
    from sglang.kernels.ops.attention.dsa.transform_index import (
        transform_index_page_table_dcp,
    )
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(745)
    heads, count, width = 16, 4099, 2051
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.float16)
    kv = torch.randn(count, 512, device="cuda", dtype=torch.float16)
    stripes = [kv[rank::2].contiguous() for rank in range(2)]
    ids = torch.zeros(rows, width, device="cuda", dtype=torch.int32)
    workspaces = [{}, {}]
    scale = 256**-0.5

    def run():
        outputs, normalizers = [], []
        for rank in range(2):
            local, lengths = transform_index_page_table_dcp(ids, 2, rank)
            out, lse = sparse_attention_v100(
                q,
                stripes[rank],
                local,
                scale,
                lengths=lengths,
                workspace=workspaces[rank],
                return_lse=True,
            )
            outputs.append(out)
            normalizers.append(lse)
        lses = torch.stack(normalizers)
        corrected = [
            correct_attn_out(
                outputs[rank],
                lses,
                rank,
                None,
                new_output=torch.empty(
                    heads, rows, 512, device="cuda", dtype=torch.float32
                ),
                is_lse_base_on_e=False,
            )[0]
            for rank in range(2)
        ]
        return (corrected[0] + corrected[1]).transpose(0, 1).half(), lses

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lses = run()
    for replay in range(3):
        q.normal_()
        kv.normal_()
        ids.random_(-1, count + 2)
        if rows > 1 or replay == 1:
            ids[0].fill_(-1)  # Every owner is empty.
        if rows > 1:
            ids[1] = 2 * (torch.arange(width, device="cuda") % (count // 2))
            ids[-1, -3:] = torch.tensor([count - 1, -1, 7], device="cuda")
        for rank in range(2):
            stripes[rank].copy_(kv[rank::2])
        graph.replay()
        cq, ck, ci = q.double().cpu(), kv.double().cpu(), ids.cpu()
        expected = torch.zeros(rows, heads, 512, dtype=torch.float64)
        expected_lse = torch.full((2, rows, heads), -torch.inf, dtype=torch.float64)
        for row in range(rows):
            selected = ci[row][(ci[row] >= 0) & (ci[row] < count)].long()
            if selected.numel():
                scores = cq[row] @ ck[selected].T * scale
                expected[row] = scores.softmax(-1) @ ck[selected]
            for rank in range(2):
                local = selected[selected % 2 == rank]
                if local.numel():
                    expected_lse[rank, row] = torch.logsumexp(
                        cq[row] @ ck[local].T * scale, -1
                    ) / math.log(2)
        torch.testing.assert_close(
            actual.double().cpu(), expected, rtol=0.001, atol=0.0002
        )
        torch.testing.assert_close(
            lses.double().cpu(), expected_lse, rtol=0.0001, atol=0.0001
        )
        if rows > 1 or replay == 1:
            assert torch.count_nonzero(actual[0]) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
