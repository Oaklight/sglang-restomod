"""GPU test for the SM70 dense block-FP8 TurboMind path.

The CPU suite pins the shape guards; this covers what only a card can show: the
arithmetic against an FP64 reference and a CUDA-graph replay whose input
changes between replays. Deselected on CPU CI (`-m "not gpu"`); run on a V100
with `scripts/test_gpu.sh`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

GROUP = 128


def _dequantized(weight, scales):
    n, k = weight.shape
    blocks = (
        scales.double()
        .repeat_interleave(GROUP, 0)
        .repeat_interleave(GROUP, 1)[:n, :k]
    )
    return weight.double() * blocks


def _reference(x, weight, scales):
    return x.double() @ _dequantized(weight, scales).T


def _relative(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _layer(n, k, weight, scales):
    layer = torch.nn.Module()
    layer.orig_dtype = torch.float16
    layer.weight_block_size = [GROUP, GROUP]
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
    layer.output_size_per_partition = n
    layer.prefix = "model.layers.0.self_attn.q_proj"
    return layer


def _prepare_kernels():
    from sglang_restomod.quant.sm70_fp8_dense import _method as kernels

    if not kernels._load_sm70_turbomind_fp8_ops():
        pytest.skip("SM70 TurboMind FP8 operators unavailable")
    return kernels


def _weight_and_scales(n, k, seed):
    torch.manual_seed(seed)
    weight = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    scales = 2.0 ** torch.randint(
        -6, -2, (n // GROUP, k // GROUP), device="cuda"
    ).float()
    return weight, scales


@pytest.mark.parametrize("rows,n,k", [(1, 128, 128), (5, 256, 384), (40, 512, 256)])
def test_dense_projection_matches_fp64(rows, n, k):
    kernels = _prepare_kernels()
    weight, scales = _weight_and_scales(n, k, rows + n + k)
    layer = _layer(n, k, weight, scales)
    kernels.prepare_sm70_turbomind_fp8_linear(layer)

    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    out = kernels.apply_sm70_turbomind_fp8_linear(layer, x, None)

    assert _relative(out, _reference(x, weight, scales)) < 5e-2


def test_dense_graph_replay_reads_the_new_input():
    kernels = _prepare_kernels()
    rows, n, k = 4, 256, 256
    weight, scales = _weight_and_scales(n, k, 11)
    layer = _layer(n, k, weight, scales)
    kernels.prepare_sm70_turbomind_fp8_linear(layer)

    static_x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    # Warm the allocator and the kernel before capture.
    for _ in range(3):
        kernels.apply_sm70_turbomind_fp8_linear(layer, static_x, None)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = kernels.apply_sm70_turbomind_fp8_linear(layer, static_x, None)
    first = static_out.clone()

    changed = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    static_x.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    replay = static_out.clone()

    assert _relative(replay, _reference(changed, weight, scales)) < 5e-2
    assert not torch.equal(first, replay), "the replay ignored the new input"
