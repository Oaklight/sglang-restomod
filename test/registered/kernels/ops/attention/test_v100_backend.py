"""Canonical FA adapters preserve padded rows and sequence-sized snapshots."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="canonical SM70 FA adapter coverage",
)


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("live,total", [(7, 8), (34, 36)])
def test_sparse_adapter_restores_prefill_padding(heads, live, total):
    """A unit batch axis hid real rows from core's zero-tail restoration.

    Padded prefill then returned fewer rows than the mHC residual contained.
    Real rows must be byte-identical to an unpadded canonical attention call.
    """
    from sglang_v100_plus.glm_dsa import SM70SparseAttnBackend

    from sglang.kernels.ops.attention.v100 import sparse_attention_v100
    from sglang.srt.layers.radix_attention import _attention_on_real_rows
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    torch.manual_seed(531)
    q = torch.randn(total, heads, 512, device="cuda", dtype=torch.float16) * 0.1
    kv = torch.randn(128, 512, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.randint(0, 128, (total, 128), device="cuda", dtype=torch.int32)
    rope = torch.empty(total, heads, 0, device="cuda", dtype=torch.float16)
    layer = SimpleNamespace(
        qk_head_dim=512, v_head_dim=512, tp_q_head_num=heads, is_cross_attention=False
    )
    locations = torch.arange(total, device="cuda")
    positions = locations.clone()
    batch = SimpleNamespace(
        out_cache_loc=locations,
        positions=positions,
        _attn_output=None,
        mha_return_lse=False,
    )

    class Backend:
        def forward(self, q, k, v, layer, batch, save, **kwargs):
            return SM70SparseAttnBackend._forward_v100_sparse(
                self, q, kwargs["q_rope"], kv, kwargs["topk_indices"], 512**-0.5, 512
            )

    backend = Backend()

    def run():
        with forward_context(ForwardContext(attn_backend=backend)):
            return _attention_on_real_rows(
                layer, live, q, None, None, batch, False, q_rope=rope, topk_indices=ids
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for replay in range(2):
        if replay:
            q.mul_(0.625)
            kv.mul_(0.75)
            ids.add_(17).remainder_(128)
        graph.replay()
        expected = sparse_attention_v100(q[:live], kv, ids[:live], 512**-0.5)
        torch.testing.assert_close(out[:live], expected, rtol=0, atol=0)
        assert out.shape == (total, heads, 512)
        assert torch.count_nonzero(out[live:]) == 0
        assert batch.out_cache_loc is locations and batch.positions is positions
        assert batch._attn_output is None


@pytest.mark.parametrize("heads", [8, 16])
def test_kda_adapter_ignores_unused_snapshot_index_capacity(heads):
    """An eager-batch-capacity table must not reject a one-sequence graph."""
    from sglang_v100_plus.attention import V100KDAKernel

    from sglang.kernels.ops.attention.v100 import recurrent_attention_v100

    torch.manual_seed(531)
    shape = (4, heads, 128)
    q, k, v, gate = [
        torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(4)
    ]
    beta = torch.randn(shape[:2], device="cuda", dtype=torch.float16)
    rate = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads * 128, device="cuda")
    initial = (
        torch.randn(2, heads, 128, 128, device="cuda", dtype=torch.float16) * 0.025
    )
    state = initial.clone()
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    snapshots = torch.full(
        (2, 4, heads, 128, 128), 123.0, device="cuda", dtype=torch.float16
    )
    table = torch.tensor([1, 123456, 123456, 123456], device="cuda", dtype=torch.int32)
    kernel = V100KDAKernel()

    def run():
        return kernel.target_verify(
            rate,
            bias,
            q,
            k,
            v,
            gate,
            beta,
            ssm_states=state,
            cache_indices=slots,
            query_start_loc=offsets,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=table,
            cache_steps=4,
            retrieve_parent_token=None,
            lower_bound=-5.0,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for replay in range(2):
        if replay:
            q.mul_(0.5)
            k.add_(0.1)
            gate.mul_(0.75)
            beta.mul_(0.8)
            table[0] = 0
            slots[0] = 1
        snapshots.fill_(123.0)
        graph.replay()
        actual_out, actual_snapshots = out.clone(), snapshots.clone()
        snapshots.fill_(123.0)
        expected = recurrent_attention_v100(
            q,
            k,
            v,
            gate,
            beta,
            state,
            slots,
            offsets,
            A_log=rate,
            dt_bias=bias,
            snapshots=snapshots,
            snapshot_slots=table[:1],
            write_state=False,
            raw_beta=True,
        )
        torch.testing.assert_close(actual_out.squeeze(0), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_snapshots, snapshots, rtol=0, atol=0)
        torch.testing.assert_close(state, initial, rtol=0, atol=0)
        unused_slot = 1 - int(table[0])
        assert torch.all(snapshots[unused_slot] == 123.0)


@pytest.mark.parametrize("heads", [8, 16])
def test_kda_prefill_prefix_checkpoint_preserves_live_recurrence(heads):
    """A prefix checkpoint must observe FP32 carry without rounding live state.

    Reusing verification snapshots here would round at every prefill token.
    Changed boundaries and inputs on graph replay must overwrite the checkpoint,
    while untracked rows and unrelated pool slots remain untouched.
    """
    from sglang_v100_plus.attention import V100KDAKernel

    from sglang.kernels.ops.attention.v100 import recurrent_attention_v100

    torch.manual_seed(531)
    lengths = [257, 17]
    total = sum(lengths)
    q, k, v = [
        torch.randn(total, heads, 128, device="cuda", dtype=torch.float16) * 0.1
        for _ in range(3)
    ]
    gate = torch.full_like(q, -0.1, dtype=torch.float32)
    beta = torch.full((total, heads), 0.3, device="cuda", dtype=torch.float16)
    initial = (
        torch.randn(4, heads, 128, 128, device="cuda", dtype=torch.float16) * 0.025
    )
    slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    state = initial.clone()
    track = torch.full((2, heads, 128, 128), 123.0, device="cuda", dtype=torch.float32)
    kernel = V100KDAKernel()

    def run():
        return kernel.extend(
            q,
            k,
            v,
            gate,
            beta,
            ssm_states=state,
            cache_indices=slots,
            query_start_loc=offsets,
            return_intermediate_states=True,
            track_state=track,
            track_chunk_idx=chunks,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, history = run()
    assert history is None
    for replay in range(2):
        if replay:
            q.mul_(0.75)
            k.mul_(0.625)
            v.add_(0.015)
            chunks[0] = 3
        state.copy_(initial)
        track.fill_(123.0)
        graph.replay()
        actual_out, actual_state, actual_track = (
            out.clone(),
            state.clone(),
            track.clone(),
        )
        expected_state = initial.clone()
        expected_out = kernel.extend(
            q,
            k,
            v,
            gate,
            beta,
            ssm_states=expected_state,
            cache_indices=slots,
            query_start_loc=offsets,
        )
        torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        steps = int(chunks[0]) * 64
        prefix_state = initial.clone()
        recurrent_attention_v100(
            q[:steps],
            k[:steps],
            v[:steps],
            gate[:steps],
            beta[:steps],
            prefix_state,
            slots[:1],
            offsets.new_tensor([0, steps]),
        )
        torch.testing.assert_close(
            actual_track[0].half(), prefix_state[2], rtol=0, atol=0
        )
        # Independent FP64 recurrence at the selected boundary.
        carry = initial[2].double()
        for t in range(steps):
            qt, kt = q[t].double(), k[t].double()
            qt *= torch.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
            kt *= torch.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            carry *= gate[t].double().exp().unsqueeze(-2)
            residual = (v[t].double() - (carry * kt.unsqueeze(-2)).sum(-1)) * beta[
                t
            ].double().unsqueeze(-1)
            carry += residual.unsqueeze(-1) * kt.unsqueeze(-2)
        torch.testing.assert_close(
            actual_track[0].double(), carry, rtol=0.001, atol=0.0002
        )
        assert torch.all(actual_track[1] == 123.0)
        torch.testing.assert_close(actual_state[1], initial[1], rtol=0, atol=0)
        torch.testing.assert_close(actual_state[3], initial[3], rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
