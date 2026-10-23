"""Canonical FA adapters preserve padded rows and sequence-sized snapshots."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

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


class TestWindowAttention(CustomTestCase):
    def test_real_triton_entry_preserves_encoder_sliding_window(self):
        """ENCODER_ONLY removes causality, but retains a query-dependent left mask."""
        from types import MethodType

        from sglang_v100_plus.dflash import forward_attention
        from transformers import Qwen3Config

        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.dflash import _get_dflash_layer_attention_params

        torch.manual_seed(531)
        q = torch.randn(8, 4, 128, device="cuda", dtype=torch.float16) * 0.5
        k = torch.randn(8, 1, 128, device="cuda", dtype=torch.float16) * 0.5
        v = torch.randn_like(k)
        kc, vc = [
            torch.randn(2311, 1, 128, device="cuda", dtype=torch.float16)
            for _ in range(2)
        ]
        ids = torch.randperm(2311, device="cuda", dtype=torch.int32)[:2048]
        ptr = torch.tensor([0, 2048], device="cuda", dtype=torch.int32)
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.use_dense_fp8_chunked_prefill = False
        backend.use_mla = backend.enable_deterministic = False
        backend.use_verify_shared_kv = backend.use_verify_splitkv = False
        backend.dcp_size = 1
        backend.page_size = 256
        backend._v100_dflash = True
        backend._v100_dflash_workspaces = {}
        backend.dllm_attention = None
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: kc,
            get_value_buffer=lambda _: vc,
        )
        backend.forward_metadata = SimpleNamespace(
            window_kv_indptr=ptr,
            window_kv_indices=ids,
            window_kv_offsets=torch.zeros(1, device="cuda", dtype=torch.int32),
            qo_indptr=ptr.new_tensor([0, 8]),
            custom_mask=None,
            mask_indptr=None,
            max_extend_len=8,
        )
        backend._forward_extend_kernel = MethodType(
            lambda self, layer, *args, **kwargs: forward_attention(
                None, self, layer, *args, **kwargs
            ),
            backend,
        )
        config = Qwen3Config(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=2048,
            use_sliding_window=True,
            is_causal=False,
        )
        left_window, attn_type = _get_dflash_layer_attention_params(config, 0)
        self.assertEqual(left_window, 2047)
        layer = SimpleNamespace(
            qk_head_dim=128,
            v_head_dim=128,
            tp_q_head_num=4,
            layer_id=0,
            logit_capping_method="tanh",
            logit_cap=0,
            is_cross_attention=False,
            attn_type=attn_type,
            sliding_window_size=left_window,
            k_scale=None,
            v_scale=None,
            xai_temperature_len=-1,
            scaling=128**-0.5,
        )
        batch = SimpleNamespace(
            _attn_output=None,
            mha_one_shot=False,
            forward_mode=ForwardMode.TARGET_VERIFY,
            extend_seq_lens_cpu=torch.tensor([8]),
        )
        actual = backend.forward_extend(
            q.reshape(8, 512), k, v, layer, batch, save_kv_cache=False
        )
        keys = torch.cat((kc[ids.long()], k)).cpu().double().view(-1, 128)
        values = torch.cat((vc[ids.long()], v)).cpu().double().view(-1, 128)
        scores = q.cpu().double().reshape(32, 128) @ keys.T * 128**-0.5
        lower = 2048 + torch.arange(8).repeat_interleave(4) - left_window
        scores.masked_fill_(
            torch.arange(len(keys))[None, :] < lower[:, None], -torch.inf
        )
        expected = scores.softmax(-1) @ values
        torch.testing.assert_close(
            actual.cpu().double().view(32, 128), expected, rtol=0.001, atol=0.0002
        )

    def test_changed_window_graph_matches_fp64_and_triton(self):
        """Native FA must preserve per-query windows, page IDs and FP16 KV.

        Poisoned tail/split scratch catches padding reads; changed graph counts
        and permuted/duplicate IDs catch frozen metadata or accidental causality.
        Q/K/V and cache views deliberately have different physical row strides.
        """
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
        from sglang.kernels.ops.attention.v100 import window_attention_v100

        torch.manual_seed(531)
        for id_dtype in (torch.int32, torch.int64):
            bank = torch.randn(8, 6, 128, device="cuda", dtype=torch.float16)
            q, k, v = bank[:, :4], bank[:, 4:5], bank[:, 5:]
            cache = torch.randn(2311, 1, 256, device="cuda", dtype=torch.float16)
            kc, vc = cache[:, :, :128], cache[:, :, 128:]
            ids = torch.arange(2065, device="cuda", dtype=id_dtype)
            ptr = torch.tensor([0, 2048], device="cuda", dtype=torch.int32)
            qo = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
            guard = torch.full((32 * 128 + 32,), 37, device="cuda", dtype=torch.float16)
            out = guard[16:-16].view(8, 4, 128)
            workspace = {}
            scale = 128**-0.5

            def run():
                return window_attention_v100(
                    q,
                    k,
                    v,
                    kc,
                    vc,
                    ids,
                    ptr,
                    scale,
                    left_window=2048,
                    workspace=workspace,
                    out=out,
                )

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for i, prefix in enumerate(
                (0, 1, 7, 16, 31, 32, 63, 64, 65, 2039, 2040, 2048)
            ):
                with self.subTest(id_dtype=id_dtype, prefix=prefix):
                    # Calibrated against the original FP16-probability path.
                    # Large random Q/K also exceed this FP64 bound in Triton;
                    # amplify V under uniform scores to test range separately.
                    bank.normal_(std=0.5)
                    if i == 0:
                        v.mul_(3)
                        q.zero_()
                    cache.normal_()
                    ids.copy_((torch.arange(2065, device="cuda") * 97 + i * 19) % 2311)
                    ids[::11] = 17
                    start = 0 if i % 2 else 17
                    ptr.copy_(ptr.new_tensor([start, start + prefix]))
                    # No poison may leak from an inactive key tile or split.
                    for scratch in workspace["window"]:
                        if scratch.is_floating_point():
                            scratch.fill_(float("nan"))
                    graph.replay()
                    saved = out.clone()
                    selected = ids[start : start + prefix].long()
                    keys = torch.cat((kc[selected], k)).cpu().double().view(-1, 128)
                    values = torch.cat((vc[selected], v)).cpu().double().view(-1, 128)
                    scores = (q.cpu().double().reshape(32, 128) @ keys.T) * scale
                    query_positions = prefix + torch.arange(8).repeat_interleave(4)
                    visible = (
                        torch.arange(len(keys))[None, :]
                        >= query_positions[:, None] - 2048
                    )
                    expected = (
                        scores.masked_fill(~visible, -torch.inf).softmax(-1) @ values
                    )
                    torch.testing.assert_close(
                        saved.cpu().double().reshape(32, 128),
                        expected,
                        rtol=0.001,
                        atol=0.0002,
                    )
                    reference = torch.empty_like(out)
                    extend_attention_fwd(
                        q,
                        k,
                        v,
                        reference,
                        kc,
                        vc,
                        qo,
                        ptr,
                        ids,
                        None,
                        False,
                        None,
                        8,
                        1.0,
                        1.0,
                        scale,
                        sliding_window_size=2048,
                    )
                    torch.testing.assert_close(
                        saved, reference, rtol=0.001, atol=0.0002
                    )
                    torch.testing.assert_close(saved, run(), rtol=0, atol=0)
                    self.assertTrue(bool(torch.all(guard[:16] == 37)))
                    self.assertTrue(bool(torch.all(guard[-16:] == 37)))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
