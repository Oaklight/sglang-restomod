"""Explicit GLM attention profile; CUDA implementations live in the FA fork."""

import torch

from sglang.kernels.ops.attention.v100 import (
    dense_attention_v100,
    recurrent_attention_v100,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


def resolve_profile(original, view):
    if view.attention_backend != "v100_plus":
        return original(view)
    from sglang.srt.arg_groups.overrides import model_config_of

    model = model_config_of(view).hf_config
    if model.architectures[0] != "Glm5NextForConditionalGeneration":
        raise ValueError(
            "v100_plus attention currently requires GLM-5.3-Flash with vision"
        )
    if view.dtype not in ("float16", "half") or view.kv_cache_dtype != "auto":
        raise ValueError("v100_plus attention requires FP16 model/KV/state")
    if view.enable_prefill_cp or view.dcp_size != 1 or view.dp_size != 1:
        raise ValueError("v100_plus attention currently supports TP/PP with DP/CP1")
    if view.max_running_requests != 1:
        raise ValueError(
            "v100_plus attention currently requires max-running-requests=1"
        )
    if (
        view.mamba_ssm_dtype != "float16"
        or view.mamba_radix_cache_strategy != "no_buffer"
    ):
        raise ValueError(
            "v100_plus attention requires FP16 recurrent state and no_buffer radix strategy"
        )
    fields = {
        "dsa_prefill_backend": "v100_plus",
        "dsa_decode_backend": "v100_plus",
        "linear_attn_backend": "v100_plus",
        "linear_attn_decode_backend": "v100_plus",
        "linear_attn_prefill_backend": "v100_plus",
        "linear_attn_verify_backend": "v100_plus",
        "mm_attention_backend": "v100_plus",
    }
    for name, target in fields.items():
        value = getattr(view, name)
        default = "triton" if name == "linear_attn_backend" else None
        if value not in (default, target):
            raise ValueError(f"v100_plus attention conflicts with {name}={value!r}")
    return fields


def kda_dispatcher_init(
    original, self, decode_backend, prefill_backend, verify_backend
):
    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    if mamba.linear_attn_backend != "v100_plus":
        return original(self, decode_backend, prefill_backend, verify_backend)
    if not all(
        x.is_custom() for x in (decode_backend, prefill_backend, verify_backend)
    ):
        raise ValueError("v100_plus KDA requires the unified profile in every mode")
    self.prefill_backend = prefill_backend
    self.verify_backend = verify_backend
    self.decode_kernel = self.extend_kernel = self.verify_kernel = V100KDAKernel()


class V100KDAKernel(LinearAttnKernelBase):
    supports_packed_decode = False
    supports_strided_target_verify_qkv = True
    supports_safe_gate_decode = True
    supports_safe_gate_verify = True

    def __init__(self):
        self._prefill_workspaces = {}

    def _prefill_workspace(self, q):
        if q.shape[0] < 128:
            return None
        stream = int(torch.cuda.current_stream(q.device).cuda_stream)
        capacity = 3 * q.numel()
        buf = self._prefill_workspaces.get(stream)
        if buf is None or buf.numel() < capacity:
            buf = torch.empty(capacity, device=q.device, dtype=torch.float32)
            self._prefill_workspaces[stream] = buf
        return buf[:capacity].view(3, *q.shape)

    @staticmethod
    def _views(q, k, v, g, beta):
        heads = q.shape[-2]
        return (
            *(x.reshape(-1, heads, 128) for x in (q, k, v, g)),
            beta.reshape(-1, heads),
        )

    def extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log=None,
        dt_bias=None,
        lower_bound=None,
        beta_is_raw=False,
        return_intermediate_states=False,
        **kwargs,
    ):
        if (
            return_intermediate_states
            or kwargs.get("track_state") is not None
            or kwargs.get("is_spec_decode")
        ):
            raise NotImplementedError(
                "v100_plus KDA prefill requires no_buffer state and ordinary extend"
            )
        q, k, v, g, beta = self._views(q, k, v, g, beta)
        if A_log is not None:
            if lower_bound != -5.0:
                raise ValueError("v100_plus GLM KDA requires safe gate lower_bound=-5")
            raw = g.float() + dt_bias.float().reshape(1, q.shape[1], 128)
            g = lower_bound * torch.sigmoid(
                A_log.float().exp().reshape(1, q.shape[1], 1) * raw
            )
        if beta_is_raw:
            # Preserve the baseline prefill's FP16 beta boundary.
            beta = beta.sigmoid()
        return recurrent_attention_v100(
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            cache_indices.int(),
            query_start_loc,
            prepared=self._prefill_workspace(q),
        ).unsqueeze(0)

    def decode(
        self,
        q,
        k,
        v,
        a,
        b,
        *,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        query_start_loc,
        lower_bound=None,
        **kwargs,
    ):
        if lower_bound != -5.0:
            raise ValueError("v100_plus GLM KDA requires safe gate lower_bound=-5")
        q, k, v, a, b = self._views(q, k, v, a, b)
        return recurrent_attention_v100(
            q,
            k,
            v,
            a,
            b,
            ssm_states,
            cache_indices.int(),
            query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            raw_beta=True,
        ).unsqueeze(0)

    def target_verify(
        self,
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        retrieve_parent_token,
        lower_bound=None,
        **kwargs,
    ):
        if lower_bound != -5.0 or kwargs.get("cache_ring"):
            raise ValueError(
                "v100_plus GLM verification requires safe gates and FP16 snapshots"
            )
        if intermediate_states_buffer.shape[1] != cache_steps:
            raise ValueError(
                "KDA verification snapshot capacity differs from cache_steps"
            )
        q, k, v, a, b = self._views(q, k, v, a, b)
        return recurrent_attention_v100(
            q,
            k,
            v,
            a,
            b,
            ssm_states,
            cache_indices.int(),
            query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            raw_beta=True,
            write_state=False,
            snapshots=intermediate_states_buffer,
            snapshot_slots=intermediate_state_indices.int(),
            parents=None
            if retrieve_parent_token is None
            else retrieve_parent_token.int(),
        ).unsqueeze(0)


class V100VisionAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens,
        bsz,
        seq_len,
        softmax_scale=None,
        forward_metadata=None,
        **kwargs,
    ):
        from sglang.srt.layers.attention.vision import (
            resolve_precomputed_max_seqlen,
            resolve_seqlens,
        )

        if forward_metadata is not None:
            offsets = forward_metadata.cu_seqlens
            max_length = forward_metadata.max_seqlen
        else:
            offsets = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            max_length = resolve_precomputed_max_seqlen(
                offsets, kwargs.get("max_seqlen")
            )
        return dense_attention_v100(
            q,
            k,
            v,
            offsets,
            max_length,
            q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
        )


def register_profile():
    from sglang.srt.arg_groups.choices import (
        add_attention_backend_choices,
        add_draft_attention_backend_choices,
        add_linear_attn_kernel_backend_choices,
    )
    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )
    from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL
    from sglang.srt.models.deepseek_common.attention_backend_handler import (
        AttentionBackendRegistry,
        handle_attention_dsa,
    )
    from sglang.srt.models.deepseek_common.utils import (
        FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    )

    add_attention_backend_choices(["v100_plus"])
    add_draft_attention_backend_choices(["v100_plus"])
    add_linear_attn_kernel_backend_choices(["v100_plus"])
    QKV_BACKEND_IMPL["v100_plus"] = V100VisionAttention
    AttentionBackendRegistry.register("v100_plus", handle_attention_dsa)
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.append("v100_plus")

    @register_attention_backend("v100_plus")
    def create_backend(runner):
        from .glm_dsa import SM70SparseAttnBackend

        return SM70SparseAttnBackend(runner)
