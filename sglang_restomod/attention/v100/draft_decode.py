"""Native multi-step draft decode on the attention_v100 split-KV kernel.

The engine's EAGLE/NEXTN draft worker runs one attention pass per draft step.
At the pin that pass goes to ``TritonMultiStepDraftBackend``, which
precomputes one ``kv_indptr`` / ``kv_indices`` pair per step because the Triton
kernel consumes them. The native decode kernel consumes a page table and one
visible length per row instead.

**Which side moves.** The container converts the per-step index form into the
page table the native kernel already takes; the kernel is not taught the index
form. The native kernel's speed comes from page-coalesced loads and a
tensor-core GEMM, and both need pages, not a token list. A token-index kernel
would reimplement the Triton gather and lose exactly what makes this path
faster. The conversion is one gather per step, shared by every draft layer.

The page table is derived from the same flat list the Triton kernel reads:
row ``r`` starts at ``kv_indptr[r]``, and its logical page ``p`` starts at the
token ``p * page_size`` later. This holds because the draft decode window is
off (``resolve_draft_decode_window`` returns ``(0, 0)``) and the token pool is
page-granular. Launch conditions outside that contract fall back to Triton.

Selection: ``--speculative-draft-attention-backend tilelang_fa_v100``. Set
``SGLANG_V100_DRAFT_DECODE_NATIVE=0`` to keep Triton instead.
"""

from __future__ import annotations

import logging
import math

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend
from sglang.srt.speculative.spec_utils import resolve_draft_decode_window

from . import draft_decode_plan as plan

logger = logging.getLogger(__name__)

_KV_DTYPES = (torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)
_MODEL_DTYPES = (torch.float16, torch.bfloat16)
# Draft tokens extend the visible list past the request's context length.
_PAGE_MARGIN = 16


class NativeDraftDecodeBackend(AttentionBackend):
    """One draft step: a page table plus the native grouped decode call."""

    needs_cpu_seq_lens = False
    # The native kernel computes active splits on device; the container only
    # needs the attribute to size a Triton-only buffer it will never read.
    max_kv_splits = 1

    def __init__(self, model_runner, topk, max_context_len, page_size, steps):
        super().__init__()
        self.topk = topk
        self.device = model_runner.device
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_index_translator = model_runner.kv_index_translator
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.page_table_width = math.ceil((max_context_len + steps + _PAGE_MARGIN) / page_size)
        self._page_table = None
        self._seq_lens = None
        self._rows_used = 0

    def _ensure_capacity(self, rows: int) -> None:
        if self._page_table is not None and self._page_table.shape[0] >= rows:
            return
        capacity = max(rows, 1)
        self._page_table = torch.zeros(
            (capacity, self.page_table_width), dtype=torch.int32, device=self.device
        )
        self._seq_lens = torch.zeros((capacity,), dtype=torch.int32, device=self.device)

    def refresh(self, kv_indptr: torch.Tensor, kv_indices: torch.Tensor) -> int:
        """Rebuild this step's page table from its flat index list."""
        rows = kv_indptr.numel() - 1
        self._ensure_capacity(rows)
        self._rows_used = plan.page_table_from_indices(
            kv_indptr, kv_indices, self._page_table, self._seq_lens, self.page_size
        )
        return self._rows_used

    def init_forward_metadata(self, forward_batch) -> None:
        spec_info = forward_batch.spec_info
        if spec_info is None or spec_info.kv_indptr is None:
            return
        self.refresh(spec_info.kv_indptr, spec_info.kv_indices)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False) -> None:
        self.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf=None,
        cuda_graph_num_kv_splits_buf=None,
    ) -> None:
        self._ensure_capacity(max_bs * self.topk)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        rows = q.shape[0]
        if rows != self._rows_used:
            raise RuntimeError(
                "native draft decode metadata does not match the query rows: "
                f"{self._rows_used} rows planned, {rows} seen"
            )
        if kwargs.get("sinks") is not None:
            raise NotImplementedError("native draft decode does not carry attention sinks")

        out_dtype = q.dtype
        if out_dtype == torch.bfloat16:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        q3 = q.reshape(rows, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if k_cache.ndim == 3:
            k_cache = k_cache.view(
                -1, self.page_size, k_cache.shape[-2], k_cache.shape[-1]
            )
            v_cache = v_cache.view(
                -1, self.page_size, v_cache.shape[-2], v_cache.shape[-1]
            )

        from .tilelang import grouped_decode_forward

        result = grouped_decode_forward(
            q3,
            k_cache,
            v_cache,
            self._page_table[:rows],
            self._seq_lens[:rows],
            softmax_scale=layer.scaling,
            k_scale=layer.k_scale_float if layer.k_scale is not None else 1.0,
            v_scale=layer.v_scale_float if layer.v_scale is not None else 1.0,
        )
        out = torch.empty(
            (rows, layer.tp_q_head_num * layer.v_head_dim),
            dtype=out_dtype,
            device=q.device,
        )
        out.copy_(result.reshape_as(out))
        return out


class NativeMultiStepDraftBackend(TritonMultiStepDraftBackend):
    """The engine's per-step draft container, with native decode children.

    It reuses the Triton container's index builder (the per-step
    ``kv_indptr`` / ``kv_indices`` production) and replaces only the attention
    children. Everything else -- buffer widths, window resolution, the CUDA
    graph state contract -- stays the engine's.
    """

    def __init__(self, model_runner, topk, speculative_num_steps, stamp):
        super().__init__(model_runner, topk, speculative_num_steps)
        self.attn_backends = [
            NativeDraftDecodeBackend(
                model_runner,
                topk,
                self.max_context_len,
                model_runner.page_size,
                speculative_num_steps,
            )
            for _ in range(self.speculative_num_steps - 1)
        ]
        self.prefill_attention_backend_str = stamp
        self.decode_attention_backend_str = stamp
        for child in self.attn_backends:
            child.prefill_attention_backend_str = stamp
            child.decode_attention_backend_str = stamp

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        # Capture plans each step through the engine's builder. Replay refills
        # the index buffers and then rebuilds each page table; the native
        # kernel has no host-side split count to compute.
        if in_capture:
            return super().init_forward_metadata_out_graph(forward_batch, in_capture=True)

        def call_fn(step, batch):
            self.attn_backends[step].init_forward_metadata(batch)

        self.common_template(forward_batch, None, call_fn)


def _attention_layers(model):
    for module in model.modules():
        if hasattr(module, "tp_q_head_num") and hasattr(module, "layer_id"):
            yield module


def unsupported_reason(model_runner, topk: int, env=None) -> str | None:
    """Return why the native draft path cannot serve this launch, or None."""
    if not plan.native_enabled(env):
        return f"disabled_by_env({plan.ENV_FLAG})"
    if int(getattr(model_runner, "page_size", 1)) != plan.PAGE_SIZE:
        return f"page_size({model_runner.page_size})"
    if getattr(model_runner, "kv_cache_dtype", None) not in _KV_DTYPES:
        return f"kv_dtype({model_runner.kv_cache_dtype})"
    model_config = model_runner.model_config
    if getattr(model_config, "dtype", None) not in _MODEL_DTYPES:
        return f"model_dtype({model_config.dtype})"

    window, sink = resolve_draft_decode_window(model_runner)
    if window or sink:
        return f"draft_window(window={window},sink={sink})"
    native_window = getattr(model_runner, "sliding_window_size", None)
    if native_window is not None and native_window > 0:
        return f"draft_sliding_window({native_window})"
    translator = model_runner.kv_index_translator
    if getattr(translator, "is_translating", False):
        return "kv_index_translation"

    pool = model_runner.token_to_kv_pool
    if getattr(pool, "full_to_swa_index_mapping", None) is not None:
        return "swa_kv_pool"

    layers = list(_attention_layers(model_runner.model))
    if not layers:
        return "no_attention_layers"
    for layer in layers:
        reason = plan.layer_shape_reason(layer)
        if reason is not None:
            return f"layer{layer.layer_id}:{reason}"
    return None


def build_native(model_runner, topk, speculative_num_steps, stamp):
    """Build the native draft container, or return None to use Triton."""
    from sglang.srt.speculative.draft_utils import _assert_draft_needs_no_conv_sidecar

    _assert_draft_needs_no_conv_sidecar(model_runner)
    reason = unsupported_reason(model_runner, topk)
    if reason is not None:
        logger.info("attention_v100: draft decode on triton (%s)", reason)
        return None
    logger.info("attention_v100: draft decode on the native split-KV kernel")
    return NativeMultiStepDraftBackend(model_runner, topk, speculative_num_steps, stamp)
