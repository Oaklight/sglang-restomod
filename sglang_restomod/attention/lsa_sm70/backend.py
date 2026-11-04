"""LSA attention backend for SM70 (V100).

The backend replaces the full-attention layers' read with LSA's: build the
index from the layer's keys and queries, cluster the closed cluster blocks into
buckets, select the keys each query reads, and read them through the pinned
Triton decode kernel.

Design (flat, D-LSA):

- index space: every KV head's position-free dims side by side, each group's
  queries summed beside them, through the released ``[768, 64]`` maps;
- buckets: 512-key cluster blocks, pooled by 4, Ward-clustered into 64 buckets
  of at most 16 keys;
- read: the selected keys, then a 512-key window and 4 sinks.

The index is kept per (layer, request): closed blocks contribute centroids and
members; the open block keeps at most 512 projected keys. A prefill builds it
from the request's prefix (if any) and the new tokens; a decode appends one key.

Tensor parallel: each rank holds ``layer.tp_k_head_num`` KV heads. The maps are
sliced to the rank's rows, each rank projects its own heads, and the partials
are summed across the attention TP group, as the reference's ``HeadShard`` does.

The selection runs on the host and reads ``req_pool_indices`` back, so a CUDA
graph would replay one batch's lists. Serve with ``--disable-cuda-graph``.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from .constants import PAD
from .design import MAP_ENV, LsaDesign, RequestIndex, row_segments
from .index_space import rownorm
from .read import lsa_sparse_read
from .selection import select_lists

logger = logging.getLogger(__name__)


class LsaSm70Backend(AttentionBackend):
    """SM70 full attention that reads the keys an LSA index selects."""

    def __init__(self, model_runner) -> None:
        super().__init__()
        self.model_runner = model_runner
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        if self.page_size != 1:
            raise RuntimeError(
                "lsa_sm70 addresses KV slots one token at a time - serve with --page-size 1, "
                f"this server pages {self.page_size}"
            )
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.req_to_token = self.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.data_type = model_runner.dtype
        self._check_kv_dtype()
        self.design = LsaDesign.from_env()
        map_path = os.environ.get(MAP_ENV)
        if not map_path:
            raise RuntimeError(f"lsa_sm70 needs the D-LSA projection file in {MAP_ENV}")
        self._map = torch.load(map_path, map_location="cpu", weights_only=True)
        if not isinstance(self._map, dict) or not self._map:
            raise ValueError(f"{map_path} is not a D-LSA projection file")
        self._index_dim = self._check_maps()
        if self._index_dim != self.design.index_dim:
            raise ValueError(
                f"the map's index dim is {self._index_dim}, the design's {self.design.index_dim}"
            )
        self._projections: dict[int, tuple[torch.Tensor, torch.Tensor, bool]] = {}
        self._indexes: dict[tuple[int, int], RequestIndex] = {}
        self._attn_tp_size, self._attn_tp_rank, dp_size, cp_size = self._parallel()
        if dp_size > 1 or cp_size > 1:
            raise RuntimeError(
                "lsa_sm70 serves with attention tensor parallel only - "
                f"this launch has data parallel {dp_size} and context parallel {cp_size}"
            )
        self._rotary_dim: int | None = None
        self._position_free: int | None = None

    # -- setup -------------------------------------------------------------

    def _check_kv_dtype(self) -> None:
        if self.kv_cache_dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                "lsa_sm70 reads the index from a bf16 or fp16 KV cache - "
                f"this one is {self.kv_cache_dtype}. Use --kv-cache-dtype auto."
            )

    def _check_maps(self) -> int:
        dims = set()
        for name, tensor in self._map.items():
            layer, _, kind = str(name).partition(".")
            if not layer.isdigit() or kind not in ("W_Q_idx", "W_K_idx"):
                raise ValueError(
                    f"the D-LSA map holds {name!r}; "
                    "expected '<layer>.W_Q_idx' and '<layer>.W_K_idx'"
                )
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"the D-LSA map holds a non-tensor {name!r}")
            dims.add(int(tensor.shape[1]))
        if len(dims) != 1:
            raise ValueError(
                f"the D-LSA map's projections disagree on the index dim: {sorted(dims)}"
            )
        return dims.pop()

    @staticmethod
    def _parallel() -> tuple[int, int, int, int]:
        try:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            return (
                int(parallel.attn_tp_size),
                int(parallel.attn_tp_rank),
                int(parallel.attn_dp_size),
                int(parallel.attn_cp_size),
            )
        except Exception:  # noqa: BLE001 - a unit test has no parallel state
            return 1, 0, 1, 1

    def _geometry(self, layer) -> tuple[int, int, int, int]:
        """Return rotary dim, position-free dim, KV heads on this rank, queries per group."""
        if self._rotary_dim is None:
            text = self.model_runner.model_config.hf_text_config
            factor = float(getattr(text, "partial_rotary_factor", 1.0))
            self._rotary_dim = int(layer.head_dim * factor)
            self._position_free = layer.head_dim - self._rotary_dim
            if self._position_free <= 0:
                raise RuntimeError(
                    "lsa_sm70 needs a partially rotated head - this one rotates all "
                    f"{layer.head_dim} dims"
                )
        kv_per_rank = int(layer.tp_k_head_num)
        if int(layer.tp_q_head_num) % kv_per_rank:
            raise RuntimeError(
                f"lsa_sm70 needs whole query groups per rank - {layer.tp_q_head_num} "
                f"queries over {kv_per_rank} KV heads"
            )
        return (
            self._rotary_dim,
            self._position_free,
            kv_per_rank,
            int(layer.tp_q_head_num) // kv_per_rank,
        )

    def _projection(self, layer) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return this rank's query map, key map and whether it adds the key."""
        layer_id = int(layer.layer_id)
        cached = self._projections.get(layer_id)
        if cached is not None:
            return cached
        _, position_free, kv_per_rank, _ = self._geometry(layer)
        kv_heads = int(layer.tp_k_head_num) * self._attn_tp_size
        if self._attn_tp_size <= kv_heads and kv_heads % self._attn_tp_size == 0:
            kv_first = self._attn_tp_rank * kv_per_rank
            counts_key = True
        elif self._attn_tp_size > kv_heads and self._attn_tp_size % kv_heads == 0:
            replicas = self._attn_tp_size // kv_heads
            kv_first = self._attn_tp_rank // replicas
            counts_key = self._attn_tp_rank % replicas == 0
        else:
            raise RuntimeError(
                f"lsa_sm70 cannot split {kv_heads} KV heads over {self._attn_tp_size} ranks"
            )
        start = kv_first * position_free
        stop = (kv_first + kv_per_rank) * position_free
        query = self._map[f"{layer_id}.W_Q_idx"][start:stop].float().to(self.device)
        key = self._map[f"{layer_id}.W_K_idx"][start:stop].float().to(self.device)
        self._projections[layer_id] = (query, key, counts_key)
        return self._projections[layer_id]

    # -- index -------------------------------------------------------------

    def _project_qk(
        self, q3: torch.Tensor, k3: torch.Tensor, layer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the rank's heads, all-reduce the partials, and normalize."""
        rotary_dim, position_free, kv_per_rank, per_group = self._geometry(layer)
        query_map, key_map, counts_key = self._projection(layer)
        rows = q3.shape[0]
        q = q3.reshape(rows, kv_per_rank, per_group, q3.shape[-1])[..., rotary_dim:]
        q = q.float().sum(2).reshape(rows, kv_per_rank * position_free)
        query = q @ query_map
        k = k3.reshape(rows, kv_per_rank, k3.shape[-1])[..., rotary_dim:]
        k = k.float().reshape(rows, kv_per_rank * position_free)
        key = k @ key_map if counts_key else torch.zeros_like(query)
        partial = torch.cat([query, key], dim=1)
        if self._attn_tp_size > 1:
            from sglang.srt.layers.dp_attention import attn_tp_all_reduce

            partial = attn_tp_all_reduce(partial)
        return rownorm(partial[:, : self._index_dim]), rownorm(partial[:, self._index_dim :])

    def _project_keys(self, k3: torch.Tensor, layer) -> torch.Tensor:
        """Project keys only, all-reducing the key partial."""
        rotary_dim, position_free, kv_per_rank, _ = self._geometry(layer)
        _, key_map, counts_key = self._projection(layer)
        rows = k3.shape[0]
        k = k3.reshape(rows, kv_per_rank, k3.shape[-1])[..., rotary_dim:]
        k = k.float().reshape(rows, kv_per_rank * position_free)
        partial = (
            k @ key_map
            if counts_key
            else torch.zeros((rows, self._index_dim), dtype=torch.float32, device=k3.device)
        )
        if self._attn_tp_size > 1:
            from sglang.srt.layers.dp_attention import attn_tp_all_reduce

            partial = attn_tp_all_reduce(partial)
        return rownorm(partial)

    def _index_of(self, layer, request: int) -> RequestIndex:
        key = (int(layer.layer_id), int(request))
        index = self._indexes.get(key)
        if index is None:
            index = RequestIndex(self.design, self.device)
            self._indexes[key] = index
        return index

    def _catch_up(self, index: RequestIndex, request: int, length: int, layer) -> None:
        """Rebuild the index from the KV pool when it does not hold the request's prefix."""
        if index.length == length:
            return
        index.reset()
        if length:
            slots = self.req_to_token[request, :length].long()
            keys = self.token_to_kv_pool.get_key_buffer(layer.layer_id)[slots]
            index.append(self._project_keys(keys, layer))

    # -- metadata ----------------------------------------------------------

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False) -> None:
        self.forward_metadata = forward_batch

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        raise RuntimeError("lsa_sm70 selects keys on the host - serve with --disable-cuda-graph")

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0

    # -- forwards ----------------------------------------------------------

    def _write_kv(self, k, v, layer, forward_batch, save_kv_cache: bool) -> None:
        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

    def _positions(self, forward_batch) -> torch.Tensor:
        positions = forward_batch.positions
        if positions.dim() > 1:
            positions = positions[0]
        return positions.long()

    def _select(
        self, query: torch.Tensor, positions: torch.Tensor, index: RequestIndex
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return select_lists(
            query,
            positions,
            index.buckets(),
            top_p=self.design.top_p,
            top_k=self.design.top_k,
            sink_size=self.design.sink_size,
            window_size=self.design.window_size,
        )

    def _lists_to_slots(
        self, positions: torch.Tensor, request: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = (positions >= 0).sum(1).to(torch.int32)
        slots = self.req_to_token[request, positions.clamp(min=0).long()]
        return torch.where(positions >= 0, slots, PAD).to(torch.int32), lengths

    def _run(
        self,
        q3: torch.Tensor,
        key_index: torch.Tensor,
        query_index: torch.Tensor,
        layer,
        forward_batch,
    ) -> torch.Tensor:
        """Select each row's keys from its request's index and read them."""
        rows = q3.shape[0]
        positions = self._positions(forward_batch)
        requests = forward_batch.req_pool_indices.tolist()
        if forward_batch.forward_mode.is_extend():
            counts = [int(count) for count in forward_batch.extend_seq_lens_cpu]
            befores = [int(length) for length in forward_batch.extend_prefix_lens_cpu]
        else:
            counts = None
            befores = [int(positions[row]) for row, _ in enumerate(requests)]
        lists = torch.full((rows, self.design.width), PAD, dtype=torch.int32, device=self.device)
        lengths = torch.zeros(rows, dtype=torch.int32, device=self.device)
        for (request, start, stop), before in zip(row_segments(requests, counts), befores):
            index = self._index_of(layer, request)
            self._catch_up(index, request, before, layer)
            index.append(key_index[start:stop])
            picked, _ = self._select(query_index[start:stop], positions[start:stop], index)
            slots, slot_lengths = self._lists_to_slots(picked, request)
            lists[start:stop] = slots
            lengths[start:stop] = slot_lengths
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        out = lsa_sparse_read(
            q3, k_cache, v_cache, lists, lengths, layer.scaling, self._logit_cap(layer)
        )
        return out.reshape(rows, layer.tp_q_head_num * layer.v_head_dim)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            raise NotImplementedError(
                "lsa_sm70 reads a plain prefill or decode - serve without speculative decoding"
            )
        rows = q.shape[0]
        q3 = q.reshape(rows, layer.tp_q_head_num, layer.head_dim).contiguous()
        k3 = k.reshape(rows, layer.tp_k_head_num, layer.head_dim).contiguous()
        self._write_kv(k3, v, layer, forward_batch, save_kv_cache)
        query_index, key_index = self._project_qk(q3, k3, layer)
        return self._run(q3, key_index, query_index, layer, forward_batch)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache)

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
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache)

    @staticmethod
    def _logit_cap(layer) -> float:
        return float(getattr(layer, "logit_cap", 0.0) or 0.0)
