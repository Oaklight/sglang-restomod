"""Stage 1: DSV4 paged MLA in torch, for a platform without flash_mla.

This stands in for ``flash_mla_with_kvcache`` -- the call the pin makes for
decode **and** for any prefill chunk it does not route to its sparse body.

The cache is a paged byte tensor: ``[pages, page_size, 1, bytes_per_token]``
uint8, where the per-token layout is one of the entries in
:mod:`sglang_restomod.attention.dsv4_sm70.kv_layout`. Attention is MLA in its
absorbed form: the 512-dim latent is both the key and the value, so the output
is a weighted sum of the same vectors that produced the scores.

Everything here is written for correctness. It loops over pages and query
chunks and materializes far more than a kernel would; stage 2 is where that
changes.

**Scope** is the stage-1 profile (TP1, no speculative decoding, no CUDA
graphs). Anything else raises with the profile named rather than producing a
plausible-looking wrong answer.
"""

from __future__ import annotations

import torch

from . import kv_layout

STAGE1_PROFILE = "tp1|no-spec|no-cuda-graph|batch-decode|chunked-prefill"

# Query rows per chunk. Only a memory bound: the cache loop is over pages.
_QUERY_CHUNK = 64


def _layout_for(cache_bytes_per_token: int, interleaved: bool) -> kv_layout.Layout:
    """Pick the layout from the cache's own per-token byte count."""
    for name in ("V41_NEW", "V41_FP4_NEW", "V41", "V41_FP4", "V4", "UNIFORM_FP8"):
        layout = kv_layout.get_layout(name)
        if layout.bytes_per_token() == cache_bytes_per_token and layout.interleaved == interleaved:
            return layout
    raise ValueError(
        f"no DSV4 KV layout has {cache_bytes_per_token} bytes per token with "
        f"interleaved={interleaved}; the cache and this component disagree"
    )


def _decode_pages(
    cache_flat: torch.Tensor,
    page_ids: torch.Tensor,
    layout: kv_layout.Layout,
    page_size: int,
    device,
) -> torch.Tensor:
    """Dequantize ``page_ids`` into ``[len(page_ids) * page_size, dims]`` fp32.

    Page ids below zero are dropped by the caller, not here: a negative id is
    the pin's "no page" marker and must not read page 0's data.
    """
    import torch as _t

    page_bytes = kv_layout.page_bytes(layout, page_size)
    rows = []
    for page in page_ids.tolist():
        base = int(page) * page_bytes
        page_tensor = cache_flat[base : base + page_bytes]
        tok = []
        for slot in range(page_size):
            d_off, s_off = kv_layout.row_offsets(layout, page_size, slot)
            tok.append(
                kv_layout.dequantize_token(
                    layout,
                    page_tensor[d_off : d_off + layout.data_bytes],
                    page_tensor[s_off : s_off + layout.scale_bytes],
                )
            )
        rows.append(_t.stack(tok))
    if not rows:
        dims = 512 if layout.data_dtype == "e4m3" else 512
        return _t.zeros((0, dims), dtype=_t.float32, device=device)
    return _t.cat(rows)


def paged_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table=None,
    cache_seqlens=None,
    head_dim_v: int = 512,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q=None,
    descale_k=None,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length=None,
    kv_format=None,
    *,
    use_legacy: bool = True,
    _interleaved: bool = False,
):
    """``(out, lse)`` with ``out`` [T, H, head_dim_v] and ``lse`` [T, H]."""
    _ = (
        block_table,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
        kv_format,
        use_legacy,
        extra_topk_length,
    )
    _ = extra_indices_in_kvcache, extra_k_cache  # ratio-4/128 keys: not in stage 1
    if indices is None:
        raise Stage1Error(
            "paged decode needs sparse page indices; a dense page table is not implemented"
        )
    if not is_fp8_kvcache:
        raise Stage1Error(f"stage 1 assumes an fp8 KV cache; got is_fp8_kvcache={is_fp8_kvcache}")
    if causal:
        raise Stage1Error(
            "the DSV4 call site passes causal=False; a causal variant is not implemented"
        )

    t, _, heads, head_dim = q.shape
    if head_dim != 512:
        raise Stage1Error(f"stage 1 expects head_dim 512, got {head_dim}")
    scale = float(softmax_scale) if softmax_scale is not None else head_dim**-0.5

    cache_flat = k_cache.view(-1)
    page_size = int(k_cache.shape[1])
    bpt = int(k_cache.shape[-1] * k_cache.shape[2])
    layout = _layout_for(bpt, interleaved=_interleaved)

    idx = indices.reshape(t, -1)
    lens = topk_length.reshape(t).to(torch.int64) if topk_length is not None else None

    out = torch.empty((t, heads, head_dim_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((t, heads), dtype=torch.float32, device=q.device)

    qf = q[:, 0].to(torch.float32)  # [T, H, 512]
    sink = None if attn_sink is None else attn_sink.to(torch.float32)

    for start in range(0, t, _QUERY_CHUNK):
        stop = min(start + _QUERY_CHUNK, t)
        rows = idx[start:stop]
        n_pages = rows.shape[1]
        # Unique pages across the chunk: the same page usually serves many rows.
        flat_ids = rows.reshape(-1)
        valid = flat_ids >= 0
        keep = torch.unique(flat_ids[valid], sorted=True)
        decoded = (
            _decode_pages(cache_flat, keep, layout, page_size, q.device) if keep.numel() else None
        )

        keys = torch.zeros((n_pages * page_size, head_dim), dtype=torch.float32, device=q.device)
        if decoded is not None:
            # Build an id -> row-block map so a page is dequantized once.
            order = {int(p): i for i, p in enumerate(keep.tolist())}
            for slot in range(n_pages):
                for r in range(stop - start):
                    pid = int(rows[r, slot])
                    if pid < 0:
                        continue
                    src = order[pid] * page_size
                    keys[slot * page_size : (slot + 1) * page_size] = decoded[src : src + page_size]

        scores = torch.einsum("thd,kd->thk", qf[start:stop], keys) * scale  # [R, H, K]

        # Mask in cache columns only, then append the sink column. A sink is a
        # logit with no value behind it: it enters the denominator and nothing
        # else, so it must not be masked by `topk_length`.
        if lens is not None:
            limit = lens[start:stop]
            pos = torch.arange(scores.shape[-1], device=q.device)
            visible = pos.view(1, -1) < limit.view(-1, 1)
            scores = scores.masked_fill(~visible.unsqueeze(1), float("-inf"))
        if sink is not None:
            scores = torch.cat(
                [scores, sink.view(1, heads, 1).expand(stop - start, heads, 1)], dim=-1
            )

        row_max = scores.amax(dim=-1, keepdim=True)
        unnormalized = torch.exp(scores - row_max)
        denom = unnormalized.sum(dim=-1, keepdim=True)
        # lse is the unnormalized denominator in log space; taking it from the
        # normalized probabilities would return the row max instead.
        lse[start:stop] = (row_max + torch.log(denom)).squeeze(-1)
        probs = unnormalized / denom
        if sink is not None:
            probs = probs[..., :-1]
        out[start:stop] = torch.einsum("rhk,kd->rhd", probs, keys).to(torch.bfloat16)

    return out, lse


class Stage1Error(RuntimeError):
    """A call outside the stage-1 profile, named rather than approximated."""


def forward_prefill_dense(*_args, **_kwargs):
    raise NotImplementedError(
        "attention_dsv4_sm70 stage 1: the sparse-body prefill is not implemented yet. "
        f"Profile: {STAGE1_PROFILE}."
    )
