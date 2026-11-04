"""The space the LSA index lives in, for a grouped-query model.

Vendored from Prefrontal-AI/latent-sparse-attention @
97b82d695b74ca341965a16bbc2a951c8b0c567a (src/lsa/core/index_space.py),
Apache-2.0, with the SM70 port's changes. See PROVENANCE.md.

The attention logit splits into a rotated part and a position-free part. The
position-free part is an exact sub-term of the true score with no position in
it, so clustering groups keys by content rather than by where they sit.

Heads come in as `[B, H, T, dh]`. Index vectors go out as `[B, T, Hkv * dh]`
fp32 unit rows. The learned variant carries those rows one step further, through
a trained `[Hkv * dh, index_dim]` map into the narrower space the index lives in.
The contract is the same either way - unit rows - so nothing downstream knows
which space it was handed.

The MLA route (a key latent, a Cholesky metric, `regroup_latent`) is not ported:
the target is Qwen3.8-27B, a grouped-query model. See PROVENANCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import NORM_EPS

ROTARY_ENDS = ("none", "prefix", "suffix")


@dataclass(frozen=True)
class RotaryLayout:
    """Where the rotation sits in a head vector, and how many dims it covers."""

    end: str = "none"
    dim: int = 0

    def __post_init__(self) -> None:
        """Validate that the end and dim fields agree with each other."""
        if self.end not in ROTARY_ENDS:
            raise ValueError(f"end must be one of {ROTARY_ENDS} - got {self.end!r}")
        if self.dim < 0:
            raise ValueError(f"dim must not be negative - got {self.dim}")
        if (self.end == "none") != (self.dim == 0):
            raise ValueError(
                f"end {self.end!r} and dim {self.dim} disagree about whether there is a rotation"
            )


def position_free(
    q: torch.Tensor, k: torch.Tensor, rotary: RotaryLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the rotated dims from pre-RoPE query and key heads.

    What remains is an exact, position-independent sub-term of the true score
    rather than an approximation. A model whose rotation covers the whole head
    vector has no such sub-term.

    Args:
        q: Pre-RoPE query heads [..., dh], floating dtype.
        k: Pre-RoPE key heads [..., dh], floating dtype.
        rotary: Where the rotation sits in the head vector and how many dims it covers.

    Returns:
        Query heads with the rotated dims dropped [..., dh - rotary.dim].
        Key heads with the rotated dims dropped [..., dh - rotary.dim].

    Raises:
        ValueError: If `q` and `k` do not share the head dim.
        ValueError: If `rotary.dim` covers the whole head.
    """
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim:
        raise ValueError(f"q and k must share the head dim - {head_dim} != {k.shape[-1]}")
    if rotary.dim >= head_dim:
        raise ValueError(
            f"rotation covers the whole head - {rotary.dim} dims of {head_dim}, "
            "nothing is position-free"
        )
    if rotary.end == "none":
        return q, k
    if rotary.end == "prefix":
        return q[..., rotary.dim :], k[..., rotary.dim :]
    return q[..., : head_dim - rotary.dim], k[..., : head_dim - rotary.dim]


def regroup_partial(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse grouped-query heads into the shared index space, without normalizing.

    `sum_h q_h . k_(h//G) == concat_kv(sum_(h in group) q_h) . concat_kv(k_kv)`,
    so one shared selection scores exactly what the attention heads sum to.

    Args:
        q: Pre-index query heads [B, Hq, Tq, dh], floating.
        k: Pre-index key heads [B, Hkv, Tk, dh], floating.

    Returns:
        Index queries [B, Tq, Hkv * dh] fp32, not normalized.
        Index keys [B, Tk, Hkv * dh] fp32, not normalized.

    Raises:
        ValueError: If `q` and `k` do not share the batch size or the head dim.
        ValueError: If the query head count does not divide evenly by the KV head count.
    """
    batch_size, query_head_count, query_count, head_dim = q.shape
    if k.shape[0] != batch_size:
        raise ValueError(f"q and k must share the batch size - {batch_size} != {k.shape[0]}")
    if k.shape[3] != head_dim:
        raise ValueError(f"q and k must share the head dim - {head_dim} != {k.shape[3]}")
    kv_head_count = k.shape[1]
    if query_head_count % kv_head_count:
        raise ValueError(
            f"query heads must group evenly - {query_head_count} does not divide by {kv_head_count}"
        )
    query_heads_per_kv_head = query_head_count // kv_head_count
    index_queries = (
        q.float()
        .view(batch_size, kv_head_count, query_heads_per_kv_head, query_count, head_dim)
        .sum(2)
        .transpose(1, 2)
        .reshape(batch_size, query_count, kv_head_count * head_dim)
    )
    index_keys = k.float().transpose(1, 2).reshape(batch_size, k.shape[2], kv_head_count * head_dim)
    return index_queries, index_keys


def rownorm(rows: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """Normalize the final dimension in fp32 without changing shape or device.

    Unit rows make the dot products cosines in `[-1, 1]`. A zero row stays zero
    and scores 0 against everything, which is the only sane reading of a key with
    no direction.
    """
    return F.normalize(rows.float(), dim=-1, eps=eps)


def regroup(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse grouped-query heads into unit rows in the shared index space."""
    index_queries, index_keys = regroup_partial(q, k)
    return rownorm(index_queries), rownorm(index_keys)


@dataclass(frozen=True)
class LearnedProjection:
    """Two trained maps that take the position-free space to a narrower one."""

    # [Hkv * dh, index_dim] the projection applied to index queries.
    query_projection: torch.Tensor
    # [Hkv * dh, index_dim] the projection applied to index keys.
    key_projection: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that the two maps describe one space."""
        if self.query_projection.ndim != 2 or self.key_projection.ndim != 2:
            raise ValueError(
                f"both projections must be [Hkv * dh, index_dim] - got "
                f"{tuple(self.query_projection.shape)} and {tuple(self.key_projection.shape)}"
            )
        if self.query_projection.shape != self.key_projection.shape:
            raise ValueError(
                f"the projections must share their shape - "
                f"{tuple(self.query_projection.shape)} != {tuple(self.key_projection.shape)}"
            )

    @property
    def index_dim(self) -> int:
        """Dims of the learned space, the index dim."""
        return int(self.key_projection.shape[1])


def project_partial(rows: torch.Tensor, index_map: torch.Tensor) -> torch.Tensor:
    """Carry index vectors through one learned map without normalizing the result."""
    if rows.shape[-1] != index_map.shape[0]:
        raise ValueError(f"rows have {rows.shape[-1]} dims and the map reads {index_map.shape[0]}")
    return rows.float() @ index_map.float()


def project(rows: torch.Tensor, index_map: torch.Tensor) -> torch.Tensor:
    """Carry index vectors through one learned map and renormalize."""
    return rownorm(project_partial(rows, index_map))
