"""The flat design's knobs and a request's bucket table.

Nothing here imports sglang, so the index maintenance is testable on a CPU-only
machine against the plain-torch oracle in ``reference``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .reference import Buckets, build_buckets

MAP_ENV = "SGLANG_RESTOMOD_LSA_MAP"
TOP_K_ENV = "SGLANG_RESTOMOD_LSA_TOP_K"
TOP_P_ENV = "SGLANG_RESTOMOD_LSA_TOP_P"
WINDOW_ENV = "SGLANG_RESTOMOD_LSA_WINDOW"
SINK_ENV = "SGLANG_RESTOMOD_LSA_SINK"


@dataclass(frozen=True)
class LsaDesign:
    """The flat design's knobs, from the environment or the paper's defaults."""

    top_k: int = 2048
    top_p: int = 512
    window_size: int = 512
    sink_size: int = 4
    pool_size: int = 4
    bucket_size: int = 8
    bucket_cap: int = 16
    index_dim: int = 64

    @classmethod
    def from_env(cls, env=None) -> LsaDesign:
        source = os.environ if env is None else env

        def read(name: str, default: int) -> int:
            raw = source.get(name)
            if raw is None or raw == "":
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
            return value

        design = cls(
            top_k=read(TOP_K_ENV, 2048),
            top_p=read(TOP_P_ENV, 512),
            window_size=read(WINDOW_ENV, 512),
            sink_size=read(SINK_ENV, 4),
        )
        if design.bucket_size % design.pool_size:
            raise ValueError("pool_size must divide bucket_size")
        if design.window_size % design.bucket_size:
            raise ValueError("bucket_size must divide window_size")
        if design.bucket_cap < 2 * design.bucket_size:
            raise ValueError("bucket_cap must be at least twice bucket_size")
        return design

    @property
    def cluster_block_size(self) -> int:
        """Keys in a cluster block - the window size, as in the reference."""
        return self.window_size

    @property
    def width(self) -> int:
        """Selected slots per query, including the window and the sinks."""
        return self.top_k + self.window_size + self.sink_size


def row_segments(requests: list[int], extend_lens: list[int] | None) -> list[tuple[int, int, int]]:
    """Split a batch's rows by request: (request, first row, row past the last).

    ``extend_lens`` is one count a request for a prefill; None means one row a
    request, as a decode batch has.
    """
    if extend_lens is None:
        return [(request, row, row + 1) for row, request in enumerate(requests)]
    segments = []
    start = 0
    for request, count in zip(requests, extend_lens):
        segments.append((request, start, start + count))
        start += count
    return segments


class RequestIndex:
    """One request's bucket table for one layer.

    Closed cluster blocks contribute centroids and members; the open block keeps
    at most ``cluster_block_size`` projected keys until it closes. Keys are
    appended in position order, so a member is also a position.
    """

    def __init__(self, design: LsaDesign, device: torch.device) -> None:
        self.design = design
        self.device = device
        self.centroids: list[torch.Tensor] = []
        self.members: list[torch.Tensor] = []
        self.sizes: list[torch.Tensor] = []
        self.open: list[torch.Tensor] = []
        self.open_count = 0
        self.closed = 0

    @property
    def length(self) -> int:
        """Keys the index holds, closed blocks and the open one."""
        return self.closed + self.open_count

    def reset(self) -> None:
        """Drop the table, keeping the design."""
        self.centroids.clear()
        self.members.clear()
        self.sizes.clear()
        self.open.clear()
        self.open_count = 0
        self.closed = 0

    def append(self, keys: torch.Tensor) -> None:
        """Append projected index keys [n, index_dim], unit rows, in position order."""
        start = 0
        while start < keys.shape[0]:
            take = min(self.design.cluster_block_size - self.open_count, keys.shape[0] - start)
            self.open.append(keys[start : start + take])
            self.open_count += take
            start += take
            if self.open_count == self.design.cluster_block_size:
                self._close_block()

    def _close_block(self) -> None:
        block = torch.cat(self.open, dim=0)
        buckets = build_buckets(
            block,
            pool_size=self.design.pool_size,
            bucket_size=self.design.bucket_size,
            bucket_cap=self.design.bucket_cap,
            cluster_block_size=self.design.cluster_block_size,
        )
        members = buckets.bucket_members.clone()
        members[members >= 0] += self.closed
        self.centroids.append(buckets.bucket_centroids)
        self.members.append(members)
        self.sizes.append(buckets.bucket_sizes)
        self.open.clear()
        self.open_count = 0
        self.closed += self.design.cluster_block_size

    def buckets(self) -> Buckets:
        """Return the closed blocks' buckets, block after block."""
        if not self.centroids:
            empty = torch.empty((0, self.design.index_dim), dtype=torch.float32, device=self.device)
            return Buckets(
                empty,
                torch.empty((0, self.design.bucket_cap), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.int32, device=self.device),
                self.design.cluster_block_size // self.design.bucket_size,
                self.design.cluster_block_size,
            )
        return Buckets(
            torch.cat(self.centroids, dim=0),
            torch.cat(self.members, dim=0),
            torch.cat(self.sizes, dim=0),
            self.design.cluster_block_size // self.design.bucket_size,
            self.design.cluster_block_size,
        )
