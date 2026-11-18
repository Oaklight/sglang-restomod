"""The DSV4 KV page layouts, as arithmetic, from the pin's own header.

Source of truth: ``sgl_kernel/deepseek_v4/kv_layout.cuh`` at the pin. Each
layout is a per-token byte pattern plus a page-level arrangement:

====================  ========  =========  =======  =========
layout                data B    scale B    tile     bytes/token
====================  ========  =========  =======  =========
``V4``                576       8          64       584
``V41``               512       16         32       528
``V41_FP4``           256       32         16       288
``UNIFORM_FP8``       512       0          512      512
====================  ========  =========  =======  =========

``V4``'s 576 data bytes are 448 e4m3 followed by 64 bf16 RoPE values; its 8
scale bytes are 7 ue8m0 (one per 64) and a pad. ``V41`` quantizes all 512 dims
-- the RoPE dims too, after an in-kernel rotation -- so its data is 512 e4m3
with 16 ue8m0 scales, one per 32. ``V41_FP4`` packs e2m1 two per byte.

**Two page arrangements**, and confusing them is the error this module exists to
prevent:

* blocked (``V4``, ``V41``, ``V41_FP4``): a page holds ``page_size`` data rows
  followed by ``page_size`` scale rows, so scales start at
  ``page_size * kDataBytes`` and each token's scales are at ``+slot * kScaleBytes``;
* interleaved (``V41_NEW``, ``V41_FP4_NEW``): each token's scales immediately
  follow its own data, at ``slot * kBytesPerToken + kDataBytes``, with the page
  padded to the reader's row stride.

Which one is in use is ``DeepSeekV4TokenToKVPool.get_swa_key_layout().is_interleaved``
(``_swa_interleaved`` in the backend), so both are implemented here and the
modern interleaved layout is selected by that flag, not by a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; the runtime import stays inside the functions
    import torch

# torch is imported inside the functions that need it: this module is reachable
# from `register()` paths, and the plugin's rule is that no torch appears at
# module import.

# Values from KVLayoutTraits<kLayout> in kv_layout.cuh.
E8M0_BIAS = 127
E8M0_NAN = 255


@dataclass(frozen=True)
class Layout:
    """One KV layout's per-token geometry."""

    name: str
    data_bytes: int
    scale_bytes: int
    tile: int  # elements covered by one scale
    page_align: int
    data_dtype: str  # "e4m3" or "e2m1"
    rope_bf16: bool = False  # V4 keeps 64 RoPE dims as bf16 inside the data bytes
    interleaved: bool = False  # scales follow their own token, not the whole page

    def bytes_per_token(self) -> int:
        return self.data_bytes + self.scale_bytes


LAYOUTS: dict[str, Layout] = {
    "V4": Layout("V4", 576, 8, 64, 576, "e4m3", rope_bf16=True),
    "V41": Layout("V41", 512, 16, 32, 512, "e4m3"),
    "V41_NEW": Layout("V41_NEW", 512, 16, 32, 528, "e4m3", interleaved=True),
    "V41_FP4": Layout("V41_FP4", 256, 32, 16, 256, "e2m1"),
    "V41_FP4_NEW": Layout("V41_FP4_NEW", 256, 32, 16, 288, "e2m1", interleaved=True),
    "UNIFORM_FP8": Layout("UNIFORM_FP8", 512, 0, 512, 512, "e4m3"),
}


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(
            f"unknown DSV4 KV layout {name!r}; known: {', '.join(sorted(LAYOUTS))}"
        ) from None


def page_bytes(layout: Layout, page_size: int) -> int:
    """Bytes of one page, padded up to the reader's row stride."""
    raw = page_size * layout.bytes_per_token()
    return (raw + layout.page_align - 1) // layout.page_align * layout.page_align


def row_offsets(layout: Layout, page_size: int, slot: int) -> tuple[int, int]:
    """Byte offsets of ``slot``'s data and scales within its page.

    Blocked layouts put every data row first and every scale row after, so the
    two offsets diverge by ``page_size * data_bytes``; interleaved layouts keep
    a token's scales next to its data.
    """
    if slot < 0 or slot >= page_size:
        raise ValueError(f"slot {slot} outside a page of {page_size}")
    if layout.interleaved:
        return slot * layout.bytes_per_token(), slot * layout.bytes_per_token() + layout.data_bytes
    return slot * layout.data_bytes, page_size * layout.data_bytes + slot * layout.scale_bytes


def decode_e8m0(bytes_u8: torch.Tensor) -> torch.Tensor:
    """E8M0 scale bytes to float32 multipliers: ``2 ** (b - 127)``, 0xFF is NaN."""
    import torch

    raw = bytes_u8.to(torch.int32)
    exponent = (raw - E8M0_BIAS).clamp(-127, 127).to(torch.float32)
    scale = torch.exp2(exponent)
    return torch.where(raw == E8M0_NAN, torch.full_like(scale, float("nan")), scale)


def decode_e4m3(values_u8: torch.Tensor) -> torch.Tensor:
    """E4M3-FN bytes to float32 without a torch fp8 cast dependency.

    sm70 has no fp8 arithmetic, but the ``float8_e4m3fn`` dtype views fine; the
    cast to fp32 is a table lookup the CPU/GPU does exactly, so this is used
    rather than hand-decoding bits.
    """
    import torch

    return values_u8.view(torch.float8_e4m3fn).to(torch.float32)


def dequantize_token(
    layout: Layout,
    data_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    out_dtype=None,
):
    """One token's KV row, dequantized: ``[dims]`` where ``dims = data_bytes``."""
    import torch

    out_dtype = out_dtype or torch.float32

    if layout.rope_bf16:
        # V4: the first 448 bytes are e4m3 with per-64 scales; the last 64 dims
        # are bf16 and carry no scale at all.
        nope = decode_e4m3(data_u8[:448])
        groups = 448 // layout.tile  # 7
        mult = decode_e8m0(scale_u8[:groups]).to(torch.float32)
        nope = (nope.view(groups, layout.tile) * mult[:, None]).reshape(448)
        rope = data_u8[448:576].view(torch.bfloat16).to(torch.float32)
        return torch.cat([nope, rope]).to(out_dtype)

    if layout.data_dtype == "e4m3":
        values = decode_e4m3(data_u8)
    else:
        raise NotImplementedError(
            "e2m1 (V41_FP4) rows are packable but have not been needed yet; mini-real is V41 fp8"
        )
    groups = layout.data_bytes // layout.tile
    mult = decode_e8m0(scale_u8[:groups]).to(torch.float32)
    return (values.view(groups, layout.tile) * mult[:, None]).reshape(-1).to(out_dtype)
