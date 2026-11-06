"""QPN2 NVFP4 expert layout: repack, layer conversion, call-time marker.

The layout is defined in ``docs/qpn2-layout.md``. A 288-byte group is 32 lanes
x 8 code bytes followed by 32 E4M3 scale bytes, so a reader gets one column's
16 codes from 8 contiguous bytes instead of gathering a permuted word. The
E4M3 group scales stay as they came from the checkpoint (no Q0.7 re-encode) and
the global scale stays raw, because the decode kernel applies the spec's single
FP16 rounding.

Selection is a load-time switch:

* ``SGLANG_SM70_FP4_MOE_LAYOUT=marlin`` (default) -- the existing path;
* ``SGLANG_SM70_FP4_MOE_LAYOUT=qpn2`` -- convert the selected experts to QPN2;
* ``SGLANG_SM70_QPN2_TENSORS=both|w13|w2`` -- which projections convert.

At call time the layout is read off the tensor: QPN2 expert weights are uint8
``[E, tiles*groups*288]``, the Marlin layout is int32. There is no second
switch to keep in sync.

QPN2 is decode-only until the prefill reader lands: a QPN2 weight that reaches
a prefill-shaped call raises instead of running a kernel that cannot read it.
"""

from __future__ import annotations

import os

ENV_LAYOUT = "SGLANG_SM70_FP4_MOE_LAYOUT"
ENV_TENSORS = "SGLANG_SM70_QPN2_TENSORS"
PREFIXES = ("w13", "w2")
GROUP_BYTES = 288


def layout() -> str:
    """``marlin`` (default) or ``qpn2``."""
    value = os.environ.get(ENV_LAYOUT, "").strip().lower()
    if value in ("", "marlin", "default"):
        return "marlin"
    if value == "qpn2":
        return "qpn2"
    raise ValueError(f"{ENV_LAYOUT} must be 'marlin' or 'qpn2'; got {value!r}")


def selected_prefixes() -> set[str]:
    """Expert projections to convert when the layout is ``qpn2``."""
    if layout() != "qpn2":
        return set()
    value = os.environ.get(ENV_TENSORS, "both").strip().lower()
    if value in ("", "both", "all"):
        return set(PREFIXES)
    chosen = {part.strip() for part in value.split(",") if part.strip()}
    unknown = chosen - set(PREFIXES)
    if unknown or not chosen:
        raise ValueError(
            f"{ENV_TENSORS} must be 'both', 'w13' or 'w2'; got {value!r}"
        )
    return chosen


def qpn2_col_from_lane(lane: int) -> int:
    """Storage lane -> logical column inside its 32-column tile (spec section 1)."""
    return ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if (lane & 16) else 0)


def pack_expert(weight_packed, scales, size_n: int, size_k: int):
    """ModelOpt ``[N, K/2]`` uint8 + E4M3 ``[N, K/16]`` -> QPN2 bytes.

    Vectorized form of ``scripts/qpn2_probe/repack_qpn2.pack_qpn2``, which is
    the CPU-verified reference for this layout.
    """
    import torch

    if weight_packed.shape != (size_n, size_k // 2):
        raise ValueError(
            f"weight must be [{size_n}, {size_k // 2}], got {tuple(weight_packed.shape)}"
        )
    if scales.shape != (size_n, size_k // 16):
        raise ValueError(
            f"scales must be [{size_n}, {size_k // 16}], got {tuple(scales.shape)}"
        )
    if size_n % 32 or size_k % 16:
        raise ValueError(f"QPN2 needs N%32==0 and K%16==0, got N={size_n} K={size_k}")
    groups, tiles = size_k // 16, size_n // 32
    packed = weight_packed
    lanes = torch.arange(32, device=packed.device)
    cols = torch.tensor(
        [qpn2_col_from_lane(int(lane)) for lane in lanes], device=packed.device
    )
    # [tiles, 32 logical columns, groups, 8 bytes] -> storage-lane order
    src = packed.view(tiles, 32, groups, 8).index_select(1, cols).to(torch.int16)
    lo, hi = src & 0xF, src >> 4
    codes = torch.zeros(tiles, groups, 32, 8, dtype=torch.int16, device=packed.device)
    for byte in range(8):
        # Byte b holds physical codes 2b (low nibble) and 2b+1 (high). Physical
        # p is logical k(p), which maps source bytes in the pattern below.
        jb = 4 * ((byte >> 2) & 1) + 2 * (byte & 1)
        half = hi if (byte >> 1) & 1 else lo
        codes[:, :, :, byte] = (
            half[:, :, :, jb] | (half[:, :, :, jb + 1] << 4)
        ).permute(0, 2, 1)
    scale_bytes = scales.view(torch.uint8)
    scale_bytes = scale_bytes.view(tiles, 32, groups).index_select(1, cols)
    out = torch.cat(
        [
            codes.reshape(tiles, groups, 256).to(torch.uint8),
            scale_bytes.permute(0, 2, 1).to(torch.uint8),
        ],
        dim=2,
    )
    return out.reshape(-1)


def is_qpn2_weight(tensor) -> bool:
    """True when the expert weight tensor is a QPN2 buffer."""
    import torch

    return (
        hasattr(tensor, "dtype")
        and tensor.dtype == torch.uint8
        and tensor.ndim == 2
    )


def prepare_prefix(layer, prefix: str) -> None:
    """Replace one projection's checkpoint tensors with the QPN2 layout."""
    import torch
    from sglang.srt.layers.utils import copy_or_rebind_param

    weight = getattr(layer, prefix + "_weight")
    scales = getattr(layer, prefix + "_weight_scale")
    global_scales = getattr(layer, prefix + "_weight_scale_2")

    if weight.dtype != torch.uint8 or weight.ndim != 3:
        raise RuntimeError(
            f"QPN2 expects a packed [E, N, K/2] uint8 weight, got "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
    if scales.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"QPN2 expects E4M3 scales, got {scales.dtype}")
    experts, size_n, packed_k = (int(d) for d in weight.shape)
    size_k = packed_k * 2
    if scales.shape != (experts, size_n, size_k // 16):
        raise RuntimeError(
            f"QPN2 scales must be [{experts}, {size_n}, {size_k // 16}], "
            f"got {tuple(scales.shape)}"
        )

    if global_scales.ndim == 2:
        if not torch.equal(global_scales[:, 0], global_scales[:, 1]):
            raise RuntimeError(
                "Unequal NVFP4 gate/up global scales are unsupported on SM70"
            )
        global_scales = global_scales[:, 0]
    global_scales = global_scales.to(torch.float32).contiguous()

    tiles, groups = size_n // 32, size_k // 16
    out = torch.empty(
        (experts, tiles * groups * GROUP_BYTES),
        dtype=torch.uint8,
        device=weight.device,
    )
    for expert in range(experts):
        out[expert] = pack_expert(weight[expert], scales[expert], size_n, size_k)

    copy_or_rebind_param(layer, prefix + "_weight", out)
    # The raw per-expert global scale: the kernel rounds e4m3 * gs once.
    copy_or_rebind_param(layer, prefix + "_weight_scale_2", global_scales)

    swizzled = prefix + "_blockscale_swizzled"
    if hasattr(layer, swizzled):
        setattr(layer, swizzled, None)
