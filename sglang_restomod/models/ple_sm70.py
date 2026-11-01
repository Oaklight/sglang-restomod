"""SM70 substitute for the fp8 PLE host-embedding gather.

With ``--ple-offload-embedding`` the model reads its n-gram table from host
memory with a Triton kernel. The pinned kernel casts the table pointer to
``tl.float8e4nv``, which the SM70 backend rejects:

    type fp8e4nv not supported in this architecture.
    The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')

Neither supported type is e4m3fn, so the byte is decoded here with integer
arithmetic instead: sign, 4-bit exponent, 3-bit mantissa, with the e4m3fn
subnormal and NaN rules. The result is the same bf16 row the kernel emits on
SM80+.

The call site resolves the kernel through a module global, so assigning the
module attribute is enough; no source patch is needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MODULE = "sglang.srt.models.qwen4_exp"
KERNEL = "_gather_ple_embedding_from_pinned_kernel"

_INSTALLED = False


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_ple_embedding_sm70(
        weight_ptr,
        ids_ptr,
        output_ptr,
        embedding_dim,
        tp_vocab_start,
        tp_vocab_end,
        is_fp8: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        global_idx = tl.load(ids_ptr + row_id)
        in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
        local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < embedding_dim
        base = local_idx * embedding_dim + offsets
        if is_fp8:
            raw = tl.load(
                weight_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8)) + base,
                mask=mask,
                other=0,
            ).to(tl.int32)
            sign = (raw >> 7) & 1
            exponent = (raw >> 3) & 0xF
            mantissa = (raw & 7).to(tl.float32)
            # Normal: (1 + m/8) * 2^(e-7). Subnormal: (m/8) * 2^-6.
            value = tl.where(
                exponent == 0,
                mantissa * 0.001953125,
                (1.0 + mantissa * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0),
            )
            value = tl.where(sign == 1, -value, value)
            value = tl.where((exponent == 15) & ((raw & 7) == 7), float("nan"), value)
            values = value.to(tl.bfloat16)
        else:
            values = tl.load(
                weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16)) + base,
                mask=mask,
                other=0.0,
            )
        tl.store(
            output_ptr + row_id * embedding_dim + offsets,
            tl.where(in_range, values, 0.0),
            mask=mask,
        )

    return gather_ple_embedding_sm70


def install() -> bool:
    """Replace the pinned gather kernel on SM70. Returns True when patched."""
    global _INSTALLED
    if _INSTALLED:
        return True
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return False
    try:
        from sglang.srt.models import qwen4_exp
    except Exception:  # noqa: BLE001 - absent model module means nothing to patch
        return False
    if not hasattr(qwen4_exp, KERNEL):
        logger.warning("sglang-restomod: %s.%s not found", MODULE, KERNEL)
        return False

    setattr(qwen4_exp, KERNEL, _kernel())
    _INSTALLED = True
    logger.info("sglang-restomod: PLE host-embedding gather patched for SM70")
    return True


def register() -> None:
    """Component entry point."""
    install()
