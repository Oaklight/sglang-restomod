"""One-launch paged MQA logits for SM70.

Semantics are the torch twin's (`dsv4/indexer.py:82`). Pinned details: `relu`
before the head weighting; the per-token fp32 scale applied once, after the
relu-weighted head sum; and the `bmm` result rounded to bf16 before `relu`,
which is what `torch.bmm` on bf16 operands returns and what the top-64 check
sees.

Triton on sm70 has no fp8 dtype and no tensor cores, so e4m3 is decoded by
integer arithmetic and the dot runs as fp32 FMA. Bytes are loaded as uint8
straight into the [tokens, 128] operand — each byte read once, rows coalesced —
and the four fp32 scale bytes of a row are a second, 4-byte load.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f32(b):
    sign = (b >> 7) & 1
    expo = (b >> 3) & 0xF
    mant = b & 0x7
    bits = ((expo + 120) << 23) | (mant << 20)  # (1 + m/8) * 2**(e-7)
    val = bits.to(tl.float32, bitcast=True)
    sub = mant.to(tl.float32) * (1.0 / 512.0)    # (m/8) * 2^-6
    val = tl.where(expo == 0, sub, val)
    return tl.where(sign == 1, -val, val)


@triton.jit
def paged_mqa_logits_kernel(
    q_ptr, kv_ptr, w_ptr, seq_ptr, pt_ptr, out_ptr,
    stride_qb, stride_qh,   # bytes
    stride_kp,              # bytes per page
    stride_wb,
    stride_ptb,
    stride_ob,
    S_MAX: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    DC: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)

    hd = tl.arange(0, H)
    dd = tl.arange(0, D)
    rows = tl.arange(0, BLOCK)


    page = tl.maximum(tl.load(pt_ptr + b * stride_ptb + p), 0)
    base = kv_ptr + page * stride_kp
    # A page is structure-of-arrays: BLOCK*D contiguous e4m3 bytes, then BLOCK
    # fp32 scales. (An AoS 132-byte row is the wrong reading of this cache; the
    # reference slices [:, :8192] and [:, 8192:] for exactly that reason.)
    acc = tl.zeros((BLOCK, H), dtype=tl.float32)
    for c in tl.static_range(0, D // DC):
        dsub = tl.arange(0, DC)
        off = c * DC + dsub
        kb = tl.load(base + rows[:, None] * D + off[None, :])
        k = _e4m3_to_f32(kb.to(tl.int32))
        qc = _e4m3_to_f32(tl.load(q_ptr + b * stride_qb + hd[:, None] * stride_qh + off[None, :]).to(tl.int32))
        # fp32 operands are the fast path on sm70/Triton: fp16 lands within
        # noise of this and bf16 is 10x slower (docs/runs/2026-11-17-indexer-kernel-first-run.md).
        acc += tl.dot(k, tl.trans(qc), input_precision="ieee", out_dtype=tl.float32)

    sp = (base + BLOCK * D + rows * 4).to(tl.pointer_type(tl.int32))
    scale = tl.load(sp).to(tl.float32, bitcast=True)

    acc = acc.to(tl.bfloat16).to(tl.float32)                    # reference's bmm
    acc = tl.maximum(acc, 0.0)
    wt = tl.load(w_ptr + b * stride_wb + hd)
    logit = tl.sum(acc * wt[None, :], axis=1) * scale

    col = p * BLOCK + rows
    seq = tl.load(seq_ptr + b)
    keep = (col < seq) & (col < S_MAX)
    tl.store(out_ptr + b * stride_ob + col, tl.where(keep, logit, 0.0), mask=col < S_MAX)


def fused_paged_mqa_logits(q_fp8, kvcache_fp8, weight, seq_lens, page_table,
                           deep_gemm_metadata, max_seq_len, clean_logits=False,
                           num_warps=8, dc=32):
    """The signature and semantics of `fp8_paged_mqa_logits` / its torch twin."""
    assert clean_logits is False, "the reference asserts this too"
    b, _, h, d = q_fp8.shape
    assert d == 128, f"head_dim {d} != 128"
    assert kvcache_fp8.shape[1] == 64, f"block {kvcache_fp8.shape[1]} != 64"
    q = q_fp8.view(torch.uint8)
    kv = kvcache_fp8.view(torch.uint8)
    pages = page_table.shape[1]
    out = torch.zeros((b, max_seq_len), dtype=torch.float32, device=q.device)
    paged_mqa_logits_kernel[(b, pages)](
        q, kv, weight.float(), seq_lens.to(torch.int32), page_table.to(torch.int32), out,
        q.stride(0), q.stride(2),
        kv.stride(0),
        weight.stride(0),
        page_table.stride(0),
        out.stride(0),
        S_MAX=max_seq_len, H=h, D=d, BLOCK=64, DC=dc,
        num_warps=num_warps,
    )
    return out
