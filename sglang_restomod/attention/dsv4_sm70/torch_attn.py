"""Stage 1: DSV4 paged attention in torch, for a platform without flash_mla.

Two entry points, matching the two terminal calls the pin makes:

* :func:`paged_mla` stands in for ``flash_mla_with_kvcache`` -- decode, and any
  forward the sparse-prefill branch declines;
* :func:`forward_prefill_dense` stands in for the bound method
  ``_forward_prefill_sparse``.

Both are correctness references. They gather and dequantize the paged cache and
run an explicit softmax; nothing here is written for speed, and stage 2 is where
that changes.

**Not implemented yet.** The signatures and the KV byte layout are being pinned
down from the pin; until they are, both raise rather than guess, because a
reference that is subtly wrong is worse than one that is absent -- it would
produce logits that look right and a top-k set that is not.
"""

from __future__ import annotations

STAGE1_PROFILE = "tp1|no-spec|no-cuda-graph|batch-decode|chunked-prefill"


def paged_mla(*_args, **_kwargs):
    raise NotImplementedError(
        "attention_dsv4_sm70 stage 1: the torch paged decode is not implemented yet. "
        f"Profile: {STAGE1_PROFILE}."
    )


def forward_prefill_dense(*_args, **_kwargs):
    raise NotImplementedError(
        "attention_dsv4_sm70 stage 1: the torch dense prefill is not implemented yet. "
        f"Profile: {STAGE1_PROFILE}."
    )
