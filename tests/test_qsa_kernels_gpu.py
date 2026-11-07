"""Numeric checks for the SM70 QSA kernel boundaries.

Each test drives the same entry point the backend uses, with the layout the
backend passes at runtime, and compares it against a float32 recomputation of
the same selected-key attention. The reference is deliberately explicit — no
SDPA — so the masking rules (causal bound, -1 padding, out-of-range indices)
are visible in the test.

Covered:

- sparse decode (`sm70_cuda_qsa_decode`) with E5M2 KV, groups 3 and 6, and with
  FP16 KV (the receipted path);
- sparse chunk-prefill (`sm70_cuda_qsa_prefill`) with E5M2 KV and the causal
  visibility rule;
- dense paged decode (`grouped_decode_forward`) at page size 16 with E5M2 KV,
  on the CUDA partial (normal and `serial` variants) and with the CUDA partial
  disabled so the TileLang path runs;
- the indexer scoring binding, which has no caller yet (#234), so its
  semantics are checked here rather than discovered later.

Run on an SM70 host with the GPU-marked suite:
``scripts/test_gpu.sh tests/test_qsa_kernels_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch

try:  # The component calls Logger.info_once, which SGLang installs on logging.
    import sglang  # noqa: F401
except ModuleNotFoundError:  # a CPU-only checkout without the pin
    sglang = None

pytestmark = pytest.mark.gpu

DIM = 256
INDEXER_DIM = 128
INDEXER_HEADS = 4
INDEXER_PAGE = 4


def _require_sm70() -> torch.device:
    if sglang is None:
        pytest.skip("the pinned sglang is required; point PYTHONPATH at pin-site")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 (capability 7.0) is required")
    return torch.device("cuda")


def _float_cache(cache: torch.Tensor) -> torch.Tensor:
    """Dequantize K/V [.., 1, D] to float32 [.., D] (exact for E5M2 -> fp16)."""
    return cache.to(torch.float16).float().squeeze(-2)


def _attend(q_row: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, scale: float):
    scores = q_row.float() @ keys.float().T * scale
    probs = torch.softmax(scores, dim=-1)
    return probs @ values.float()


def _slot_table(requests: int, stride: int, seed: int) -> torch.Tensor:
    """A per-request permutation so a wrong request or position mapping shows."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    table = torch.empty((requests, stride), dtype=torch.int32, device="cuda")
    for request in range(requests):
        table[request] = (
            torch.randperm(stride, generator=generator, device="cuda") + request * stride
        ).to(torch.int32)
    return table


def _selected(seq_lens, topk: int, seed: int) -> torch.Tensor:
    """Selected logical positions, with -1 padding and out-of-range entries."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    indices = torch.full((len(seq_lens), topk), 999, dtype=torch.int32, device="cuda")
    for row, length in enumerate(seq_lens):
        count = min(int(length), topk)
        indices[row, :count] = torch.randint(
            0, length, (count,), generator=generator, device="cuda"
        ).to(torch.int32)
        if count >= 4:
            indices[row, count - 2] = -1  # padding
            indices[row, count - 3] = length + 7  # beyond the sequence
    return indices


def _sparse_reference(
    q, k_cache, v_cache, req_to_token, req_indices, indices, seq_lens, scale
):
    keys = _float_cache(k_cache)
    values = _float_cache(v_cache)
    outputs = []
    for row in range(q.shape[0]):
        length = int(seq_lens[row])
        scanned = min(length, indices.shape[1])
        logical = indices[row, :scanned].long()
        valid = (logical >= 0) & (logical < length)
        slots = req_to_token[int(req_indices[row]), logical.clamp(min=0)].long()
        outputs.append(
            _attend(q[row], keys[slots][valid], values[slots][valid], scale)
        )
    return torch.stack(outputs)


def _sparse_case(group: int, dtype, seq_lens, topk: int, seed: int):
    device = _require_sm70()
    stride = 256
    requests = len(seq_lens)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    req_to_token = _slot_table(requests, stride, seed)
    k_cache = torch.randn(
        (requests * stride, 1, DIM), generator=generator, device=device
    ).to(dtype)
    v_cache = torch.randn(
        (requests * stride, 1, DIM), generator=generator, device=device
    ).to(dtype)
    q = torch.randn((requests, group, DIM), generator=generator, device=device).to(
        torch.float16
    )
    indices = _selected(seq_lens, topk, seed + 1)
    return req_to_token, k_cache, v_cache, q, indices


@pytest.mark.parametrize("group", [6, 3])
def test_sparse_decode_e5m2_matches_reference(group: int):
    """E5M2 KV, decode: the branch that has never run (page-16 dense aside)."""
    from sglang_restomod.attention.v100.tilelang._decode_cuda import (
        sm70_cuda_qsa_decode,
    )

    scale = DIM**-0.5
    seq_lens = torch.tensor([200, 40], dtype=torch.int32, device="cuda")
    req_to_token, k_cache, v_cache, q, indices = _sparse_case(
        group, torch.float8_e5m2, [200, 40], 64, seed=10
    )
    output = sm70_cuda_qsa_decode(
        q, k_cache, v_cache, req_to_token,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        indices, seq_lens, scale,
    )
    reference = _sparse_reference(
        q, k_cache, v_cache, req_to_token,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        indices, seq_lens, scale,
    )
    diff = (output.float() - reference).abs().max().item()
    print(f"sparse decode e5m2 group={group}: max abs diff {diff:.2e}")
    assert diff < 3e-3


def test_sparse_decode_fp16_matches_reference():
    """FP16 KV, decode: the layout the receipted Flash-Next runs use."""
    from sglang_restomod.attention.v100.tilelang._decode_cuda import (
        sm70_cuda_qsa_decode,
    )

    scale = DIM**-0.5
    seq_lens = torch.tensor([128, 64], dtype=torch.int32, device="cuda")
    req_to_token, k_cache, v_cache, q, indices = _sparse_case(
        6, torch.float16, [128, 64], 64, seed=20
    )
    req_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    output = sm70_cuda_qsa_decode(
        q, k_cache, v_cache, req_to_token, req_indices, indices, seq_lens, scale
    )
    reference = _sparse_reference(
        q, k_cache, v_cache, req_to_token, req_indices, indices, seq_lens, scale
    )
    diff = (output.float() - reference).abs().max().item()
    print(f"sparse decode fp16 group=6: max abs diff {diff:.2e}")
    assert diff < 3e-3


def test_sparse_prefill_e5m2_matches_reference():
    """E5M2 chunk-prefill with the causal visibility rule (never run)."""
    from sglang_restomod.attention.v100.tilelang._decode_cuda import (
        sm70_cuda_qsa_prefill,
    )

    device = _require_sm70()
    scale = DIM**-0.5
    tokens, prefix, topk = 8, 32, 16
    seq_len = prefix + tokens
    generator = torch.Generator(device=device)
    generator.manual_seed(30)
    req_to_token = _slot_table(1, 256, seed=30)
    k_cache = torch.randn((256, 1, DIM), generator=generator, device=device).to(
        torch.float8_e5m2
    )
    v_cache = torch.randn((256, 1, DIM), generator=generator, device=device).to(
        torch.float8_e5m2
    )
    q = torch.randn((tokens, 6, DIM), generator=generator, device=device).to(torch.float16)
    # Candidates inside the causal bound for every row, plus a few that are not.
    indices = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    for row in range(tokens):
        visible = prefix + row + 1
        indices[row] = torch.randint(
            0, visible, (topk,), generator=generator, device=device
        ).to(torch.int32)
        indices[row, -1] = visible - 1  # exactly at the bound
        indices[row, -2] = visible + 5  # one past it
        indices[row, -3] = -1
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    req_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    output = sm70_cuda_qsa_prefill(
        q, k_cache, v_cache, req_to_token, req_indices, indices, seq_lens, scale
    )

    keys = _float_cache(k_cache)
    values = _float_cache(v_cache)
    expected = []
    for row in range(tokens):
        visible = prefix + row + 1
        scanned = min(topk, visible)
        logical = indices[row, :scanned].long()
        valid = (logical >= 0) & (logical < visible)
        slots = req_to_token[0, logical.clamp(min=0)].long()
        expected.append(_attend(q[row], keys[slots][valid], values[slots][valid], scale))
    reference = torch.stack(expected)
    diff = (output.float() - reference).abs().max().item()
    print(f"sparse prefill e5m2: max abs diff {diff:.2e} over {tokens} rows")
    assert diff < 3e-3


def _dense_case(seq_lens, page_size: int, pages: int, seed: int):
    device = _require_sm70()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    k_cache = torch.randn(
        (pages, page_size, 1, DIM), generator=generator, device=device
    ).to(torch.float8_e5m2)
    v_cache = torch.randn(
        (pages, page_size, 1, DIM), generator=generator, device=device
    ).to(torch.float8_e5m2)
    page_table = torch.empty((len(seq_lens), pages), dtype=torch.int32, device=device)
    for row in range(len(seq_lens)):
        page_table[row] = torch.randperm(pages, generator=generator, device=device).to(
            torch.int32
        )
    q = torch.randn((len(seq_lens), 6, DIM), generator=generator, device=device).to(
        torch.float16
    )
    return q, k_cache, v_cache, page_table


def _dense_reference(q, k_cache, v_cache, page_table, seq_lens, scale):
    keys = _float_cache(k_cache).reshape(-1, DIM)
    values = _float_cache(v_cache).reshape(-1, DIM)
    page_size = k_cache.shape[1]
    offsets = torch.arange(page_size, device=q.device)
    outputs = []
    for row in range(q.shape[0]):
        length = int(seq_lens[row])
        pages = page_table[row, : (length + page_size - 1) // page_size].long()
        slots = (pages[:, None] * page_size + offsets[None, :]).reshape(-1)[:length]
        outputs.append(_attend(q[row], keys[slots], values[slots], scale))
    return torch.stack(outputs)


@pytest.mark.parametrize("variant", ["occ2", "serial"])
def test_dense_paged_decode_e5m2_cuda_matches_reference(variant, monkeypatch):
    """Page-16 E5M2 dense decode through the CUDA partial (never run)."""
    from sglang_restomod.attention.v100.tilelang import grouped_decode_forward

    scale = DIM**-0.5
    page_size, pages = 16, 32
    seq_lens = torch.tensor([200, 480], dtype=torch.int32, device="cuda")
    q, k_cache, v_cache, page_table = _dense_case([200, 480], page_size, pages, seed=40)
    monkeypatch.setenv("SGLANG_V100_DECODE_CUDA", "1")
    monkeypatch.setenv("SGLANG_SM70_DECODE_VARIANT", variant)
    output = grouped_decode_forward(
        q, k_cache, v_cache, page_table, seq_lens, softmax_scale=scale
    )
    reference = _dense_reference(q, k_cache, v_cache, page_table, seq_lens, scale)
    diff = (output.float() - reference).abs().max().item()
    print(f"dense decode e5m2 cuda/{variant}: max abs diff {diff:.2e}")
    assert diff < 3e-3


def test_dense_paged_decode_e5m2_tilelang_matches_reference(monkeypatch):
    """The same case with the CUDA partial disabled (the env's other setting)."""
    from sglang_restomod.attention.v100.tilelang import grouped_decode_forward

    scale = DIM**-0.5
    page_size, pages = 16, 32
    seq_lens = torch.tensor([200, 480], dtype=torch.int32, device="cuda")
    q, k_cache, v_cache, page_table = _dense_case([200, 480], page_size, pages, seed=50)
    monkeypatch.setenv("SGLANG_V100_DECODE_CUDA", "0")
    output = grouped_decode_forward(
        q, k_cache, v_cache, page_table, seq_lens, softmax_scale=scale
    )
    reference = _dense_reference(q, k_cache, v_cache, page_table, seq_lens, scale)
    diff = (output.float() - reference).abs().max().item()
    print(f"dense decode e5m2 tilelang: max abs diff {diff:.2e}")
    assert diff < 3e-3


def test_indexer_decode_fp16_matches_reference():
    """The indexer binding has no caller (#234); check its scoring rule here."""
    from sglang_restomod.attention.v100.tilelang._decode_cuda import (
        sm70_cuda_qsa_indexer_decode,
    )

    device = _require_sm70()
    generator = torch.Generator(device=device)
    generator.manual_seed(60)
    batch, max_model_len, pages, page_size = 2, 64, 32, INDEXER_PAGE
    context_lens = torch.tensor([64, 40], dtype=torch.int32, device=device)
    score_scale = 0.5
    q = torch.randn(
        (batch, INDEXER_HEADS, INDEXER_DIM), generator=generator, device=device
    ).to(torch.float16)
    k_cache = torch.randn(
        (pages, INDEXER_HEADS, 1, INDEXER_DIM), generator=generator, device=device
    ).to(torch.float16)
    page_table = torch.empty((batch, pages), dtype=torch.int32, device=device)
    for row in range(batch):
        page_table[row] = torch.randperm(pages, generator=generator, device=device).to(
            torch.int32
        )
    logits = sm70_cuda_qsa_indexer_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )
    keys = k_cache.float().squeeze(2)  # [pages, page_size, D]
    expected = torch.zeros((batch, max_model_len), dtype=torch.float32, device=device)
    for row in range(batch):
        for position in range(int(context_lens[row])):
            page = int(page_table[row, position // page_size])
            key = keys[page, position % page_size]
            dots = (q[row].float() * key).sum(-1)
            expected[row, position] = torch.clamp(dots, min=0).sum() / score_scale
    # Positions past each row's context are never written by the kernel.
    diffs = []
    for row in range(batch):
        length = int(context_lens[row])
        diffs.append(
            (logits[row, :length] - expected[row, :length]).abs().max().item()
        )
    diff = max(diffs)
    print(
        f"indexer decode fp16: max abs diff {diff:.2e} over "
        f"{int(context_lens.sum())} positions"
    )
    assert diff < 3e-3
