"""End-to-end check for the SM70 block-FP8 MoE method (GPU).

Exercises ``SM70Fp8MoEMethod`` against a dequantized dense reference, with and
without expert-parallel routing (-1 drop ids):

    python -m sglang_restomod.quant.sm70_fp8_moe.check
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

EXPERT = 128


def _quant_block(w):
    """[N, K] fp16 -> block-128 e4m3 weights and scales."""
    n, k = w.shape
    blocks = w.float().reshape(n // EXPERT, EXPERT, k // EXPERT, EXPERT)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-6) / 448.0
    q = (blocks / amax[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(n, k), amax


def _dequant(w, s):
    n, k = w.shape
    return (
        w.float().reshape(n // EXPERT, EXPERT, k // EXPERT, EXPERT) * s[:, None, :, None]
    ).reshape(n, k)


def _reference(x, ids, weights, w13, s13, w2, s2):
    """Dense reference over all slots, dropping negative ids."""
    hidden = x.shape[1]
    inter = w13.shape[-2] // 2
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[token, slot])
            if e < 0:
                continue
            xe = x[token].float()
            y13 = xe @ _dequant(w13[e], s13[e]).T
            y = torch.nn.functional.silu(y13[:inter]) * y13[inter:]
            out[token] += float(weights[token, slot]) * (y @ _dequant(w2[e], s2[e]).T)
    return out


def _build_layer(experts, hidden, inter, seed=0):
    torch.manual_seed(seed)
    quant13, scales13, quant2, scales2 = [], [], [], []
    for _ in range(experts):
        q13, s13 = _quant_block((torch.randn(2 * inter, hidden, device="cuda") * 0.2).half())
        q2, s2 = _quant_block((torch.randn(hidden, inter, device="cuda") * 0.2).half())
        quant13.append(q13)
        scales13.append(s13)
        quant2.append(q2)
        scales2.append(s2)
    w13, s13 = torch.stack(quant13), torch.stack(scales13)
    w2, s2 = torch.stack(quant2), torch.stack(scales2)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(s13, requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(s2, requires_grad=False)
    layer.should_fuse_routed_scaling_factor_in_topk = False
    return layer, w13, s13, w2, s2


def _method(layer, experts, local_experts, top_k):
    from .method import SM70Fp8MoEMethod

    method = SM70Fp8MoEMethod(
        SimpleNamespace(
            weight_block_size=[EXPERT, EXPERT],
            use_mxfp8=False,
            is_fp4_experts=False,
            dequant_fp4_to_fp8=False,
        )
    )
    method.moe_runner_config = SimpleNamespace(
        activation="silu",
        is_gated=True,
        top_k=top_k,
        num_experts=experts,
        num_local_experts=local_experts,
        routed_scaling_factor=1.0,
        swiglu_limit=None,
        gemm1_clamp_limit=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        apply_router_weight_on_input=False,
        no_combine=False,
        wide_output_scale=1.0,
        gate_up_input_scale=1.0,
    )
    method.process_weights_after_loading(layer)
    return method


def _dispatch(x, ids, weights):
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=None),
    )


def run(expert_parallel: bool) -> float:
    """Relative error of one forward, optionally with EP drop ids."""
    experts, hidden, inter, top_k = 4, 128, 128, 3
    local = 2 if expert_parallel else experts
    tokens = 2 if expert_parallel else 8
    layer, w13, s13, w2, s2 = _build_layer(local, hidden, inter)
    method = _method(layer, experts, local, top_k)

    torch.manual_seed(7)
    x = (torch.randn(tokens, hidden, device="cuda") * 0.5).half()
    ids = torch.randint(0, local, (tokens, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, top_k, device="cuda", dtype=torch.float32)
    if expert_parallel:
        ids[0, 0] = -1
        ids[1, -1] = -1

    out = method.apply(layer, _dispatch(x, ids, weights)).hidden_states
    ref = _reference(x, ids, weights, w13, s13, w2, s2)
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def run_empty() -> int:
    """All routes dropped: the output must stay zero."""
    experts, hidden, inter, top_k = 2, 128, 128, 2
    layer, _, _, _, _ = _build_layer(experts, hidden, inter)
    method = _method(layer, experts, experts, top_k)
    x = torch.zeros(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.full((3, top_k), -1, device="cuda", dtype=torch.int32)
    out = method.apply(layer, _dispatch(x, ids, torch.rand(3, top_k, device="cuda")))
    return int(torch.count_nonzero(out.hidden_states))


def main() -> None:
    dense = run(False)
    ep = run(True)
    empty = run_empty()
    print(f"dense rel={dense:.5f} ep rel={ep:.5f} empty_nonzero={empty}")
    assert dense < 5e-2, "dense FP8 MoE does not match the reference"
    assert ep < 5e-2, "EP FP8 MoE does not match the reference"
    assert empty == 0, "dropped routes must not contribute"


if __name__ == "__main__":
    main()
