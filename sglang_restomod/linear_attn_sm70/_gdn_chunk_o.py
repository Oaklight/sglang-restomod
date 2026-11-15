"""TileLang SM70 GDN chunk_o: o = (q @ h^T) * exp(g) + (A * exp(g_i-g_j)) @ v.

Mirrors sglang fla chunk_o.py chunk_fwd_kernel_o. Shapes are per TP rank.
The gating is applied between the two gemms, matching the reference order.

Domain: this kernel does not recur, but it consumes the chunk_h state. That
recurrence is not contractive for arbitrary inputs: on unnormalized random w/u,
both the Triton reference and the TileLang chunk_h produce identical failure
counts (h nan=11,795,324, inf=1,160, absmax=65,504). Keep keys l2-normalized
and w/u scaled as the model does; a NaN here is the input regime, not the port.
"""
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False


@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def chunk_o_kernel(B, T_len, H, Hg, K, V, BT=64, BV=64, threads=128):
    NT = T_len // BT
    assert T_len % BT == 0 and V % BV == 0 and K % 16 == 0
    scale = K ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor([B, T_len, Hg, K], T.float16),
        Kt: T.Tensor([B, T_len, Hg, K], T.float16),
        Vt: T.Tensor([B, T_len, H, V], T.float16),
        Ht: T.Tensor([B * NT, H, V, K], T.float16),
        G: T.Tensor([B, T_len, H], T.float32),
        O: T.Tensor([B, T_len, H, V], T.float16),
    ):
        with T.Kernel(V // BV, NT, B * H, threads=threads) as (bx, by, bz):
            b = bz // H
            h = bz % H
            kh = h // (H // Hg)
            t0 = by * BT
            v0 = bx * BV

            Qs = T.alloc_shared([BT, K], T.float16)
            Ks = T.alloc_shared([BT, K], T.float16)
            Hs = T.alloc_shared([BV, K], T.float16)
            Vs = T.alloc_shared([BT, BV], T.float16)
            As = T.alloc_shared([BT, BT], T.float16)
            acc_o = T.alloc_fragment([BT, BV], T.float32)
            acc_A = T.alloc_fragment([BT, BT], T.float32)
            Ac = T.alloc_fragment([BT, BT], T.float16)

            for i, j in T.Parallel(BT, K):
                Qs[i, j] = Q[b, t0 + i, kh, j]
            for i, j in T.Parallel(BT, K):
                Ks[i, j] = Kt[b, t0 + i, kh, j]
            for i, j in T.Parallel(BV, K):
                Hs[i, j] = Ht[b * NT + by, h, v0 + i, j]
            for i, j in T.Parallel(BT, BV):
                Vs[i, j] = Vt[b, t0 + i, h, v0 + j]

            T.clear(acc_o)
            T.clear(acc_A)
            T.gemm(Qs, Hs, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.gemm(Qs, Ks, acc_A, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BV):
                acc_o[i, j] *= T.exp(G[b, t0 + i, h])
            for i, j in T.Parallel(BT, BT):
                d = G[b, t0 + i, h] - G[b, t0 + j, h]
                dd = T.if_then_else(d <= 0, d, -1.0e30)
                acc_A[i, j] = T.if_then_else(i >= j, acc_A[i, j] * T.exp(dd), T.cast(0, T.float32))
            for i, j in T.Parallel(BT, BT):
                As[i, j] = T.cast(acc_A[i, j], T.float16)
            T.copy(As, Ac)
            T.gemm(Ac, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BV):
                O[b, t0 + i, h, v0 + j] = T.cast(acc_o[i, j] * scale, T.float16)

    return main
