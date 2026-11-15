"""TileLang SM70 GDN chunk_h: sequential state pass over BT=64 chunks.

v_new_t = u_t - w_t @ h_{t-1}^T
h_t     = exp(g_last_t) * h_{t-1} + v_new_t_scaled^T @ k_t
h[t] is stored before the update (the state entering chunk t), matching the
Triton reference; the final state is written back to S0 in place.
State is carried in an fp32 fragment and cast to fp16 only as a gemm operand.

Domain: h_t = exp(g_last) * h_{t-1} + v_new^T k is not contractive for
arbitrary inputs. On unnormalized random w/u, this kernel and the Triton
reference produce identical failure counts (h nan=11,795,324, inf=1,160,
absmax=65,504), so the limit is the recurrence, not the port. The model keeps
it in range with l2-normalized keys, w scaled by K^-0.5, u scaled by 0.5, and a
chunk-local decay cumsum.
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


@tilelang.jit(out_idx=[5, 6], pass_configs=pass_configs)
def chunk_h_kernel(B, T_len, H, Hg, K, V, BT=64, BV=32, threads=128):
    NT = T_len // BT
    assert T_len % BT == 0 and V % BV == 0
    G = H // Hg

    @T.prim_func
    def main(
        Kt: T.Tensor([B, T_len, Hg, K], T.float16),
        W: T.Tensor([B, T_len, H, K], T.float16),
        U: T.Tensor([B, T_len, H, V], T.float16),
        Gt: T.Tensor([B, T_len, H], T.float32),
        S0: T.Tensor([B, H, V, K], T.float16),
        Hout: T.Tensor([B, NT, H, V, K], T.float16),
        Vnew: T.Tensor([B, T_len, H, V], T.float16),
    ):
        with T.Kernel(V // BV, B * H, threads=threads) as (bx, bz):
            b = bz // H
            h = bz % H
            kh = h // G
            v0 = bx * BV

            Ws = T.alloc_shared([BT, K], T.float16)
            Ks = T.alloc_shared([BT, K], T.float16)
            Vs = T.alloc_shared([BT, BV], T.float16)
            Hf = T.alloc_shared([BV, K], T.float16)
            VfT = T.alloc_shared([BV, BT], T.float16)
            acc_h = T.alloc_fragment([BV, K], T.float32)
            acc_v = T.alloc_fragment([BT, BV], T.float32)

            for i, j in T.Parallel(BV, K):
                acc_h[i, j] = S0[b, h, v0 + i, j]

            for it in T.serial(NT):
                t0 = it * BT
                # state entering this chunk, before the update
                for i, j in T.Parallel(BV, K):
                    Hout[b, it, h, v0 + i, j] = T.cast(acc_h[i, j], T.float16)
                for i, j in T.Parallel(BV, K):
                    Hf[i, j] = T.cast(acc_h[i, j], T.float16)
                for i, j in T.Parallel(BT, K):
                    Ws[i, j] = W[b, t0 + i, h, j]
                for i, j in T.Parallel(BT, K):
                    Ks[i, j] = Kt[b, t0 + i, kh, j]
                for i, j in T.Parallel(BT, BV):
                    Vs[i, j] = U[b, t0 + i, h, v0 + j]

                T.clear(acc_v)
                T.gemm(Ws, Hf, acc_v, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BT, BV):
                    acc_v[i, j] = Vs[i, j] - acc_v[i, j]
                for i, j in T.Parallel(BT, BV):
                    Vnew[b, t0 + i, h, v0 + j] = T.cast(acc_v[i, j], T.float16)

                glast = Gt[b, t0 + BT - 1, h]
                for i, j in T.Parallel(BT, BV):
                    VfT[j, i] = T.cast(acc_v[i, j] * T.exp(glast - Gt[b, t0 + i, h]), T.float16)
                for i, j in T.Parallel(BV, K):
                    acc_h[i, j] *= T.exp(glast)
                T.gemm(VfT, Ks, acc_h, policy=T.GemmWarpPolicy.Square)

            for i, j in T.Parallel(BV, K):
                S0[b, h, v0 + i, j] = T.cast(acc_h[i, j], T.float16)

    return main
