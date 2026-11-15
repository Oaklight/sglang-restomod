"""TileLang SM70 GDN chunk kernels: the chunked output and the state pass.

Vendored from `sm70-kernels` at d75987a (merged as #13), files
`sm70_kernels/gdn_chunk_o/_chunk_o.py` and `sm70_kernels/gdn_chunk_h/_chunk_h.py`.
The kernel bodies are unchanged; this module only merges them and adds the
cached getters below.

  chunk_o: o = (q @ h^T) * exp(g) + (A * exp(g_i - g_j)) @ v, causal over BT.
  chunk_h: v_new_t = u_t - w_t @ h_{t-1}^T
           h_t     = exp(g_last_t) * h_{t-1} + v_new_t_scaled^T @ k_t

Import is not cheap: this module loads `tilelang` at import. The component
imports it lazily, on the first call that passes the gate.
"""

from __future__ import annotations

from functools import lru_cache

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


@lru_cache(maxsize=8)
def chunk_o(B, T_len, H, Hg, K, V):
    """Compiled chunk_o kernel for one shape; ``kernel(q, k, v, h, g) -> o``."""
    return chunk_o_kernel(B, T_len, H, Hg, K, V, 64, 64, 128)


@lru_cache(maxsize=8)
def chunk_h(B, T_len, H, Hg, K, V):
    """Compiled chunk_h kernel for one shape; ``kernel(k, w, u, g, s0) -> (h, v_new)``."""
    return chunk_h_kernel(B, T_len, H, Hg, K, V, 64, 32, 128)
