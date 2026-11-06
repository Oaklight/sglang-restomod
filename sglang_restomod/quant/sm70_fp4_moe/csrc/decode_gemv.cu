// Decode-shaped MoE GEMV for the SM70 Marlin NVFP4 expert layout.
//
// The Marlin kernel runs a 64-row M tile whatever M is. At decode M is 4-8, so
// that tile carries ~1.4 real rows per expert and the call spends 106 us on
// 18 MB of weights -- 16% of the card's bandwidth (sglang-restomod#201).
//
// This kernel drops the M tile: a CTA owns one 64-column repack tile and one
// route tile (up to 8 rows), streams the weights once, and keeps the
// accumulators in registers. A row pair whose slots are all sentinels does no
// arithmetic at all, so the 8-row padding costs nothing when a tile carries one
// real row -- which is what the decode router produces.
//
// Layout (notes/nvfp4-marlin-layout-2026-11-05.md), valid when N/64 is odd:
//   word = (e*(K/16) + g)*(N*2) + n_tile*128 + k_local*8 + b
//   slot i of the word holds column 64*n_tile + 8*b + PERM[i]
// with PERM = {0,2,4,6,1,3,5,7} and b = (n%64)/8.
//
// Shape of the work: the weight stream is 18.4 MB and runs at 994 GB/s
// (18.5 us) on its own, so everything above that is arithmetic. The inner loop
// therefore keeps pre-converted, pre-scaled half2 weights in shared memory and
// issues one half2 FMA per column pair per row, accumulating 128 terms in
// and issues one half2 FMA per column pair per row. The accumulator is fp32:
// a 128-term fp16 sum overflows on the real scale range (measured NaN at
// kSlabK = 128), so the half2 registers are promoted before each accumulate.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTileN = 64;       // columns per repack tile
constexpr int kTileWords = 128;  // 16 k rows x 8 eight-column blocks
constexpr int kBlockM = 8;       // route tile rows
constexpr int kGroups = 2;      // k groups per shared-memory slab
constexpr int kSlabK = 16 * kGroups;
constexpr int kColPairs = kTileN / 2;
constexpr int kThreads = kColPairs * (kBlockM / 2);  // 32 column pairs x 4 row pairs

// E2M1 code -> fp16 bits. Bit 3 is the sign.
__device__ __constant__ unsigned short kE2M1Bits[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
};

// Column n -> nibble slot inside its word (PERM_INV).
__device__ __constant__ int kSlot[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// The two columns carried by half2 slot i of a word: PERM[2i] and PERM[2i+1].
__device__ __constant__ int kColPairA[4] = {0, 4, 1, 5};
__device__ __constant__ int kColPairB[4] = {2, 6, 3, 7};

// encode_scales() reverses each group of four columns before storing, so a
// logical column n lives at stored index n ^ 3 when n % 4 is 1 or 2.
__device__ __host__ __forceinline__ int scale_index(int n) {
    const int r = n & 3;
    return (r == 1 || r == 2) ? (n ^ 3) : n;
}

// One Q0.7 byte -> the true block scale. The iterator rebuilds it as
// (byte << 8) >> 1, which is the scale times 128; encode_global_scale folds a
// matching factor of 128 into the global scale, so both carry it and both are
// divided here.
__device__ __forceinline__ float q07(unsigned char byte) {
    __half_raw raw;
    raw.x = static_cast<unsigned short>(byte) << 7;
    return __half2float(__half(raw)) * (1.0f / 128.0f);
}

__global__ void __launch_bounds__(kThreads)
decode_moe_gemv_kernel(const __half* __restrict__ a,
                       __half* __restrict__ c,
                       const int* __restrict__ b_q_weight,
                       const unsigned char* __restrict__ b_scales,
                       const float* __restrict__ global_scale,
                       const int* __restrict__ sorted_token_ids,
                       const int* __restrict__ expert_ids,
                       int num_rows, int top_k, int size_n, int size_k,
                       bool mul_topk_weights,
                       const float* __restrict__ topk_weights,
                       float* __restrict__ part, int k_split) {
    const int n_tile = blockIdx.x;
    const int route_tile = blockIdx.y;
    const int expert = expert_ids[route_tile];

    const int k16 = size_k / 16;
    const int n2 = size_n * 2;

    // The 4-way path uses one 16-K slab per warp, so the buffers are sized for
    // 4 warps x 16 K x 32 column pairs and re-used by the multi-row path.
    __shared__ half2 s_w[4 * 16 * kColPairs];  // pre-converted, scale folded in
    __shared__ half2 s_x[(kBlockM / 2) * kSlabK];  // 4 pairs x 32 K, or 4 warps x 16
    __shared__ __half s_s[4 * kTileN];
    __shared__ float2 s_red[4][32];
    __shared__ __half s_lut[16];

    const int tid = threadIdx.x;
    if (tid < 16) s_lut[tid] = __ushort_as_half(kE2M1Bits[tid]);

    // Gather this route tile's rows. A sentinel row is never written.
    int rows[kBlockM];
    int tokens[kBlockM];
#pragma unroll
    for (int s = 0; s < kBlockM; ++s) {
        const int row = sorted_token_ids[route_tile * kBlockM + s];
        rows[s] = row;
        tokens[s] = (row < num_rows) ? (row / top_k) : -1;
    }
    __syncthreads();

    // Column pair cp covers the two columns that share a half2 slot: slot j of
    // a word block holds column 8*b + PERM[j % 8].
    const int cp = tid & (kColPairs - 1);
    const int pair = tid >> 5;
    const int b = cp >> 2;
    const int colA = 8 * b + kColPairA[cp & 3];
    const int colB = 8 * b + kColPairB[cp & 3];
    const bool active = (tokens[2 * pair] >= 0) || (tokens[2 * pair + 1] >= 0);
    const int lane = tid & 31;

    const int w_base = expert * k16 * n2 + n_tile * kTileWords;
    const int s_base = expert * k16 * size_n + n_tile * kTileN;

    float2 acc[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};

    const int groups_per_split = k16 / k_split;
    const int group_begin = blockIdx.z * groups_per_split;
    const int group_end = group_begin + groups_per_split;

    // How many rows this tile really carries. At M=4/top_k=10 it is one, and
    // the multi-row path would leave three warps idle: ncu puts the kernel at
    // 37.9% issue with 16.0M warp instructions, four times what the arithmetic
    // needs. When there is one row, all four warps split its K range instead.
    int n_real = 0, r_first = -1;
#pragma unroll
    for (int s = 0; s < kBlockM; ++s) {
        if (tokens[s] >= 0) {
            if (r_first < 0) r_first = s;
            ++n_real;
        }
    }
    const bool single = (n_real == 1) && (groups_per_split >= 4);

    if (single) {
        const int span = groups_per_split / 4;
        const int g_begin = group_begin + pair * span;
        const int tok = tokens[r_first];
        half2* sw = &s_w[pair * 16 * kColPairs];
        __half* ss = &s_s[pair * kTileN];
        half2* sx = &s_x[pair * 16];
        float2 acc = make_float2(0.f, 0.f);
        for (int g = g_begin; g < g_begin + span; ++g) {
            // 64 scales, 128 words, 16 activations: one group per warp.
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n_local = lane + 32 * j;
                ss[n_local] = __float2half(q07(
                    b_scales[s_base + g * size_n + scale_index(n_local)]));
            }
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int within = lane + 32 * j;
                const int word = b_q_weight[w_base + g * n2 + within];
                const int k_local = within >> 3;
                const int bb = within & 7;
                const __half* sc = &ss[8 * bb];
                half2* dst = &sw[k_local * kColPairs + bb * 4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __half v0 = __hmul(s_lut[(word >> (8 * i)) & 0xF], sc[kColPairA[i]]);
                    const __half v1 = __hmul(s_lut[(word >> (8 * i + 4)) & 0xF], sc[kColPairB[i]]);
                    dst[i] = __halves2half2(v0, v1);
                }
            }
            if (lane < 16) {
                sx[lane] = __half2half2((tok >= 0) ? a[tok * size_k + g * 16 + lane]
                                                   : __float2half(0.f));
            }
            __syncwarp();
#pragma unroll 4
            for (int kk = 0; kk < 16; ++kk) {
                const float2 wv = __half22float2(sw[kk * kColPairs + cp]);
                const float xv = __half2float(__low2half(sx[kk]));
                acc.x = __fmaf_rn(wv.x, xv, acc.x);
                acc.y = __fmaf_rn(wv.y, xv, acc.y);
            }
            __syncwarp();
        }
        s_red[pair][lane] = acc;
        __syncthreads();
        if (pair == 0) {
            const float2 sum = s_red[0][lane];
            float2 total = make_float2(sum.x + s_red[1][lane].x + s_red[2][lane].x + s_red[3][lane].x,
                                       sum.y + s_red[1][lane].y + s_red[2][lane].y + s_red[3][lane].y);
            const float gscale1 = global_scale[expert] * (1.0f / 128.0f);
            float* dst1 = part + (static_cast<size_t>(blockIdx.z) * num_rows * size_n);
            const int row = rows[r_first];
            if (row < num_rows) {
                dst1[static_cast<size_t>(row) * size_n + n_tile * kTileN + colA] = total.x * gscale1;
                dst1[static_cast<size_t>(row) * size_n + n_tile * kTileN + colB] = total.y * gscale1;
            }
        }
        return;
    }

    for (int slab = group_begin; slab < group_end; slab += kGroups) {
        // The slab's block scales, one per (group, column).
#pragma unroll
        for (int j = 0; j < kGroups * kTileN / kThreads; ++j) {
            const int idx = tid + j * kThreads;
            const int g_local = idx / kTileN;
            const int n_local = idx % kTileN;
            s_s[idx] = __float2half(q07(
                b_scales[s_base + (slab + g_local) * size_n + scale_index(n_local)]));
        }
        // The conversion below reads s_s across warp boundaries, so the fill
        // has to complete first.
        __syncthreads();
        // Convert one word per thread pass: eight nibbles at one k, eight
        // columns. Each lands at s_w[k][8*b + slot], already scaled.
#pragma unroll
        for (int j = 0; j < kGroups * kTileWords / kThreads; ++j) {
            const int idx = tid + j * kThreads;
            const int g_local = idx / kTileWords;
            const int within = idx % kTileWords;
            const int word = b_q_weight[w_base + (slab + g_local) * n2 + within];
            const int k_local = within >> 3;
            const int bb = within & 7;
            const __half* sc = &s_s[g_local * kTileN + 8 * bb];
            half2* dst = &s_w[(g_local * 16 + k_local) * kColPairs + bb * 4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __half v0 = __hmul(s_lut[(word >> (8 * i)) & 0xF], sc[kColPairA[i]]);
                const __half v1 = __hmul(s_lut[(word >> (8 * i + 4)) & 0xF], sc[kColPairB[i]]);
                dst[i] = __halves2half2(v0, v1);
            }
        }
        // Activations, staged as row pairs so the inner loop is a half2 read.
#pragma unroll
        for (int j = 0; j < (kBlockM / 2) * kSlabK / kThreads; ++j) {
            const int idx = tid + j * kThreads;
            const int p = idx / kSlabK;
            const int kk = idx % kSlabK;
            const int t0 = tokens[2 * p];
            const int t1 = tokens[2 * p + 1];
            const int k = slab * 16 + kk;
            s_x[idx] = __halves2half2(
                (t0 >= 0) ? a[t0 * size_k + k] : __float2half(0.f),
                (t1 >= 0) ? a[t1 * size_k + k] : __float2half(0.f));
        }
        __syncthreads();

        if (active) {
#pragma unroll 4
            for (int kk = 0; kk < kSlabK; ++kk) {
                const float2 wv = __half22float2(s_w[kk * kColPairs + cp]);
                const float2 xv = __half22float2(s_x[pair * kSlabK + kk]);
                acc[0].x = __fmaf_rn(wv.x, xv.x, acc[0].x);
                acc[0].y = __fmaf_rn(wv.y, xv.x, acc[0].y);
                acc[1].x = __fmaf_rn(wv.x, xv.y, acc[1].x);
                acc[1].y = __fmaf_rn(wv.y, xv.y, acc[1].y);
            }
        }
        __syncthreads();
    }

    const float gscale = global_scale[expert] * (1.0f / 128.0f);
    float* dst = part + (static_cast<size_t>(blockIdx.z) * num_rows * size_n);
    const int cols[2] = {colA, colB};
    const float accs[2][2] = {{acc[0].x * gscale, acc[0].y * gscale},
                              {acc[1].x * gscale, acc[1].y * gscale}};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = rows[2 * pair + i];
        if (row >= num_rows) continue;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            dst[static_cast<size_t>(row) * size_n + n_tile * kTileN + cols[q]] = accs[i][q];
        }
    }
}

// Sum the split-K partials, apply the routing weight, and emit fp16.
__global__ void reduce_split_k_kernel(const float* __restrict__ part,
                                      __half* __restrict__ c,
                                      const float* __restrict__ topk_weights,
                                      int k_split, int num_rows, int size_n,
                                      bool mul_topk_weights) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (n >= size_n) return;
    float sum = 0.f;
    for (int z = 0; z < k_split; ++z) {
        sum += part[(static_cast<size_t>(z) * num_rows + row) * size_n + n];
    }
    // `sorted_token_ids` carries route indices (`token * top_k + slot`), which
    // is why the activation row above is `row / top_k`. `topk_weights` is
    // indexed by the same route index -- see the reference kernel,
    // `marlin_template.h`, which loads `topk_weights_ptr[idx]` with
    // `idx = sh_block_sorted_ids[...]`. The server passes one weight per route.
    if (mul_topk_weights) sum *= topk_weights[row];
    c[static_cast<size_t>(row) * size_n + n] = __float2half(sum);
}

}  // namespace

torch::Tensor decode_moe_gemv(torch::Tensor a, torch::Tensor c,
                              torch::Tensor b_q_weight, torch::Tensor b_scales,
                              torch::Tensor global_scale,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor expert_ids,
                              torch::Tensor topk_weights,
                              int64_t num_rows, int64_t top_k,
                              int64_t size_n, int64_t size_k,
                              bool mul_topk_weights,
                              torch::Tensor workspace, int64_t k_split) {
    const int route_tiles = expert_ids.size(0);
    dim3 grid(size_n / kTileN, route_tiles, k_split);
    // Launch on the current stream: a bare <<<...>>> goes to the legacy
    // default stream, which CUDA graph capture does not record, so the
    // captured graph comes out empty and the server fails at capture time.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    decode_moe_gemv_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(a.data_ptr()),
        reinterpret_cast<__half*>(c.data_ptr()),
        b_q_weight.data_ptr<int>(),
        reinterpret_cast<const unsigned char*>(b_scales.data_ptr()),
        global_scale.data_ptr<float>(),
        sorted_token_ids.data_ptr<int>(),
        expert_ids.data_ptr<int>(),
        static_cast<int>(num_rows), static_cast<int>(top_k),
        static_cast<int>(size_n), static_cast<int>(size_k),
        mul_topk_weights, topk_weights.data_ptr<float>(),
        workspace.data_ptr<float>(), static_cast<int>(k_split));
    const int reduce_grid = (size_n + 255) / 256;
    reduce_split_k_kernel<<<dim3(reduce_grid, num_rows), 256, 0, stream>>>(
        workspace.data_ptr<float>(),
        reinterpret_cast<__half*>(c.data_ptr()),
        topk_weights.data_ptr<float>(),
        static_cast<int>(k_split), static_cast<int>(num_rows),
        static_cast<int>(size_n), mul_topk_weights);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_moe_gemv", &decode_moe_gemv,
          "decode-shaped NVFP4 MoE GEMV (SM70 Marlin layout)");
}
