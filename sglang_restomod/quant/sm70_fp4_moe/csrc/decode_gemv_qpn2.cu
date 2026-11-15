// Decode-shaped MoE GEMV over the QPN2 NVFP4 expert layout (SM70).
//
// Layout: docs/qpn2-layout.md. A 288-byte group is 32 lanes x 8 code bytes
// followed by 32 E4M3 scale bytes, so a warp's 32 lanes read the group with one
// coalesced 256-byte transaction and each lane already holds its own column's
// 16 codes. The nibble expansion below yields weight pairs in logical K order,
// so the activation operand is the plain contiguous 16-half slice and there is
// no transpose and no per-word gather.
//
// Routing matches the Marlin call: `sorted_token_ids` is the align kernel's
// route index (`token * top_k + slot`), one block of 8 slots per route at
// decode. A block with one real route takes the fast path; a block where two
// routes landed on the same expert takes the general path, which reuses the
// decoded words for every real row.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarps = 4;         // warps per CTA, each a K slice of one tile
constexpr int kThreads = 32 * kWarps;
constexpr int kGroupBytes = 288;  // 256 code bytes + 32 scale bytes
constexpr int kBlockM = 8;        // route slots per block, the align kernel's

__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ half2 fp8e4m3_to_half2(unsigned value) {
  const unsigned short bits = static_cast<unsigned short>(
      ((value & 0x80u) << 8) | ((value & 0x7fu) << 7));
  const half converted = __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

// spec section 3: multiply by the FP32 global scale, then round once to FP16
__device__ __forceinline__ half2 group_scale(unsigned char byte, float global) {
  const half raw = __low2half(fp8e4m3_to_half2(byte));
  const half scaled = __float2half_rn(__fmul_rn(__half2float(raw), global));
  return __halves2half2(scaled, scaled);
}

// 8 E2M1 codes in one 32-bit word -> 4 half2, pairs in logical K order.
__device__ __forceinline__ void expand_word(unsigned packed, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExpMant = 0x0e000e00u;
  unsigned v[4];
  v[0] = ((packed << 12) & kSign) | ((packed << 9) & kExpMant);
  v[1] = ((packed << 8) & kSign) | ((packed << 5) & kExpMant);
  v[2] = ((packed << 4) & kSign) | ((packed << 1) & kExpMant);
  v[3] = (packed & kSign) | ((packed >> 3) & kExpMant);
  const half2 bias = __float2half2_rn(16384.0f);  // undo the FP4 exponent bias
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = __hmul2(*reinterpret_cast<const half2*>(&v[i]), bias);
  }
}

template <bool kCompute>
__global__ void __launch_bounds__(kThreads)
qpn2_moe_gemv_kernel(const unsigned char* __restrict__ weights,
                     const float* __restrict__ global_scale,
                     const __half* __restrict__ a,
                     const int* __restrict__ sorted_token_ids,
                     const int* __restrict__ expert_ids,
                     const int* __restrict__ num_tokens_post_padded,
                     float* __restrict__ partial, int num_rows, int top_k,
                     int n, int k, int split_k) {
  const int route_tile = blockIdx.y;
  // The grid covers the worst-case expert_ids allocation; only the blocks the
  // align kernel filled are real.
  if (route_tile * kBlockM >= num_tokens_post_padded[0]) return;

  const int tile = blockIdx.x;
  const int split = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int groups = k >> 4;
  const int tiles = n >> 5;

  const int g0 = (split * groups) / split_k;
  const int g1 = ((split + 1) * groups) / split_k;
  const int gb = g0 + (warp * (g1 - g0)) / kWarps;
  const int ge = g0 + ((warp + 1) * (g1 - g0)) / kWarps;

  const int expert = expert_ids[route_tile];
  const float gs = global_scale[expert];
  const unsigned char* base =
      weights + (static_cast<size_t>(expert) * tiles + tile) * groups * kGroupBytes;
  const int col = (tile << 5) + qpn2_col_from_lane(lane);
  const int plane = split * kWarps + warp;
  float* dst = partial + static_cast<size_t>(plane) * num_rows * n;

  // Real routes in this block. At decode a block carries one; two tokens
  // routed to the same expert put two here.
  int routes[kBlockM];
  int tokens[kBlockM];
  int n_real = 0;
#pragma unroll
  for (int s = 0; s < kBlockM; ++s) {
    const int route = sorted_token_ids[route_tile * kBlockM + s];
    routes[s] = route;
    tokens[s] = (route < num_rows) ? route / top_k : -1;
    if (tokens[s] >= 0) ++n_real;
  }
  if (n_real == 0) return;

  if (n_real == 1) {
    // Fast path: four independent accumulator chains, one real row.
    int first = 0;
#pragma unroll
    for (int s = 0; s < kBlockM; ++s) {
      if (tokens[s] >= 0) first = s;
    }
    const __half* av = a + static_cast<size_t>(tokens[first]) * k;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int g = gb; g < ge; ++g) {
      const unsigned char* grp = base + static_cast<size_t>(g) * kGroupBytes;
      const uint2 q = __ldcs(reinterpret_cast<const uint2*>(grp + lane * 8));
      if constexpr (!kCompute) {
        const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(av + g * 16));
        const uint4 a1 = __ldg(reinterpret_cast<const uint4*>(av + g * 16 + 8));
        acc0 += static_cast<float>(q.x ^ q.y) + grp[256 + lane] +
                static_cast<float>((a0.x ^ a0.y) ^ (a0.z ^ a0.w)) +
                static_cast<float>((a1.x ^ a1.y) ^ (a1.z ^ a1.w));
        continue;
      }
      const half2 sh = group_scale(grp[256 + lane], gs);
      half2 w[8];
      expand_word(q.x, w);
      expand_word(q.y, w + 4);
      const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(av + g * 16));
      const uint4 a1 = __ldg(reinterpret_cast<const uint4*>(av + g * 16 + 8));
      const unsigned araw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 wf = __half22float2(__hmul2(w[i], sh));
        const float2 af = __half22float2(*reinterpret_cast<const half2*>(&araw[i]));
        if ((i & 1) == 0) {
          acc0 = fmaf(wf.x, af.x, acc0);
          acc1 = fmaf(wf.y, af.y, acc1);
        } else {
          acc2 = fmaf(wf.x, af.x, acc2);
          acc3 = fmaf(wf.y, af.y, acc3);
        }
      }
    }
    const float acc = (acc0 + acc1) + (acc2 + acc3);
    dst[static_cast<size_t>(routes[first]) * n + col] = acc;
    return;
  }

  // General path: the decoded words serve every real row in the block. Two
  // accumulators per row; this is the rare shape, so it trades ILP for reach.
  float acc[kBlockM][2];
#pragma unroll
  for (int s = 0; s < kBlockM; ++s) acc[s][0] = acc[s][1] = 0.f;
  for (int g = gb; g < ge; ++g) {
    const unsigned char* grp = base + static_cast<size_t>(g) * kGroupBytes;
    const uint2 q = __ldcs(reinterpret_cast<const uint2*>(grp + lane * 8));
    if constexpr (!kCompute) {
      acc[0][0] += static_cast<float>(q.x ^ q.y) + grp[256 + lane];
      continue;
    }
    const half2 sh = group_scale(grp[256 + lane], gs);
    half2 w[8];
    expand_word(q.x, w);
    expand_word(q.y, w + 4);
    half2 ws[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) ws[i] = __hmul2(w[i], sh);
#pragma unroll
    for (int s = 0; s < kBlockM; ++s) {
      if (tokens[s] < 0) continue;
      const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(a + static_cast<size_t>(tokens[s]) * k + g * 16));
      const uint4 a1 = __ldg(reinterpret_cast<const uint4*>(a + static_cast<size_t>(tokens[s]) * k + g * 16 + 8));
      const unsigned araw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 wf = __half22float2(ws[i]);
        const float2 af = __half22float2(*reinterpret_cast<const half2*>(&araw[i]));
        acc[s][0] = fmaf(wf.x, af.x, acc[s][0]);
        acc[s][1] = fmaf(wf.y, af.y, acc[s][1]);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kBlockM; ++s) {
    if (tokens[s] < 0) continue;
    dst[static_cast<size_t>(routes[s]) * n + col] = acc[s][0] + acc[s][1];
  }
}

// Sum the split-K partials, apply the routing weight, and emit fp16.
__global__ void reduce_split_k_kernel(const float* __restrict__ partial,
                                      __half* __restrict__ out,
                                      const float* __restrict__ topk_weights,
                                      int planes, int num_rows, int n,
                                      bool mul_topk_weights) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= n) return;
  float sum = 0.f;
  for (int z = 0; z < planes; ++z) {
    sum += partial[(static_cast<size_t>(z) * num_rows + row) * n + col];
  }
  if (mul_topk_weights) sum *= topk_weights[row];
  out[static_cast<size_t>(row) * n + col] = __float2half(sum);
}

}  // namespace

// Same argument order as the Marlin-layout decode GEMV minus `b_scales`:
// QPN2 keeps the E4M3 group scales inside the weight buffer.
torch::Tensor decode_moe_gemv_qpn2(torch::Tensor a, torch::Tensor c,
                                   torch::Tensor b_q_weight,
                                   torch::Tensor global_scale,
                                   torch::Tensor sorted_token_ids,
                                   torch::Tensor expert_ids,
                                   torch::Tensor topk_weights,
                                   int64_t num_rows, int64_t top_k,
                                   int64_t size_n, int64_t size_k,
                                   bool mul_topk_weights,
                                   torch::Tensor workspace, int64_t k_split,
                                   torch::Tensor num_tokens_post_padded) {
  TORCH_CHECK(b_q_weight.scalar_type() == torch::kUInt8, "QPN2 weights must be uint8");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16 && c.scalar_type() == torch::kFloat16,
              "QPN2 GEMV is fp16 in and out");
  TORCH_CHECK(size_n % 32 == 0 && size_k % 16 == 0, "bad QPN2 shape");
  const int route_tiles = static_cast<int>(expert_ids.size(0));
  const int planes = static_cast<int>(k_split) * kWarps;
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(planes) * num_rows * size_n,
              "QPN2 workspace too small");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(size_n / 32), static_cast<unsigned>(route_tiles),
                  static_cast<unsigned>(k_split));
  const unsigned char* w = b_q_weight.data_ptr<unsigned char>();
  const float* gs = global_scale.data_ptr<float>();
  const __half* av = reinterpret_cast<const __half*>(a.data_ptr());
  const int* sid = sorted_token_ids.data_ptr<int>();
  const int* eid = expert_ids.data_ptr<int>();
  const int* padded = num_tokens_post_padded.data_ptr<int>();
  float* part = workspace.data_ptr<float>();
  __half* out = reinterpret_cast<__half*>(c.data_ptr());
  qpn2_moe_gemv_kernel<true><<<grid, kThreads, 0, stream>>>(
      w, gs, av, sid, eid, padded, part, static_cast<int>(num_rows),
      static_cast<int>(top_k), static_cast<int>(size_n), static_cast<int>(size_k),
      static_cast<int>(k_split));
  reduce_split_k_kernel<<<dim3((size_n + 255) / 256, num_rows), 256, 0, stream>>>(
      part, out, topk_weights.data_ptr<float>(), planes,
      static_cast<int>(num_rows), static_cast<int>(size_n), mul_topk_weights);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

torch::Tensor decode_moe_gemv_qpn2_loads(torch::Tensor a, torch::Tensor c,
                                         torch::Tensor b_q_weight,
                                         torch::Tensor global_scale,
                                         torch::Tensor sorted_token_ids,
                                         torch::Tensor expert_ids,
                                         torch::Tensor topk_weights,
                                         int64_t num_rows, int64_t top_k,
                                         int64_t size_n, int64_t size_k,
                                         bool mul_topk_weights,
                                         torch::Tensor workspace, int64_t k_split,
                                         torch::Tensor num_tokens_post_padded) {
  const int route_tiles = static_cast<int>(expert_ids.size(0));
  const int planes = static_cast<int>(k_split) * kWarps;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(size_n / 32), static_cast<unsigned>(route_tiles),
                  static_cast<unsigned>(k_split));
  qpn2_moe_gemv_kernel<false><<<grid, kThreads, 0, stream>>>(
      b_q_weight.data_ptr<unsigned char>(), global_scale.data_ptr<float>(),
      reinterpret_cast<const __half*>(a.data_ptr()),
      sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
      num_tokens_post_padded.data_ptr<int>(), workspace.data_ptr<float>(),
      static_cast<int>(num_rows), static_cast<int>(top_k),
      static_cast<int>(size_n), static_cast<int>(size_k),
      static_cast<int>(k_split));
  reduce_split_k_kernel<<<dim3((size_n + 255) / 256, num_rows), 256, 0, stream>>>(
      workspace.data_ptr<float>(), reinterpret_cast<__half*>(c.data_ptr()),
      topk_weights.data_ptr<float>(), planes, static_cast<int>(num_rows),
      static_cast<int>(size_n), false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_moe_gemv_qpn2", &decode_moe_gemv_qpn2,
          "decode-shaped NVFP4 MoE GEMV (SM70 QPN2 layout)");
    m.def("decode_moe_gemv_qpn2_loads", &decode_moe_gemv_qpn2_loads,
          "QPN2 decode GEMV with the arithmetic removed (loads-only floor)");
}
