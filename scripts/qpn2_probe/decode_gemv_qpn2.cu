// Direct-register QPN2 decode GEMV -- the #224 P1 probe.
//
// The Marlin-layout decode GEMV (#201) reads one 8-byte word per lane per K16
// group from a permuted layout that costs 2.5x address arithmetic and lands at
// 234 GB/s. QPN2 stores the group as 32 contiguous lane words, so the reader is
// base + g*288 + lane*8 with no gather, and the nibble expansion yields weight
// pairs already in logical K order:
//
//   word 0 -> (k0,k1) (k2,k3) (k4,k5) (k6,k7)
//   word 1 -> (k8,k9) ... (k14,k15)
//
// so the activation operand is the plain contiguous 16-half slice a[g*16 ..].
// Layout: docs/qpn2-layout.md. One warp owns one 32-column tile; its lanes are
// the 32 stored lanes, whose columns are qpn2_col_from_lane(lane).
//
// Shape of the work at TP4: gate_up 40 route rows x 460,800 B = 18.4 MB of
// weights. 45 us -- the #224 gate -- is 409 GB/s. Split K four ways inside a
// CTA so the 18.4 MB stream has 4 warps x 1600 CTAs of load-level parallelism.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kGroupBytes = 288;  // 256 code bytes + 32 scale bytes

// Storage lane -> logical column inside the 32-column tile (spec section 1).
__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

// E4M3 byte -> exact FP16, both halves (spec section 3).
__device__ __forceinline__ half2 fp8e4m3_to_half2(unsigned value) {
  const unsigned short bits = static_cast<unsigned short>(
      ((value & 0x80u) << 8) | ((value & 0x7fu) << 7));
  const half converted = __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

// Multiply by the FP32 global scale, then round once to FP16.
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

// kCompute == false is the loads-only floor: same addresses, same traffic, no
// dequant and no dot product.
template <bool kCompute>
__global__ void __launch_bounds__(kThreads)
qpn2_gemv_kernel(const unsigned char* __restrict__ weights,
                 const float* __restrict__ global_scale,
                 const __half* __restrict__ activation,
                 const int* __restrict__ expert_ids,
                 const int* __restrict__ token_ids,
                 float* __restrict__ partial,
                 __half* __restrict__ out, int n, int k, int split_k) {
  const int tile = blockIdx.x;
  const int row = blockIdx.y;
  const int split = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int groups = k >> 4;
  const int tiles = n >> 5;

  // Even split of this CTA's K range over the four warps: 160 groups / 4 = 40.
  const int g0 = (split * groups) / split_k;
  const int g1 = ((split + 1) * groups) / split_k;
  const int gb = g0 + (warp * (g1 - g0)) / kWarps;
  const int ge = g0 + ((warp + 1) * (g1 - g0)) / kWarps;

  const int expert = expert_ids[row];
  const int token = token_ids[row];
  const float gs = global_scale[expert];
  const unsigned char* base =
      weights + (static_cast<size_t>(expert) * tiles + tile) * groups * kGroupBytes;
  const __half* av = activation + static_cast<size_t>(token) * k;
  const int col = (tile << 5) + qpn2_col_from_lane(lane);

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int g = gb; g < ge; ++g) {
    const unsigned char* grp = base + static_cast<size_t>(g) * kGroupBytes;
    const uint2 q = __ldcs(reinterpret_cast<const uint2*>(grp + lane * 8));
    if constexpr (!kCompute) {
      // Producer floor: every global load the real path issues (codes, scale,
      // both activation words), no dequant and no dot product.
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
      const float2 af =
          __half22float2(*reinterpret_cast<const half2*>(&araw[i]));
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
  // The four warps of a CTA cover disjoint K ranges of the same tile, so every
  // warp writes its own partial plane; the reduce kernel sums them. A direct
  // write here would let one warp overwrite the other three.
  const int plane = split * kWarps + warp;
  partial[(static_cast<size_t>(plane) * gridDim.y + row) * n + col] = acc;
}

__global__ void reduce_split_k_kernel(const float* __restrict__ partial,
                                      __half* __restrict__ out, int split_k,
                                      int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= n) return;
  float sum = 0.f;
  for (int z = 0; z < split_k; ++z) {
    sum += partial[(static_cast<size_t>(z) * gridDim.y + row) * n + col];
  }
  out[static_cast<size_t>(row) * n + col] = __float2half(sum);
}

}  // namespace

// weights:   uint8 [experts, tiles*groups*288], expert-contiguous
// gscale:    float32 [experts]
// activation:half  [tokens, k], plain logical K order
// expert_ids/token_ids: int32 [rows]
// out:       half [rows, n]
// workspace: float [(split_k * 4), rows, n] (one plane per CTA warp), reduced
//            into `out` by the second kernel
torch::Tensor decode_qpn2_gemv(torch::Tensor weights, torch::Tensor global_scale,
                               torch::Tensor activation, torch::Tensor expert_ids,
                               torch::Tensor token_ids, torch::Tensor out,
                               torch::Tensor workspace, int64_t split_k,
                               bool loads_only) {
  TORCH_CHECK(weights.scalar_type() == torch::kUInt8, "weights must be uint8");
  TORCH_CHECK(activation.scalar_type() == torch::kFloat16, "activation must be fp16");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
  const int rows = static_cast<int>(expert_ids.size(0));
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(activation.size(1));
  TORCH_CHECK(n % 32 == 0 && k % 16 == 0 && k % 8 == 0, "bad QPN2 shape");
  TORCH_CHECK(split_k >= 1 && split_k <= 32, "bad split_k");
  const int planes = static_cast<int>(split_k) * kWarps;
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(planes) * rows * n,
              "workspace too small");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(n / 32, rows, static_cast<unsigned>(split_k));
  const unsigned char* w = weights.data_ptr<unsigned char>();
  const float* gs = global_scale.data_ptr<float>();
  const __half* a = reinterpret_cast<const __half*>(activation.data_ptr());
  const int* eids = expert_ids.data_ptr<int>();
  const int* tids = token_ids.data_ptr<int>();
  float* part = workspace.data_ptr<float>();
  __half* o = reinterpret_cast<__half*>(out.data_ptr());
  if (loads_only) {
    qpn2_gemv_kernel<false><<<grid, kThreads, 0, stream>>>(
        w, gs, a, eids, tids, part, o, n, k, static_cast<int>(split_k));
  } else {
    qpn2_gemv_kernel<true><<<grid, kThreads, 0, stream>>>(
        w, gs, a, eids, tids, part, o, n, k, static_cast<int>(split_k));
  }
  reduce_split_k_kernel<<<dim3((n + 255) / 256, rows), 256, 0, stream>>>(
      part, o, planes, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
