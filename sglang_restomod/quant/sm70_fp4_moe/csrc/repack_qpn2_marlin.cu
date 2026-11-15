// Fused QPN2 -> Marlin repack for the prefill fallback.
//
// The prefill reader needs the Marlin layout, so a QPN2-resident model repacks
// one layer into a scratch buffer per prefill. The two-step form (torch unpack
// + the per-expert repack kernel) measured 70.5 ms/layer, 30x over the budget,
// because it is launch-bound. This is a single launch: one thread per output
// word reads the eight nibbles it needs directly from the QPN2 bytes.
//
// Output layout is marlin_v100's gptq_marlin_repack for 4-bit, no perm:
//   word index (k_tile, col) with col = cta_n_tile*cta_n_tiles*tile_words
//                                        + local_word*cta_n_tiles + subtile
//   local_k = local_word / 8, local_n_vec = local_word % 8
//   nibble i of the word = code at (n = n_tile*64 + local_n_vec*8 + pack_idx[i],
//                                   k = k_tile*16 + local_k)
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

// Copied verbatim from qpn2_layout.cuh (the layout's reference header): the
// plugin csrc does not vendor it.
__device__ __forceinline__ int lane_from_col(int col) {
  return ((col & 24) >> 1) | (col & 3) | ((col & 4) << 2);
}

__device__ __forceinline__ int physical_k(int k) {
  const int r = k & 7;
  return (k & 8) | ((r >> 1) | ((r & 1) << 2));
}


namespace {

constexpr int kTileK = 16;
constexpr int kTileN = 64;
constexpr int kTileWords = kTileK * kTileN / 8;  // 128 int32 per 16x64 tile
constexpr int kPackFactor = 8;

__device__ __forceinline__ uint32_t qpn2_nibble(uint8_t const* __restrict__ q, int n,
                                                int k, int groups) {
  const int tile = n >> 5;
  const int lane = lane_from_col(n & 31);
  const int group = k >> 4;
  const int p = physical_k(k & 15);
  const uint8_t byte = q[(static_cast<int64_t>(tile) * groups + group) * 288 + lane * 8 +
                         (p >> 1)];
  return (p & 1) ? (byte >> 4) : (byte & 0xF);
}

__global__ void repack_qpn2_marlin_kernel(uint8_t const* __restrict__ q,
                                          uint32_t* __restrict__ out, int experts,
                                          int size_n, int size_k, int cta_n_tiles) {
  constexpr int kPackIdx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  const int groups = size_k >> 4;
  const int tiles_n = size_n >> 5;
  const int64_t words_per_expert = static_cast<int64_t>(size_k / kTileK) * size_n * 2;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(experts) * words_per_expert) return;

  const int expert = static_cast<int>(idx / words_per_expert);
  int64_t rem = idx % words_per_expert;
  const int k_tile = static_cast<int>(rem / (size_n * 2));
  const int col = static_cast<int>(rem % (size_n * 2));

  const int block = cta_n_tiles * kTileWords;
  const int cta_n_tile = col / block;
  const int pos = col % block;
  const int local_word = pos / cta_n_tiles;
  const int subtile = pos % cta_n_tiles;
  const int local_k = local_word >> 3;
  const int local_n_vec = local_word & 7;
  const int n_tile = cta_n_tile * cta_n_tiles + subtile;
  const int first_n = n_tile * kTileN + local_n_vec * 8;
  const int k = k_tile * kTileK + local_k;

  uint8_t const* qe = q + static_cast<int64_t>(expert) * tiles_n * groups * 288;
  uint32_t res = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int n = first_n + kPackIdx[i];
    res |= qpn2_nibble(qe, n, k, groups) << (i * 4);
  }
  out[idx] = res;
}

// E4M3 group scales live at byte 256 + lane of each 288-byte group. A prefill
// fallback needs them as the [E, N, K/16] tensor the Marlin path expects.
__global__ void repack_qpn2_scales_kernel(uint8_t const* __restrict__ q,
                                         uint8_t* __restrict__ out, int experts,
                                         int size_n, int size_k) {
  const int groups = size_k >> 4;
  const int tiles_n = size_n >> 5;
  const int64_t per_expert = static_cast<int64_t>(size_n) * groups;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(experts) * per_expert) return;
  const int expert = static_cast<int>(idx / per_expert);
  int64_t rem = idx % per_expert;
  const int n = static_cast<int>(rem / groups);
  const int g = static_cast<int>(rem % groups);
  const int tile = n >> 5;
  const int lane = lane_from_col(n & 31);
  uint8_t const* qe = q + static_cast<int64_t>(expert) * tiles_n * groups * 288;
  out[idx] = qe[(static_cast<int64_t>(tile) * groups + g) * 288 + 256 + lane];
}

// Fused form of scales.encode_scales: E4M3 byte -> *128 -> fp16 -> bits >> 7,
// with the group-of-four column reorder the SM70 iterator expects. Removes
// ~1.7 ms per call of Python that overlap cannot hide.
__device__ __forceinline__ float e4m3_to_float(uint8_t byte) {
  const float sign = (byte & 0x80) ? -1.0f : 1.0f;
  const int exponent = (byte >> 3) & 0xF;
  const int mantissa = byte & 0x7;
  if (exponent == 0) return sign * static_cast<float>(mantissa) / 512.0f;
  if (exponent == 15) {
    // E4M3FN encodes NaN only at 0x7F and 0xFF. The rest of the exponent-15
    // band, 0x78..0x7E, is the finite 256..448 range, and the served
    // checkpoint uses it: its down-projection scales top out at 0x7E. Folding
    // the whole band into NaN turned those scales into NaN weights, which is
    // what the sampled-decoding assert was seeing.
    if (mantissa == 7) return __int_as_float(0x7fc00000);
    return sign * (1.0f + static_cast<float>(mantissa) / 8.0f) * 256.0f;
  }
  return sign * __powf(2.0f, static_cast<float>(exponent) - 7.0f) *
         (1.0f + static_cast<float>(mantissa) / 8.0f);
}

__global__ void repack_qpn2_scales_encoded_kernel(uint8_t const* __restrict__ q,
                                                  uint8_t* __restrict__ out, int experts,
                                                  int size_n, int size_k) {
  constexpr int kPerm[4] = {0, 2, 1, 3};
  const int groups = size_k >> 4;
  const int tiles_n = size_n >> 5;
  const int64_t per_expert = static_cast<int64_t>(size_n) * groups;  // [E, K/16, N]
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(experts) * per_expert) return;
  const int expert = static_cast<int>(idx / per_expert);
  int64_t rem = idx % per_expert;
  const int g = static_cast<int>(rem / size_n);
  const int n = static_cast<int>(rem % size_n);
  const int n_src = (n & ~3) + kPerm[n & 3];
  const int lane = lane_from_col(n_src & 31);
  uint8_t const* qe = q + static_cast<int64_t>(expert) * tiles_n * groups * 288;
  const uint8_t byte = qe[(static_cast<int64_t>(n_src >> 5) * groups + g) * 288 + 256 + lane];
  const __half h = __float2half_rn(e4m3_to_float(byte) * 128.0f);
  out[idx] = static_cast<uint8_t>(__half_as_ushort(h) >> 7);
}

int cta_n_tiles_for(int64_t size_n) {
  if (size_n % 256 == 0) return 4;
  if (size_n % 128 == 0) return 2;
  if (size_n % 64 == 0) return 1;
  TORCH_CHECK(false, "SM70 Marlin requires size_n divisible by 64, got ", size_n);
  return 1;
}

torch::Tensor repack_qpn2_marlin(torch::Tensor qpn2, int64_t size_n, int64_t size_k) {
  TORCH_CHECK(qpn2.is_cuda() && qpn2.is_contiguous() && qpn2.dtype() == at::kByte,
              "qpn2 bank must be a contiguous uint8 CUDA tensor");
  TORCH_CHECK(size_n % 64 == 0 && size_k % 16 == 0, "shape gate");
  const int64_t per_expert = (size_n / 32) * (size_k / 16) * 288;
  TORCH_CHECK(qpn2.numel() % per_expert == 0,
              "qpn2 bank is not a whole number of experts");
  const int experts = static_cast<int>(qpn2.numel() / per_expert);
  auto options = torch::TensorOptions().dtype(at::kInt).device(qpn2.device());
  torch::Tensor out = torch::empty({experts, size_k / kTileK, size_n * 2}, options);
  const int64_t total = static_cast<int64_t>(experts) * (size_k / kTileK) * size_n * 2;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  repack_qpn2_marlin_kernel<<<blocks, threads, 0, stream>>>(
      qpn2.data_ptr<uint8_t>(), reinterpret_cast<uint32_t*>(out.data_ptr<int>()),
      experts, static_cast<int>(size_n), static_cast<int>(size_k),
      cta_n_tiles_for(size_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void repack_qpn2_marlin_into(torch::Tensor qpn2, torch::Tensor out_weight,
                             torch::Tensor out_scales, int64_t size_n, int64_t size_k) {
  TORCH_CHECK(qpn2.is_cuda() && qpn2.is_contiguous() && qpn2.dtype() == at::kByte, "bad qpn2");
  const int64_t per_expert = (size_n / 32) * (size_k / 16) * 288;
  const int experts = static_cast<int>(qpn2.numel() / per_expert);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t words = static_cast<int64_t>(experts) * (size_k / kTileK) * size_n * 2;
  const int threads = 256;
  repack_qpn2_marlin_kernel<<<static_cast<int>((words + threads - 1) / threads), threads, 0,
                              stream>>>(
      qpn2.data_ptr<uint8_t>(), reinterpret_cast<uint32_t*>(out_weight.data_ptr<int>()),
      experts, static_cast<int>(size_n), static_cast<int>(size_k), cta_n_tiles_for(size_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t scale_elems = static_cast<int64_t>(experts) * size_n * (size_k / 16);
  repack_qpn2_scales_kernel<<<static_cast<int>((scale_elems + threads - 1) / threads), threads, 0,
                              stream>>>(
      qpn2.data_ptr<uint8_t>(), out_scales.data_ptr<uint8_t>(), experts,
      static_cast<int>(size_n), static_cast<int>(size_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void repack_qpn2_scales_encoded_into(torch::Tensor qpn2, torch::Tensor out_encoded,
                                     int64_t size_n, int64_t size_k) {
  const int64_t per_expert = (size_n / 32) * (size_k / 16) * 288;
  const int experts = static_cast<int>(qpn2.numel() / per_expert);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = static_cast<int64_t>(experts) * size_n * (size_k / 16);
  const int threads = 256;
  repack_qpn2_scales_encoded_kernel<<<static_cast<int>((total + threads - 1) / threads),
                                      threads, 0, stream>>>(
      qpn2.data_ptr<uint8_t>(),
      reinterpret_cast<uint8_t*>(out_encoded.data_ptr()), experts,
      static_cast<int>(size_n), static_cast<int>(size_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor repack_qpn2_scales(torch::Tensor qpn2, int64_t size_n, int64_t size_k) {
  const int64_t per_expert = (size_n / 32) * (size_k / 16) * 288;
  const int experts = static_cast<int>(qpn2.numel() / per_expert);
  auto options = torch::TensorOptions().dtype(at::kByte).device(qpn2.device());
  torch::Tensor out = torch::empty({experts, size_n, size_k / 16}, options);
  const int64_t total = static_cast<int64_t>(experts) * size_n * (size_k / 16);
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  repack_qpn2_scales_kernel<<<blocks, threads, 0, stream>>>(
      qpn2.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), experts,
      static_cast<int>(size_n), static_cast<int>(size_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("repack_qpn2_marlin", &repack_qpn2_marlin,
        "QPN2 bytes -> SM70 Marlin layout, one launch for a whole layer");
  m.def("repack_qpn2_scales", &repack_qpn2_scales,
        "QPN2 bytes -> [E, N, K/16] E4M3 group scales");
  m.def("repack_qpn2_marlin_into", &repack_qpn2_marlin_into,
        "QPN2 bytes -> caller-owned Marlin weight and scale buffers");
  m.def("repack_qpn2_scales_encoded_into", &repack_qpn2_scales_encoded_into,
        "QPN2 bytes -> encoded [E, K/16, N] scales for the Marlin kernel");
}
