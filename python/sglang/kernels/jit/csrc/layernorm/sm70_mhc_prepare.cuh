#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <climits>
#include <cstdint>
#include <cuda_fp16.h>

namespace sglang {
// Match torch's contiguous float mean: four vector accumulators, 512 threads,
// descending warp shuffle, then descending 16-warp tree. Keep full FP32 GEMM X.
__global__ void sm70_mhc_prepare_kernel(
    const __half* __restrict__ input, float* __restrict__ x, float* __restrict__ mean, int32_t first, int32_t count) {
  const int32_t row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool owned = row >= first && row < first + count;
  float sums[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int32_t v = tid; v < 4096; v += 512) {
    const int64_t off = int64_t(row) * 16384 + v * 4;
    float4 f;
    f.x = __half2float(input[off]);
    f.y = __half2float(input[off + 1]);
    f.z = __half2float(input[off + 2]);
    f.w = __half2float(input[off + 3]);
    *reinterpret_cast<float4*>(x + off) = f;
    if (owned) {
      sums[0] = __fadd_rn(sums[0], __fmul_rn(f.x, f.x));
      sums[1] = __fadd_rn(sums[1], __fmul_rn(f.y, f.y));
      sums[2] = __fadd_rn(sums[2], __fmul_rn(f.z, f.z));
      sums[3] = __fadd_rn(sums[3], __fmul_rn(f.w, f.w));
    }
  }
  if (!owned) return;
  float sum = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
#pragma unroll
  for (int32_t d = 16; d > 0; d >>= 1)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, d));
  __shared__ float partial[16];
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (tid < 16) {
    sum = partial[tid];
#pragma unroll
    for (int32_t d = 8; d > 0; d >>= 1)
      sum = __fadd_rn(sum, __shfl_down_sync(0xffff, sum, d));
    if (tid == 0) mean[row - first] = sum * (1.f / 16384.f);
  }
}
/// Prepare the full FP32 mHC projection input and one shard's exact squared mean.
inline void
sm70_mhc_prepare(tvm::ffi::TensorView residual, tvm::ffi::TensorView x, tvm::ffi::TensorView mean, int64_t first) {
  using namespace host;
  SymbolicSize rows{"rows"}, hc{"hc"}, hidden{"hidden"}, flat{"flat"}, count{"count"};
  hc.set_value(4);
  hidden.set_value(4096);
  flat.set_value(16384);
  SymbolicDevice device;
  TensorMatcher({rows, hc, hidden})
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(2)
      .verify(residual);
  TensorMatcher({rows, flat}).with_dtype<float>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({count}).with_dtype<float>().with_device<kDLCUDA>(device).ensure_alignment(4).verify(mean);
  CHECK_HOST(rows.unwrap() >= 1024 && rows.unwrap() <= INT_MAX);
  CHECK_HOST(first >= 0 && count.unwrap() > 0 && first <= rows.unwrap() - count.unwrap());
  LaunchKernel(rows.unwrap(), 512, device.unwrap())(
      sm70_mhc_prepare_kernel,
      reinterpret_cast<const __half*>(residual.data_ptr()),
      static_cast<float*>(x.data_ptr()),
      static_cast<float*>(mean.data_ptr()),
      static_cast<int32_t>(first),
      static_cast<int32_t>(count.unwrap()));
}
}  // namespace sglang
