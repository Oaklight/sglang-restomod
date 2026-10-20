#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>

#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>
#include <cublas_v2.h>

namespace sglang {

// Use the caller's thread-local, current-stream cuBLAS handle. Scope its math
// mode instead of mutating PyTorch's process-wide accumulation policy.
inline void sm70_fp8_gemm_fp16_output(
    tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView y, int64_t handle_address) {
  using namespace host;
  SymbolicSize m{"m"}, k{"k"}, pk{"pk"}, pn{"pn"}, n{"n"};
  SymbolicDevice device;
  TensorMatcher({m, k}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).ensure_alignment(2).verify(x);
  TensorMatcher({pk, pn}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).ensure_alignment(2).verify(w);
  TensorMatcher({m, n}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).ensure_alignment(2).verify(y);
  CHECK_HOST(handle_address != 0 && m.unwrap() > 0 && m.unwrap() <= INT_MAX);
  CHECK_HOST(0 < k.unwrap() && k.unwrap() <= pk.unwrap() && pk.unwrap() <= INT_MAX);
  CHECK_HOST(0 < n.unwrap() && n.unwrap() <= pn.unwrap() && pn.unwrap() <= INT_MAX);
  auto handle = reinterpret_cast<cublasHandle_t>(handle_address);
  cublasPointerMode_t pointer_mode;
  CHECK_HOST(cublasGetPointerMode(handle, &pointer_mode) == CUBLAS_STATUS_SUCCESS);
  CHECK_HOST(pointer_mode == CUBLAS_POINTER_MODE_HOST);
  cublasMath_t previous;
  CHECK_HOST(cublasGetMathMode(handle, &previous) == CUBLAS_STATUS_SUCCESS);
  CHECK_HOST(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) == CUBLAS_STATUS_SUCCESS);
  float alpha = 1.0f, beta = 0.0f;
  // Row-major X*W is column-major W*X. The physical leading dimension of W
  // preserves Marlin's padded columns while Y contains only logical columns.
  const auto status = cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n.unwrap(),
      m.unwrap(),
      k.unwrap(),
      &alpha,
      w.data_ptr(),
      CUDA_R_16F,
      pn.unwrap(),
      x.data_ptr(),
      CUDA_R_16F,
      k.unwrap(),
      &beta,
      y.data_ptr(),
      CUDA_R_16F,
      n.unwrap(),
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  const auto restored = cublasSetMathMode(handle, previous);
  CHECK_HOST(status == CUBLAS_STATUS_SUCCESS) << "SM70 FP8 prefill cuBLAS GEMM failed: " << status;
  CHECK_HOST(restored == CUBLAS_STATUS_SUCCESS) << "Could not restore cuBLAS math mode";
}

}  // namespace sglang
