// Bind the FP8 MoE GEMM that the prebuilt SM70 TurboMind library already
// exports, without rebuilding that library.
//
// The symbol `fp8_moe_gemm_sm70_out` is present in
// `_sm70_turbomind_v100.so` but is not registered with torch, and the kernel
// sources are not part of the SGLang tree. Resolve the symbol through
// dlopen/dlsym and expose it as `sglang_sm70_fp8_moe::fp8_moe_gemm`.
//
// Signature and schema follow the origin implementation in
// `1CatAI/1Cat-vLLM` (`csrc/ops.h`, `csrc/torch_bindings.cpp`), which the
// prebuilt library was compiled from.
#include <torch/all.h>
#include <torch/library.h>

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>

namespace {

using Fn = void (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                    int64_t, int64_t, int64_t, int64_t, bool);

Fn resolve() {
  static Fn fn = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    const char* path = std::getenv("SM70_TURBOMIND_SO");
    TORCH_CHECK(path != nullptr, "SM70_TURBOMIND_SO is not set");
    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    TORCH_CHECK(handle != nullptr, "dlopen failed: ", dlerror());
    fn = reinterpret_cast<Fn>(dlsym(
        handle, "_Z21fp8_moe_gemm_sm70_outN2at6TensorES0_S0_S0_S0_llllb"));
    TORCH_CHECK(fn != nullptr, "dlsym failed: ", dlerror());
  });
  return fn;
}

void fp8_moe_gemm(at::Tensor out, at::Tensor sorted_input,
                  at::Tensor expert_offsets, at::Tensor strided_ptrs_w,
                  at::Tensor strided_ptrs_s, int64_t num_experts, int64_t k,
                  int64_t n, int64_t group_size, bool gated_silu) {
  resolve()(out, sorted_input, expert_offsets, strided_ptrs_w, strided_ptrs_s,
            num_experts, k, n, group_size, gated_silu);
}

}  // namespace

TORCH_LIBRARY(sglang_sm70_fp8_moe, ops) {
  ops.def(
      "fp8_moe_gemm(Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, int k, "
      "int n, int group_size, bool gated_silu) -> ()");
  ops.impl("fp8_moe_gemm", torch::kCUDA, &fp8_moe_gemm);
}
