## 更正：20.6% / 76.7 us 是尾部伪影，不是 kernel 成本（#219 复测）

#219 在安静岛上用同一脚本重测了 custom 与 NCCL 两臂（commit `1c11d10150`，lambda6 0-3，TP4，4096 prompt，400 greedy tokens，16 步 × 2 次）。结论：**本 issue 的结论（custom AR 应该开）成立，但 20.6% / 76.7 us 与 44.8% / 245.5 us 这两个绝对份额不成立。**

### 同一批 #206 trace 的分布

用 `trace_decode_budget.py` 重新聚合 #206 自己的 TP0 trace：

| 臂 | kernel | calls | p50 us | p90 us | p99 us | max us | mean us |
|---|---|---:|---:|---:|---:|---:|---:|
| custom | `all_reduce_1shot_push_kernel` | 2114 | **8.7** | 48.6 | 1559 | 1750 | 76.7 |
| NCCL | `ncclDevKernel_AllReduce_Sum_f16_RING_LL` | 2114 | **30.2** | 47.9 | 4839 | 24316 | 245.5 |

两个 mean 都由少数 multi-ms 停顿拉高：custom 最慢 5 次在 step 3/5/6（1.65—1.75 ms），NCCL 最慢 5 次在 step 0/1（7.5—24.3 ms）。#206 的 custom trace 还有 17.6 ms/step 的 idle/gap（#219 是 0.54 ms），并含 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（prefill kernel，0.515 ms/step），说明窗口不是纯 decode。

### 安静岛复测（#219）

| 臂 | kernel | calls/step | mean us | p50 | p90 | p99 | max | ms/step | 占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| custom | `all_reduce_1shot_push_kernel` | 106 | 5.95 | 5.06 | 8.96 | 20.61 | 46.8 | 0.631 | 2.4% |
| NCCL | `ncclDevKernel_AllReduce_Sum_f16_RING_LL` | 106 | 32.42 | 29.44 | 47.01 | 57.35 | 88.4 | 3.436 | 11.6% |

总计：custom 0.790 ms/step（3.0%），NCCL 3.664 ms/step（12.4%）。step 26.40 → 29.65 ms（-3.25 ms），anchor 155.75 → 132.28 tok/s。

### 差异来自哪里

1. **尾部**：两臂的 mean 都被稀疏停顿主导；p50 才是稳态成本。#206 custom p50 8.7 us 与本次 5.06 同量级；NCCL p50 30.2 vs 29.44 几乎一致。
2. **checkout/组件**：#206 的 trace 里 gate_up 仍是 Marlin（128.9 us/call，无 `decode_moe_gemv_kernel`），align 仍是 Triton（28.5 us/call），说明那次 launch 早于 #202、也没开 `moe_align_sm70`；本次两者都开，step 从 41.33 降到 26.40 ms。
3. **窗口**：#206 的窗口含 prefill 内核和 17.6 ms/step 主机 gap；本次是 decode-only，gap 0.54 ms。
4. **步数**：#206 的 anchor（`_moe_align_small_numel_kernel`）是 969 次 / 51 = 19 个 forward，不是 20；因此 per-step 总量应除以 19（785.3/19 = 41.33 ms，不是 39.3）。份额（20.6%）不受影响。

### 结论

- **custom AR 应该开**：安静岛 A/B 省 3.25 ms/step；#206 自己的 p50 也显示 8.7 vs 30.2 us。
- **20.6% 不是当前 main 的 allreduce 份额**：安静岛是 **3.0%**；NCCL 是 12.4%。
- **它不是最大的单项**：当前 main 最大单项是 dense cutlass 16x16 GEMM（4.02 ms，15.2%）。下一步工作项应是 dense GEMM，不是 AR kernel。

原始数据：#219 的 `run-20261106-161227/`；#206 的 trace 在 `tmp/ar_probe_206/profiles/`，上面的分布就是用它重算的。
