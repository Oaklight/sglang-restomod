## 2026-11-06 decode 重新预算（#219）

**Commit**: `1c11d10150`（当前 main；含 #202、#206/#216、#207）
**Launch**: lambda6 GPUs 0-3，4x V100-SXM2-32GB，TP4/EP1，MTP 3/4，greedy（temperature 0），prompt 4096，输出 400，`--context-length 8192`，`--mem-fraction-static 0.78`（slot 内存余量下调；recipe 原值 0.82），custom allreduce 开（pin-site patch 0002）。
**Components**: `attention_v100,fp4_moe,moe_align_sm70,ple_sm70,spec_sm70,fp8_dense,fp8_lm_head,fp8_dense_quant`
**Window**: 16 个 decode forward，同一 launch 采 2 次；下表用第 2 次（干净窗口）。
**Anchor**: 400 greedy tokens，decode 2.562 s，accept len 4.00 → **155.75 tok/s**；trace span 26.94 ms/step。

Accept 4.00 与 README 行（3.12）不同，prompt 也是 4096 而非 3481，所以 tok/s 不能直接与 89.22 比。可比的量是 step time：旧 trace 31.36 ms → 本次 26.40 ms（-15.8%）。

### 每步预算（TP-0，kernel device time）

| 项 | ms/step | 占比 |
|---|---:|---:|
| MoE gate_up | 3.350 | 12.7% |
| MoE down | 2.363 | 9.0% |
| attention | 2.887 | 10.9% |
| allreduce | 0.790 | 3.0% |
| norm/其他 | 17.007 | 64.4% |
| **合计** | **26.398** | 100% |

`norm/其他` 拆开：

| 子项 | ms/step | 占比 |
|---|---:|---:|
| dense GEMM | 10.355 | 39.2% |
| 其他（hc_combine、triton_poi、copy 等） | 2.779 | 10.5% |
| MoE router | 1.641 | 6.2% |
| MoE align/act/sum | 1.000 | 3.8% |
| MTP draft MoE | 0.691 | 2.6% |
| norm | 0.541 | 2.1% |

`moe_gate_up` = `decode_moe_gemv_kernel` 3.198 ms（48 calls，66.6 us/call）+ `reduce_split_k_kernel` 0.153 ms。down 仍是 Marlin（48 calls，49.2 us/call）。`moe_align/act/sum` 里的 align 已从 Triton 的 28.3 us/call 降到 CUDA pair 的 5.0 us/call（`moe_align_sm70`）。

### 与 2026-11-04 旧预算的差

| 项 | 2026-11-04 | 本次 | Δ |
|---|---:|---:|---:|
| gate_up | 6.266 | 3.350 | -2.916 |
| align/act/sum | 2.067 | 1.000 | -1.067 |
| draft MoE | 1.560 | 0.691 | -0.869 |
| allreduce | 0.988 | 0.790 | -0.198 |
| down | 2.266 | 2.363 | +0.097 |
| 其余 | 18.213 | 18.204 | -0.009 |
| **合计** | **31.36** | **26.40** | **-4.96** |

对应 #202（GEMV + per-shape split-K）、`moe_align_sm70`、pin-site 的 draft Triton config。dense GEMM 没变（10.32 → 10.36），现在是最大单项。

### allreduce：20.6% 不复现

两臂同 launch、同请求、各 2 次采样；干净窗口（第 2 次）：

| 臂 | kernel | calls/step | mean us | p50 | p90 | p99 | max | ms/step | 占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| custom v2 | `all_reduce_1shot_push_kernel` | 106 | 5.95 | 5.06 | 8.96 | 20.61 | 46.8 | 0.631 | 2.4% |
| custom v2 | `ncclDevKernel_AllGather_RING_LL` | 4 | 38.45 | 28.38 | 71.68 | 83.84 | 83.8 | 0.154 | 0.6% |
| NCCL | `ncclDevKernel_AllReduce_Sum_f16_RING_LL` | 106 | 32.42 | 29.44 | 47.01 | 57.35 | 88.4 | 3.436 | 11.6% |
| NCCL | `ncclDevKernel_AllGather_RING_LL` | 4 | 48.69 | 38.72 | 95.31 | 104.62 | 104.6 | 0.195 | 0.7% |

- custom 总 allreduce **0.790 ms/step（3.0%）**；NCCL 总 **3.664 ms/step（12.4%）**。step 26.40 → 29.65 ms（-3.25 ms），anchor 155.75 → 132.28 tok/s。
- #206 的 76.7 us/call、20.6%（custom）与 245.5 us/call、44.8%（NCCL）**在本机安静岛上不复现**。本次 custom mean 5.95 us，NCCL mean 32.42 us；绝对值低 7—13 倍，两臂比值 5.4x（#206 是 3.2x），方向一致。
- 采样窗口的解释：custom 第 1 次采样的 step 0 有两个离群值（19.2 ms、11.1 ms），first-step share 71%，把 allreduce 抬到 2.679 ms（9.5%）；同一 launch 的第 2 次是 0.631 ms（3.0%）。所以 #206 的绝对值更像是 profiler 起始/主机状态的产物，而不是 kernel 属性（**推断**，不是已验证的因果）。
- 结论：custom allreduce 保留是对的（省 3.25 ms/step），但它**不是**当前 decode 的最大项；最大单项是 dense GEMM 的 cutlass 16x16（4.02 ms，15.2%）。

### 候选（按可优化上限排序）

1. **dense GEMM 剩余 bf16 线性层转 block FP8**（10.355 ms/step，39.2%）
   - 依据：bf16 子项约 7.735 ms（cutlass 16x16 4.021 + 32x32 1.734+0.595 + gemvSN 0.370 + splitKreduce 0.623+0.392）。M≤4 时是带宽受限；字节减半的上限 **3.9 ms/step**。已测同类比值：#196 的 49 个线性层转换净省 1.71 ms；按同比例推算剩余子集约 **1.6—2.5 ms/step**。
   - 动作：把 `fp8_dense_quant` 的准入扩到 attention/GDN 投影，而不只是 N%128==0 的那些；或为剩余 N 形状加 block-128 内核。用同一 trace 复测。
2. **MoE router**（1.641 ms/step，6.2%；`_router_triton_kernel` 1.489 ms，51 calls，29.2 us/call）
   - 依据：每 call 读 router 权重 512x2560 bf16 = 2.62 MB，带宽下限 2.9 us/call；实测 29.2 us，约 10% 带宽。上限 **(29.2-2.9)x51 = 1.34 ms/step**。
   - 动作：给 M=4 的 router 写 tuned Triton config（同 draft MoE 的做法），或把 gate sigmoid/top-k 融进 router；目标 ≤10 us/call。
3. **GDN 线性注意力 update**（1.199 ms/step，4.5%；`fused_sigmoid_gating_delta_rule_update_kernel` 36 calls，33.3 us/call）
   - 依据：每层 state 48x128x128 fp32 = 3.15 MB，读+写 6.3 MB，带宽下限约 7.0 us/call；实测 33.3 us，约 21% 带宽。上限 **(33.3-7.0)x36 = 0.95 ms/step**。
   - 动作：先查该 kernel 的 tile/occupancy 和 self-time；state 流量是下限，改之前要确认不是 launch/occupancy 受限。

补充（未进前三）：MoE down 2.363 ms（49.2 us/call）。gate_up GEMV 用 66.6 us 读 2x 字节，按同类比值 down 的 floor 约 33 us；但 even-N 的 GEMV 已经测过（49.3 vs Marlin 48.3 us），没有收益，所以不作为候选，除非换 split-K/tile 方案。

### 产物

- 原始 trace 与 JSON：`/nfs/lambda_stor_01/homes/pding/issue219/run-20261106-161227/`
- 两臂 `launch_config.txt`、`apps.before.csv`、`clocks.before.csv`、`ar_init.log` 同目录。
- 分析脚本：`scripts/trace_decode_budget.py`（worktree `sglang-restomod-wt-219`，CPU 测试 5/5，ruff clean）。

验证状态：预算表与 allreduce A/B **已验证**（两臂各 2 次采样，干净窗口一致）；三个候选的上限是**未验证**的估算，按 issue 的 DoD 只给有测量依据的上限，没有实现。
