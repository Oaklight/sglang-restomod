# task.md — 交接

最后更新：2026-11-01
跟踪：`sglang-restomod` #93 / #94 / #95

## 一句话

NVFP4 已在 4×V100 单岛上跑通：无 MTP 44.05 tok/s，MTP3/4 **88.8 tok/s**。
FP8 8 卡 + MTP 是 73.9 tok/s（accept len 2.83）。

## 现在能跑什么

| 模型 | 卡数 | decode | 备注 |
|---|---|---|---|
| Flash-Next-NVFP4 | 4 | 44.05 tok/s | 无 MTP，端口 11446 |
| Flash-Next-NVFP4 + MTP3/4 | 4 | **88.8 tok/s** | accept 3.12，端口 11446，当前在跑 |
| Flash-Next-FP8 + MTP3/4 | 8 | 73.9 tok/s | accept 2.83；temp 0 时 98—106 |
| Qwen3.8-27B-FP8 | 2 / 4 | 33.7 / 48.2 tok/s | 见 #94 |

测量方法：prompt 74—76 token、输出 128、temperature 0、3 次重复，丢弃首次预热。
吞吐与 accept len 强相关，比较前先对齐采样参数。

## NVFP4 路径（本轮新增）

1. **扩展**：`zhinianqin/marlin_v100@3f16d442`，加 3 个补丁（见
   `sglang_restomod/quant/sm70_fp4_moe/PROVENANCE.md`）。
   产物在 lambda1 的 `/homes/pding/sm70_marlin/`：`_C.abi3.so`、`_moe_C.abi3.so`。
   源码在 `/homes/pding/marlin_build`，CUTLASS 用 `/homes/pding/cutlass`（v4.2.1）。
2. **`fp4_moe` 组件**：
   - `method.py`：替换 `prepare_moe_nvfp4_layer_for_marlin` 与
     `moe_wna16_marlin_gemm`，并桥接两边不同的签名。
   - `scales.py`：把 group-16 E4M3 scale 编成 S0E5M3 字节，保持逻辑
     `[E, K/16, N]` 顺序。不得调用 `marlin_permute_scales`。
   - `sum_reduce.py`：补 `sgl_kernel.moe_sum_reduce`。安装的 sgl_kernel
     没有这个算子；runner 在第一次前向就会失败。
3. **`ple_sm70` 组件**：替换 PLE host-embedding 取表内核。pin 的版本用
   `tl.float8e4nv`，SM70 不支持；新内核用整数位运算解码 e4m3fn。
4. **启动**：`--quantization modelopt_fp4 --moe-runner-backend marlin
   --tp-size 4 --ep-size 1 --ple-offload-embedding`。

## 必须知道的约束

1. **4 卡必须 TP4、无 EP。** Marlin 内核按全局专家 id 索引本 rank 的权重；
   EP 会把远程专家标成 -1。
2. **PLE 必须 offload。** 4 卡放不下 n-gram 表。
3. **显存很紧。** 每卡常驻约 23 GiB；`--mem-fraction-static 0.85` 只剩 0.89 GiB。
   MTP 用 0.82。
4. **第一次前向会 JIT 编译 `sum_reduce` 内核。** 日志会出现
   `device-loaded after serving started` 警告，之后正常。
5. **`/nfs` 只剩约 293 GB。** 7 个模型目录已移到
   `/nfs/ml_lab/projects/ml_lab/pding/models/`，原路径是符号链接。

## 启动命令

```sh
# NVFP4 TP4 + MTP（当前在跑，端口 11446）
ssh lambda1 'bash /homes/pding/pin_launch_nvfp4_tp4_mtp.sh'

# NVFP4 TP4，无 MTP
ssh lambda1 'bash /homes/pding/pin_launch_nvfp4_tp4_marlin.sh'
```

两个脚本都用 `PYTHONPATH=/homes/pding/pin-site:/homes/pding/restomod-test`，
并设置 `SM70_MARLIN_DIR=/homes/pding/sm70_marlin`。

部署：

```sh
rsync -a --delete --exclude '.build' --exclude 'third_party' --exclude '*.egg-info' \
  --exclude '__pycache__' --exclude '.git' --exclude 'site' \
  /home/pding/projects/sglang-restomod/ lambda1:/homes/pding/restomod-test/
```

## 未验证

- 真实 checkpoint 的 FP4 nibble 顺序。只由生成质量间接支持；未用原始权重反量化对比。
- NVFP4 的 accept len（3.12）为何低于 FP8（3.48）。原因未确认。
- 长上下文（>4k）的 prefill 与 decode。当前 context-length 是 4096。

## 建议的下一步（按性价比）

1. 用参考树的 `_configure_sm70_nvfp4_stage` 参数替换自动 CTA 选择：
   gate/up `32x64x64x4x32x64x16`，down `64x256x32x4x64x64x32`。
   目标：35.2 ms/step → 32 ms 以下。
2. 查 accept len 的差距：确认是量化噪声还是 draft 路径。
3. 测 8k / 32k 上下文，按用户目标补齐 prefill 数据。
4. 把当前改动走 PR 合入（规则要求不以 main 直接提交）。

## 关键文件

- `sglang_restomod/quant/sm70_fp4_moe/` — NVFP4 全部代码
- `sglang_restomod/models/ple_sm70.py` — PLE 内核
- `sglang_restomod/quant/sm70_fp8_moe/` — FP8 专家 GEMM 范式
- `sglang_restomod/overrides.py` — 运行时钩子（含 `_install_fp4_min_capability`）
