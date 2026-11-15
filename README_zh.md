<p align="center">
  <a href="README_en.md">English</a> | <a href="README_zh.md">中文</a>
</p>

# sglang-restomod

[![PyPI version](https://img.shields.io/pypi/v/sglang-restomod?color=green)](https://pypi.org/project/sglang-restomod/)
[![CI](https://github.com/Oaklight/sglang-restomod/actions/workflows/restomod-ci.yml/badge.svg)](https://github.com/Oaklight/sglang-restomod/actions/workflows/restomod-ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

一个面向 [SGLang](https://github.com/sgl-project/sglang) 的**轻量下游层**。

`main` 不包含 SGLang 源码树，只包含：

- 固定的上游基线（`UPSTREAM_PIN.toml`）；
- 插件包（`sglang_restomod/`）；
- 小而可审计的 patch 队列（`patches/`）；
- recipes 和 CI。

上游组件按需逐个引入，每个都记录来源、commit 和测试。本仓库不是 SGLang 或任何 V100 项目的 fork。

## 状态

目前有十三个组件，都通过 `SGLANG_RESTOMOD_COMPONENTS` 显式启用：

| 组件 | 作用 |
|---|---|
| `fp8_moe` | sm70 上的 block-FP8 expert GEMM。SGLang 随包提供的 SM70 TurboMind 库导出了 `fp8_moe_gemm_sm70_out`，但从未向 torch 注册；本组件用一个小的 `dlopen`/`dlsym` shim 解析该符号并暴露类型化接口。官方 block-FP8 MoE 权重能在 V100 上跑，靠的就是它。 |
| `fp8_dense` | 固定 pin 引擎的 SM70 W8A16 稠密 FP8 线性层内核。 |
| `fp8_lm_head` | 加载后把 LM head 量化为 block FP8，交给 `fp8_dense`。27B 权重把 `lm_head` 放在 `ignored_layers`，所以 head 保持 f16，每个 draft step 每卡读 1.27 GB 权重。单卡 M=6（TP2 分片形状）时 f16 线性层 2.001 ms，FP8 0.793 ms；引擎内 8k decode 从 63.08 升到 70.37 tok/s（三次新起进程，512 输出 token），32k 基本不变。依赖 `fp8_dense`，缺失时拒绝安装。 |
| `fp8_dense_quant` | 加载后把 BF16 权重的稠密线性层量化为 block FP8，交给 `fp8_dense`。M=1..4 时这些 GEMM 受带宽限制，字节数就是成本。准入很窄：二维权重、两侧都是 128 的倍数、不是 expert、不是 LM head。Flash-Next NVFP4 上配合 `fp8_lm_head`，decode step 从 32.79 降到 31.08 ms（issue #95）。 |
| `attention_v100` | pin 上的 V100 全注意力。内置 `flash_attn_v100_backend.py`、SM70 QSA backend 与 `tilelang_fa_v100` 内核，注册 pin 未定义的 `tilelang_fa_v100` 后端名，并在该 backend 上实现 `init_forward_metadata_out_graph`（graph runner 会调用，基类留成 no-op）。缺最后这项则 decode CUDA graph 无法捕获；缺前两项则启动直接失败。MTP 的 draft decode 也走同一个 split-KV kernel：把每个 draft step 的 KV 索引表转成 page table。 |
| `fp4_moe` | 经 `marlin_v100` 扩展实现的 NVFP4 expert GEMM：加载期 repack、SM70 scale 编码、GEMM 桥接。SGLang 随包的 Marlin 内核在 SM80 以下是空实现。 |
| `mtp_fp4_moe` | 把投机解码草稿模型的专家放到与目标模型同一条 NVFP4 Marlin GEMM 上。检查点的 `quantization_config.ignore` 列了 `mtp.*`，草稿的专家层因此以未量化权重加载，引擎无论 `--moe-runner-backend` 怎么设都会为它构造 Triton runner。本组件在加载后把这些专家量化成 NVFP4 检查点布局（block 16、每个专家一个全局 scale），补上 NVFP4 导出本应提供的张量，并把该层交给引擎的 NVFP4 MoE 方法；repack 与执行仍由 `fp4_moe` 完成。按名选择，默认关闭。准入率（accept）是验收门：4 bit 草稿比对照集里的任何先例都激进（本检查点把 `mtp.*` 排除在 NVFP4 外，DeepSeek 的 FP8 检查点保留 MTP 为 BF16，1Cat 的草稿是 FP8），所以步进更快而 accept 下降是交换，不是收益。 |
| `fp4_dense` | 经 `marlin_v100` 扩展实现的 dense NVFP4（W4A16）线性 GEMM。带 `--fp4-gemm-backend marlin` 时，pin 的 `ModelOptFp4LinearMethod` 在量化激活之前就返回 `apply_fp4_marlin_linear`，所以标为 `NVFP4` 的层在 Volta 上按 W4A16 运行，缺的只是内核。本组件把 `marlin_utils_fp4` 导入到自身命名空间里的三个名字（GEMM、repack、scale 置换）接到 SM70 构建上，并按 A16 方法的方式折叠融合层的按分片 global scale。 |
| `sm70_marlin_v100` | 经 `marlin_v100` 扩展实现的稠密 WNA16（INT4 GPTQ/AWQ）线性 GEMM。pin 会 JIT 编译自己的 Marlin 内核，而该构建在 sm70 上无法编译（`namespace sglang::device::marlin has no member marlin_mm`）。本组件把引擎的 GPTQ 调用转到 `SM70_MARLIN_DIR` 指向的预编译 SM70 内核，并解除上游在 compute capability 7.0 上拒绝 INT4 的门槛。 |
| `ple_sm70` | fp8 PLE host-embedding gather 的 SM70 版本。固定 pin 的内核会 cast 到 `tl.float8e4nv`，在 SM70 上无法编译。 |
| `spec_sm70` | top-k/top-p renorm 替换，使非贪心 EAGLE verify 不再触到 FlashInfer 的 sm75 下限。 |
| `moe_align_sm70` | 把 tiny-batch MoE 对齐转到 SM70 上的 CUDA 双内核。pin 在 decode 尺寸下选单 CTA 的 Triton 内核；在 Flash-Next TP4 NVFP4 MoE 上它每次调用 29.3 us，而通用 CUDA 双内核为 5.0 + 2.3 us，因为单个 CTA 让 80 SM 的卡空转。把 `SMALL_NUMEL_LIMIT` 设为 0 即可让 Triton 分支不可达，所有调用走双内核；两者产出的元数据相同。 |
| `gdn_tilelang_sm70` | SM70 上的 TileLang GDN 分块预填内核。pin 的 Triton 分块内核占 27B 单次预填分块的 63.9%；本组件用 TileLang 内核接管同一次调用：8,041 token 时 1,933.6 / 1,924.9 tok/s，31,158 token 时 1,658.3 / 1,655.1，Triton 下分别为 719.9 与 684.6（凭证 `evidence-55cb6e7-gdn-tilelang`）。组件同时把 FLA 包装函数 `chunk_fwd_o` 与 `chunk_gated_delta_rule_fwd_h` 重定向到 `sm70-kernels` 的 TileLang 内核对，因此分块路径的所有调用方都能受益；内核无法精确计算的情况一律回落到 Triton。`SGLANG_V100_GDN_TILELANG_IMPL=auto|pair|whole` 选择替换方式，`SM70_GDN_TILELANG=0` 是 A/B 对比用的 Triton 对照组。 |
| `lsa_sm70` | SM70 全注意力的 Latent Sparse Attention（LSA）。key 在学习到的隐空间里聚类，每个 query 只对桶中心打分，读取时取选中的 key 加本地窗口和 sink。移植 Qwen3.8-27B 的 flat D-LSA 设计（16 个全注意力层，GQA 24/4），索引按 attention TP 组分片，选择由两个 Triton program 完成，读取走 pin 的 Triton decode 内核。选择 `--attention-backend lsa_sm70`；需要 `SGLANG_RESTOMOD_LSA_MAP`、`--page-size 1` 和 `--disable-cuda-graph`。实现是独立包 `Oaklight/lsa-sm70`，commit 固定在 `UPSTREAM_PIN.toml`；本组件只负责注册，先按该 commit 安装：`pip install 'lsa-sm70 @ git+https://github.com/Oaklight/lsa-sm70@3a4798093224933988175ee685084d521474da0b'`。 |

```bash
SGLANG_RESTOMOD_COMPONENTS=fp8_moe,attention_v100
```

不设置这个变量时插件保持惰性。patch 队列有三处上游修改，路径和理由都记录在
`patches/manifest.json`：HiCache/OpenSSL 头文件回退、`cuda_ipc` JIT 修复（缺它会
静默关闭 custom allreduce）、以及 `fused_marlin_moe` 中的 QPN2 形状门。其余能力
全部通过插件入口生效。

## 参考项目

以下项目定义本层要复现的行为。只使用组件 `PROVENANCE.md` 中列出的代码，其余仅作参考。

| 项目 | 用途 |
|---|---|
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 固定 pin 的上游引擎。 |
| [heislera763/sglang-v100-plus](https://github.com/heislera763/sglang-v100-plus) | V100 基线，也是集成参考；本地快照为 `sglang-restomod-reference`。 |
| [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) | 该谱系中最早的 Volta 移植。 |
| [dg1kjd/sglang-sxm2](https://github.com/dg1kjd/sglang-sxm2) | `attention_v100` backend 与长上下文 decode kernel。 |
| [heislera763/flash-attention-v100](https://github.com/heislera763/flash-attention-v100) | `attention_v100` 使用的 dense prefill attention。 |
| [zhinianqin/marlin_v100](https://github.com/zhinianqin/marlin_v100) | block-FP8 与 NVFP4 expert GEMM 的 SM70 Marlin 内核。 |
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | FP8 expert GEMM 的接口约定，以及本仓库对标的 Flash-Next 公开数字。 |
| TurboMind（[LMDeploy](https://github.com/InternLM/lmdeploy)） | `fp8_moe` 绑定的预编译 SM70 block-FP8 expert GEMM。 |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | SM70 Marlin 扩展的构建依赖（v4.2.1）。 |

## 性能

Qwen3.8-Flash-Next 与 Qwen3.8-27B，V100-SXM2-32GB，单节点，输出 128 token 时的
decode 吞吐（tok/s），数据来自 `scripts/bench.py`。一行数据只有同时给出模型、精度、
卡数和采样方式才有意义，不同轴之间不可比较。

| 模型 | 精度 | 卡数 | 并行 | MTP | 采样 | 提示长度 | Decode | 来源 |
|---|---|---|---|---|---|---|---|---|
| Flash-Next | FP8 | 8 | TP8 / EP8 | 关 | 默认 | 76 | 40.0 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| Flash-Next | FP8 | 8 | TP8 / EP8 | 3/4 | temp 1, top-p 0.95, top-k 20 | 74 | 73.9 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| Flash-Next | FP8 | 8 | TP8 / EP8 | 3/4 | temperature 0 | 74 | 98.3—106.2 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| Flash-Next | NVFP4 | 4 | TP4，无 EP | 3/4 | temperature 0 | 993 | 87.27 / 87.24 | receipt `evidence-6294562-nvfp4` |
| Flash-Next | NVFP4 | 4 | TP4，无 EP | 3/4 | temperature 0 | 3473 | 72.43 / 90.68 | receipt `evidence-6294562-nvfp4` |
| Flash-Next | NVFP4 | 4 | TP4，无 EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 74 | 81.83 / 88.06 | receipt `evidence-6294562-nvfp4` |
| Flash-Next | NVFP4 | 4 | TP4，无 EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 4027 | 87.59 / 86.97 | receipt `evidence-199-ef2662a` |
| Flash-Next | NVFP4 | 4 | TP4，无 EP | 3/4 | temperature 0 | 3481 | 81.47 / 89.22（kernel 关 / 开，400 个 greedy token，decode graph 开，8 次中位数，去掉 warmup） | receipt `evidence-07c26a2`（PR #202） |
| 27B | FP8 | 2 | TP2，无 EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 8037 | 69.30 / 63.24 | receipt `evidence-e5a5740-draft-extend` |
| 27B | FP8 | 2 | TP2，无 EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 31109 | 44.59 / 45.79 | receipt `evidence-e5a5740-draft-extend` |
| 27B | FP8 | 2 | TP2，无 EP，40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 8037 | 73.33 / 66.43 | receipt `evidence-153a4c2-kv-splits` |
| 27B | FP8 | 2 | TP2，无 EP，40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 31158 | 54.42 / 47.13 | receipt `evidence-153a4c2-kv-splits` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 8030 | 67.11 / 65.54 | receipt `evidence-7e15a2b-fp8-draft-extend` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 32112 | 54.39 / 46.69 | receipt `evidence-7e15a2b-fp8-draft-extend` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, top-p 0.95, top-k 20 | 8034 | 80.01 / 66.98 | receipt `evidence-5097450-spec-steps` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, top-p 0.95, top-k 20 | 32138 | 61.62 / 46.39 | receipt `evidence-5097450-spec-steps` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temperature 0 | 4023 | 80.62 / 85.07 | receipt `evidence-d0054cfd-bench-spec` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temperature 0 | 32138 | 61.57 / 69.99 | receipt `evidence-d0054cfd-bench-spec` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8041 | 107.1 / 106.4（2 请求，512 输出 token，合计） | receipt `evidence-e1924f3-serving-active` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8041 | 181.8 / 181.3（4 请求，512 输出 token，合计） | receipt `evidence-e1924f3-serving-active` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31372 | 80.7 / 82.6（2 请求，512 输出 token，合计） | receipt `evidence-e1924f3-serving-active` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31372 | 117.8 / 125.4（4 请求，512 输出 token，合计） | receipt `evidence-e1924f3-serving-active` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8037 | 75.76 / 68.45（1 请求，512 输出 token） | receipt `evidence-8b7aaef-inner-loop` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8041 | 75.83 / 71.60（1 请求，512 输出 token） | receipt `evidence-175a630-e5m2-shift` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8041 | 75.84 / 76.97（1 请求，512 输出 token） | receipt `evidence-79aef7a-vec-gather` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8041 | 120.26 / 122.89（2 请求，512 输出 token，合计） | receipt `evidence-e3b28bb-fp8-lm-head` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 8041 | 212.12 / 200.68（4 请求，512 输出 token，合计） | receipt `evidence-e3b28bb-fp8-lm-head` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31372 | 66.22 / 55.34（1 请求，512 输出 token） | receipt `evidence-8b7aaef-inner-loop` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31358 | 60.93 / 70.17（1 请求，512 输出 token） | receipt `evidence-175a630-e5m2-shift` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31372 | 72.51 / 72.35（1 请求，512 输出 token） | receipt `evidence-79aef7a-vec-gather` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31372 | 95.71 / 97.68（2 请求，512 输出 token，合计） | receipt `evidence-e3b28bb-fp8-lm-head` |
| 27B | FP8，e5m2 KV，fp8 head | 2 | TP2，无 EP，40 KV splits | 5/6 | temp 1, default | 31372 | 135.40 / 131.73（4 请求，512 输出 token，合计） | receipt `evidence-e3b28bb-fp8-lm-head` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits，128k 上下文 | 3/4 | temp 1, top-p 0.95, top-k 20 | 127463 | 28.71 / 30.06 | receipt `evidence-dccc907-long-context` |
| 27B | FP8，e5m2 KV | 2 | TP2，无 EP，40 KV splits，256k 上下文 | 3/4 | temp 1, top-p 0.95, top-k 20 | 254914 | 17.60 / 18.84 | receipt `evidence-dccc907-long-context` |

27B 行需要 `SGLANG_V100_DRAFT_EXTEND_NATIVE=1` 与
`--speculative-draft-attention-backend tilelang_fa_v100`。缺少这两项时，
同一台机器在 31.7k 上测得 17.9 tok/s（issue #75）。

40 splits 的行是独立配置：它加了 `--triton-attention-num-kv-splits 40`，把 draft 解码内核从 16 个 CTA 提到 80 个。代价见配方的 log-probability 一节；这些行不能当作「只改了代码」与默认行比较。

fp8 head 的行是 e5m2 启动加上组件列表里的 `fp8_lm_head`，LM head 由 f16 改成块 FP8。这些行不能与上面的 serving 行比较：那批数据在 bench 于 1 以上并发改用共享前缀之前取得，其窗口仍包含 prefill。

e5m2 的行是同一套启动参数加上 `--kv-cache-dtype fp8_e5m2`，KV 字节减半。这些行需要 #129 的 draft-extend 路由修复：修复前，fp8 cache 会把 draft extend 送去 Triton extend 内核，32k decode 掉到 36.4 tok/s。本次测量中 32k 与 fp16 行持平，8k 低 5-10%；prefill 不变。
两个长上下文行需要 `--context-length 262144 --mem-fraction-static 0.82`：此时池为 278,432 个 token；
0.85 能分到 303,232，但图捕获后只剩 5.34 GB，在已被其他租户占用部分显存的卡上不足以完成首次生成；
0.80 只有 257,120 个 token，装不下 262k 请求。decode 随前缀增长而下降（32k 为 54，128k 为 29，256k 为 18），
因为每个 draft 行与 verify 行都要读完整前缀。
MTP 5/6 的行是同一套 e5m2 启动加上 `--speculative-num-steps 5` 与 `--mem-fraction-static 0.80`；
八次新测中五步的均值 62.87、中位数 63.40，3/4 为 58.41 / 58.56（#140）。32k 行两个值即 61.62 与 46.39。

读法：

- decode 吞吐等于「每步接受的 token 数 / 每步耗时」。贪心行与采样行的接受长度不同，不得互相比较。
- 不同日期的行，差异既来自代码也来自主机状态。同一台机器上，`6294562` 在 16:00 测得 87.27 tok/s，
  一小时后 `303e162` 测得 70.32 tok/s。两个 head 在同等条件下的背靠背 A/B（pull request #96）显示
  commit 之间最多约 10% 的差异，因此那 19 tok/s 中剩下的部分来自主机状态，原因未确认。
- prefill tok/s 只有在前缀缓存未命中时才是 prefill 速率。`scripts/bench.py`
  默认给每个请求加盐（`--fresh`）；不加盐时 radix cache 会命中前缀，
  `prompt_tokens / TTFT` 在 V100 的 3.5k prompt 上会报出 12-14k。
  引用 prefill 数字时必须注明是冷前缀测量（#95）。
- FP8 行是 8 卡，NVFP4 行是 4 卡；可比较的量是单卡吞吐，不是整行总和。
- `evidence-175a630-e5m2-shift` 与 `evidence-79aef7a-vec-gather` 四行与 `8b7aaef` 行是同一个启动配置，差别在 verify kernel 内部：FP8 查表换成位移，随后 KV gather 加宽到每次读 4 字节。31.5k 处稳定 step 时间中位数为 57.1、53.1、50.9 ms；单次 512 token 的运行分辨不出这些差异，表中 32k 的两组在本行与 `175a630` 之间也有重叠。
- 新的签名 receipt 落地时更新此表，且只收录能引用 receipt 或 recipe 的数字。没有来源的行不得写入。

哪些配置已验证、哪些仍在计划中，见 [`COMPATIBILITY.md`](COMPATIBILITY.md)：每种硬件与栈一行，每行都带收据或阻塞它的 issue。

## 快速开始

```bash
pip install sglang-restomod
SGLANG_PLUGINS=restomod python -m sglang.launch_server --help
```

开发环境：

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install -e . --no-deps
```

有 patch 时，改为从固定源码构建：

```bash
scripts/build.sh
```

## 目录

```
pyproject.toml          本层的打包配置；引擎 pin 在 UPSTREAM_PIN.toml
sglang_restomod/        插件、overrides、选入的组件
patches/                上游文件修改 + manifest
recipes/                已验证的启动配置
scripts/                fetch、patch、build、check、bump
UPSTREAM_PIN.toml       上游基线
```

## 分支

| 分支 | 含义 |
|---|---|
| `main` | 轻量层本体 |
| `upstream` | 固定 pin 的 pristine SGLang 树；仅作参考，永不合并 |
| `bump/upstream-*` | CI 打开的 pin bump |

组件选入和 patch 退役规则见 [RESTMOD.md](RESTMOD.md)，patch 生命周期见 [patches/README.md](patches/README.md)。

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。编码 agent 的说明在 [AGENTS.md](AGENTS.md)。

## 许可证

Apache-2.0。SGLang 和任何选入组件保留各自许可证；见 [THIRD_PARTY.md](THIRD_PARTY.md)。
