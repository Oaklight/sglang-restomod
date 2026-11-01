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

目前有六个组件，都通过 `SGLANG_RESTOMOD_COMPONENTS` 显式启用：

| 组件 | 作用 |
|---|---|
| `fp8_moe` | sm70 上的 block-FP8 expert GEMM。SGLang 随包提供的 SM70 TurboMind 库导出了 `fp8_moe_gemm_sm70_out`，但从未向 torch 注册；本组件用一个小的 `dlopen`/`dlsym` shim 解析该符号并暴露类型化接口。官方 block-FP8 MoE 权重能在 V100 上跑，靠的就是它。 |
| `fp8_dense` | 固定 pin 引擎的 SM70 W8A16 稠密 FP8 线性层内核。 |
| `attention_v100` | 在 V100 attention backend 上实现 `init_forward_metadata_out_graph`。graph runner 会调用它，而基类留成了 no-op；没有它，decode 的 CUDA graph 根本捕获不起来。 |
| `fp4_moe` | 经 `marlin_v100` 扩展实现的 NVFP4 expert GEMM：加载期 repack、SM70 scale 编码、GEMM 桥接。SGLang 随包的 Marlin 内核在 SM80 以下是空实现。 |
| `ple_sm70` | fp8 PLE host-embedding gather 的 SM70 版本。固定 pin 的内核会 cast 到 `tl.float8e4nv`，在 SM70 上无法编译。 |
| `spec_sm70` | top-k/top-p renorm 替换，使非贪心 EAGLE verify 不再触到 FlashInfer 的 sm75 下限。 |

```bash
SGLANG_RESTOMOD_COMPONENTS=fp8_moe,attention_v100
```

不设置这个变量时插件保持惰性。patch 队列仍为空（`patches/manifest.json`），
所以这一层不修改上游文件，全部通过插件入口生效。

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

Qwen3.8-Flash-Next，V100-SXM2-32GB，单节点，输出 128 token 时的 decode 吞吐（tok/s），
数据来自 `scripts/bench.py`。一行数据只有同时给出精度、卡数和采样方式才有意义，
不同轴之间不可比较。

| 精度 | 卡数 | 并行 | MTP | 采样 | 提示长度 | Decode | 来源 |
|---|---|---|---|---|---|---|---|
| FP8 | 8 | TP8 / EP8 | 关 | 默认 | 76 | 40.0 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| FP8 | 8 | TP8 / EP8 | 3/4 | temp 1, top-p 0.95, top-k 20 | 74 | 73.9 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| FP8 | 8 | TP8 / EP8 | 3/4 | temperature 0 | 74 | 98.3—106.2 | `task.md` |
| NVFP4 | 4 | TP4，无 EP | 3/4 | temperature 0 | 993 | 87.27 / 87.24 | receipt `evidence-6294562-nvfp4` |
| NVFP4 | 4 | TP4，无 EP | 3/4 | temperature 0 | 3473 | 72.43 / 90.68 | receipt `evidence-6294562-nvfp4` |
| NVFP4 | 4 | TP4，无 EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 74 | 81.83 / 88.06 | receipt `evidence-6294562-nvfp4` |

读法：

- decode 吞吐等于「每步接受的 token 数 / 每步耗时」。贪心行与采样行的接受长度不同，不得互相比较。
- 不同日期的行，差异既来自代码也来自主机负载。同一 commit 在同一台机器上相隔一小时测到 87.27 与 70.32 tok/s，
  prefill 分别是 3.6k 与 1.4k tok/s。支撑这个差异的背靠背 A/B 在 pull request #96 中。
- FP8 行是 8 卡，NVFP4 行是 4 卡；可比较的量是单卡吞吐，不是整行总和。
- 新的签名 receipt 落地时更新此表，且只收录能引用 receipt 或 recipe 的数字。没有来源的行不得写入。

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
pyproject.toml          sglang-restomod；固定 sglang==0.5.22
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
