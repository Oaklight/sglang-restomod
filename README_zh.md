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

目前有两个组件，都通过 `SGLANG_RESTOMOD_COMPONENTS` 显式启用：

| 组件 | 作用 |
|---|---|
| `fp8_moe` | sm70 上的 block-FP8 expert GEMM。SGLang 随包提供的 SM70 TurboMind 库导出了 `fp8_moe_gemm_sm70_out`，但从未向 torch 注册；本组件用一个小的 `dlopen`/`dlsym` shim 解析该符号并暴露类型化接口。官方 block-FP8 MoE 权重能在 V100 上跑，靠的就是它。 |
| `attention_v100` | 在 V100 attention backend 上实现 `init_forward_metadata_out_graph`。graph runner 会调用它，而基类留成了 no-op；没有它，decode 的 CUDA graph 根本捕获不起来。 |

```bash
SGLANG_RESTOMOD_COMPONENTS=fp8_moe,attention_v100
```

不设置这个变量时插件保持惰性。patch 队列仍为空（`patches/manifest.json`），
所以这一层不修改上游文件，全部通过插件入口生效。

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
