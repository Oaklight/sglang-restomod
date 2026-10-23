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

目前只有基线：`sglang==0.5.22`、空 patch 队列、no-op 插件。V100 支持会随组件逐个加入；在那之前这一层本身不做任何事。

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
