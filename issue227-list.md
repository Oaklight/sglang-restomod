## 未被 `fp8_dense_quant` 覆盖的 dense linear（#219 launch 日志）

`fp8_dense_quant` 在 #219 的 launch 里已经打印了 admission 报告（TP3、target model、48 层）：

```
49 dense linears quantized to block FP8, 0 refused; admission by type:
{'VocabParallelEmbedding': {'not-a-linear': 2},
 'QKVParallelLinear': {'shape:864x1152': 27, 'admitted': 12},
 'RowParallelLinear': {'not-a-linear': 151},
 'ColumnParallelLinear': {'shape:1076x1152': 27, 'admitted': 1},
 'MergedColumnParallelLinear': {'admitted': 36, 'shape:24x2560': 36, 'shape:320x2560': 48},
 'ReplicatedLinear': {'not-a-linear': 62},
 'Linear': {'shape:1x2560': 48, 'shape:320x10240': 97, 'shape:10240x320': 97, 'shape:4x10240': 96}}
```

MTP draft（1 层）是另一条 `3 dense linears quantized`，形状同类，量很小。

### 分类

| 组 | 数量 | 每 rank 形状 | 拒绝原因 | 现在可修? | bf16 字节/step |
|---|---:|---|---|---|---:|
| RowParallelLinear | 151 | 含 o_proj/out_proj `[2560,1536]`、shared down_proj `[2560,160]` 等 | `not-a-linear`：谓词只认 `output_size_per_partition`，Row 没有该属性 | **是**（o_proj/out_proj 形状合格；down_proj K=160 不合格） | o_proj/out_proj 48×7.86MB=377MB |
| ReplicatedLinear | 62 | PLE key/value、QSA key/value 等 | 同上 | 形状合格，但要先确认 router 是否在此组、是否走融合 kernel | 待 instrumented run |
| Linear（hc low-rank） | 97+97 | `[320,10240]` / `[10240,320]` | `shape:` N 或 K=320，不是 128 的倍数 | 需要 block-64 kernel（issue 非目标） | 1.27GB |
| MergedColumnParallelLinear | 48 | `[320,2560]` | N=320 | 需要 block-64 | 79MB |
| QKVParallelLinear | 27 | `[864,1152]` | N=864 | 需要 block-32 或 padding | 54MB |
| ColumnParallelLinear | 27 | `[1076,1152]` | N=1076 | 需要 padding（1076 不是 32 的倍数） | 67MB |
| MergedColumnParallelLinear | 36 | `[24,2560]` | N=24 | 不值得 | 4MB |
| Linear | 96 | `[4,10240]` | N=4 | 不值得 | 8MB |
| Linear | 48 | `[1,2560]` | N=1 | 不值得 | 0.2MB |
| experts / lm_head / embedding | — | — | 设计排除 | 否 | — |

Row 的 151 里还有约 100 个模块的精确形状，要靠 instrumented run 打出来；上表的 o_proj/out_proj 和 shared down_proj 是能从模型代码确认的部分。

### 根因：谓词写窄了，不是形状限制

`_is_linear_base` 检查 `quant_method` + `output_size_per_partition`，但该属性只有 `ColumnParallelLinear.__init__` 设置（`linear.py:480`）。`RowParallelLinear` 和 `ReplicatedLinear` 都是 `LinearBase`，没有这个属性，于是 `refusal_reason` 返回 `not-a-linear`；`convert()` 还会因为 `plain=True` 把 forward 换成 `_plain_linear_forward`，对 Row 会丢掉 all-reduce。正确的修法是 `isinstance(module, LinearBase)`，让 `plain=False`、保留模块自己的 forward（它本来就走 `quant_method.apply`）。

### 取舍

1. **谓词修复是免费的**：改一行 predicate，o_proj/out_proj 的 48 个 `[2560,1536]` 可以转。按字节减半的上限，预计 **~0.2—0.4 ms/step**，低于 1.6—2.5 ms 的目标。
2. **1.6—2.5 ms 的大头在 320-wide hc low-rank（1.27GB）**，它需要 block-64 kernel。这是改 kernel，不是 admission/dispatch，和本 issue 的 non-goal 冲突。
3. 所以建议先做谓词修复的 A/B（小、风险低、可直接量出实际节省）；如果要把 1.6—2.5 ms 作为目标，需要把 block-64 路径纳入范围，或者把目标下调到 0.2—0.4 ms。

我的默认下一步：谓词修复 + instrumented run（把 151/62 的名字和形状打全）+ 同机同 launch 的 A/B，先量 o_proj/out_proj 的真实节省；block-64 等你决定。
