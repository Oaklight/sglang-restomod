# 2026-11-06 — Flash-Next hyper-connection mixer under block FP8

Status: the mechanism works and does not pay for itself. Decode CUDA-graph
capture survives with the mixer routed through its converted linears, and the
launch that needs that routing costs **+2.7 ms/step (+10%)** against the same
launch without the dense-FP8 conversion. The component stays off by default.

## What fails without a fix

`fp8_dense_quant` converts the mixer pair `[320, 10240]` / `[10240, 320]`
through zero padding, because 320 misses the block-128 rule. `GatedResidual.mix`
does not call the modules: it passes `input_mix_weight_down.weight` and
`input_mix_weight_up.weight` into `torch.compile(_mix_compute)`, which calls
`F.linear` on the raw tensor. Capture then dies:

```
Capture cuda graph failed: Dynamo failed to run FX node with fake tensors:
call_function <built-in function linear>(*(FakeTensor(..., size=(4, 10240),
dtype=torch.float16), Parameter(FakeTensor(..., size=(10240, 384),
dtype=torch.uint8))))
  File ".../sglang/srt/layers/hyperconnection.py", line 206, in _mix_compute
```

The pair is 97 + 97 calls per decode step per rank.

## The override

`sglang_restomod/quant/sm70_hc_mix` (component `hc_mix_fp8`) wraps
`GatedResidual.__init__` and replaces `_mix_compute` on each instance with a
plain-Python version that calls the two modules. The replacement is
deliberately not compiled: the pin's compiled closure is the thing that cannot
take fake tensors. While the linears are unconverted the wrapper calls the
pin's function unchanged, so an unconverted launch is unaffected.

Two install-time guards, both added after they caught real changes:

1. The layout check reads `inspect.getsource(cls)`, not `cls.__init__`.
   `overrides.py` installs `_install_hyperconnection_fp16` before components
   load, so `__init__` is already a closure that contains none of the mixer
   lines. Reading it rejected a correct pin.
2. The converted/unconverted choice is made on the first `mix` call, not at
   `__init__`. `fp8_dense_quant` converts after `load_model`, so at
   construction both linears are still `nn.Linear`; deciding there left the
   override inert and capture still failed, with no marker logged.

## Measurement

4× V100-SXM2 on one island (lambda4, GPUs 4-7), TP4, `--mem-fraction-static
0.80`, MTP 3/1/4, 8192 context. One `/generate` request per rep: 3294-token
prompt, 256 output tokens, greedy, `stream=True`, timings and
`meta_info.spec_verify_ct` from the same request. Step time is
`decode_s / spec_verify_ct`, so it does not depend on how accept is estimated.

| Arm | Components | Step ms (reps) | Median | Decode |
|---|---|---|---:|---:|
| A | `...,fp8_dense` | 26.75 / 27.13 / 26.89 / 27.13 | 27.13 | 143-145 tok/s |
| C | `...,fp8_dense,fp8_dense_quant,hc_mix_fp8` | 29.79 / 29.80 / 29.81; repeat 29.26 / 29.80 / 29.79 | 29.80 | 130 tok/s |

Both arms reported accept 3.879 and 66 verify passes on the same prompt, so
acceptance does not explain the gap. Arm C logs `hc mixer override active` on
all four ranks and its decode lines report `cuda graph: True`.

The delta is conversion plus routing together, not the mixer alone: the arm
that would separate them (conversion without the override) cannot capture, so
it is not a measurement.

A supporting `scripts/bench.py` run (8 reps, 972 and 3403 prompt tokens) points
the same way, but this host's rep-to-rep spread was 85-142 tok/s on one arm, so
those numbers are not quoted.

## Verdict

The dense shapes are launch-bound on SM70; converting them buys nothing and the
routing costs 2.7 ms/step. `hc_mix_fp8` is a working mechanism that does not
pay for itself here. It stays selectable, off by default, with CPU tests, GPU
tests and the guards above.
