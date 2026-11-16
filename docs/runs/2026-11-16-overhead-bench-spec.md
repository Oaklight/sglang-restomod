# 2026-11-16 — The per-card overhead instrument: four measurements, and how to run them on a new model

The decomposition in [the overhead run](2026-11-16-per-card-overhead.md) was built on
Flash-Next. **Its numbers are Flash-Next's, not a property of the stack.** This page is
the instrument without the model: four measurements, the order to take them, and the
cross-check that decides whether the result may be believed.

Every number a capacity plan needs is one of these four. Nothing else has to be
measured, and nothing here depends on which model is loaded.

## What the four measurements are

| # | measurement | arms | what it yields |
|---|---|---|---|
| M1 | bare CUDA context | one process that only initialises CUDA, per card | the floor no server can go below |
| M2 | graph pool | served config twice, `--disable-cuda-graph` vs default | the CUDA graph capture cost |
| M3 | KV cost | two served arms whose KV token count differs | bytes of KV per token per rank |
| M4 | engine accounting cross-check | one steady-state reading of the served arm | the unattributed residual, and whether to trust it |

Fixed footprint per card = M1 + M2 + KV + named caches the log prints + M4 residual.
Expert capacity = `cards x (vram_decimal - dense_per_card - fixed)`, with
`vram_decimal = GiB x 1.073741824`.

## M1 — bare CUDA context

One process per card, in the same container image, on the same device list:

```python
import os, time, torch
torch.cuda.init()
held = [torch.zeros(1, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
torch.cuda.synchronize()
print("DEVICES", torch.cuda.device_count(), flush=True)
time.sleep(int(os.environ.get("HOLD", "30")))
```

Read `nvidia-smi --query-gpu=index,memory.used` before and after; the delta is the
context. **Take it on every rank and report the spread, not the mean** — an unequal
context across ranks means something else is already on the card.

Flash-Next reference: 366 MiB, identical on four ranks and on two hosts. If a new
measurement disagrees by more than ~30 MiB, check the image and the driver before
blaming the model.

## M2 — graph pool

Two arms, identical except for `--disable-cuda-graph`, one arm per window, each having
served a request before the reading. **Assert the KV token count is identical between
the two arms**; if it is not, the arms differ in more than the graphs and the delta is
not the pool. Report `used(graphs on) - used(graphs off)`.

Flash-Next reference: 716 MiB, at 0.82 mem-fraction with identical KV (172,032 tokens
both).

## M3 — KV cost per token

Two arms whose KV token count differs — either two `--mem-fraction-static` values or
two `--context-length` values that both fit — and read the engine's own line:

```
KV Cache is allocated. dtype: ..., #tokens: N, K size: A GB, V size: B GB
```

Slope = `(A+B of arm 2 - A+B of arm 1) / (N2 - N1)` = bytes per token per rank. Report
the slope, not a single KV figure: the KV number depends on how many tokens the engine
chose to admit, which is exactly what the fixed footprint determines. **Do not compute
bytes/token by dividing one arm's KV by its token count** — that mixes in the allocator's
own rounding.

## M4 — engine accounting cross-check

At steady state, with a request served:

1. `used` = mean of `nvidia-smi` used over the ranks.
2. `accounted` = `restomod-load-memory` reserved (weights) + KV bytes (M3, at this arm's
   token count) + `available_gpu_mem` inversion, or the direct sum:
   weights + KV + M1 context + M2 pool + every cache the log names.
3. `residual` = `used - accounted`, reported as a **sum**, never split by guess.
4. Cross-check: the engine prints `available_gpu_mem=... GB` and `avail_mem=... GB` at
   admission and at capture. `vram - used` and the engine's free-memory figure should
   agree to about 0.2 GB. **If they disagree, stop:** the accounting is wrong somewhere
   and the residual is not a number you can plan against.

The reason this cross-check exists is that Flash-Next's residual looked like a cache
until the log named one: `Mamba Cache is allocated ... conv 0.04, ssm 0.91,
intermediate_ssm 0.47, intermediate_conv 0.01 GB` = 1.43 GB, inside what had been
called unattributed. **Grep the log for every "Cache is allocated" line first**; only
what remains after naming them is the residual.

## Applying it to a model the instrument has not seen

The four measurements transfer; the terms inside them do not. Before reusing any
Flash-Next number, check that the model has the component at all:

- **mamba / SSM cache** — Flash-Next pays 1.43 GB per card for it. A model with no
  linear-attention layers pays zero. Check the log for the line; absence is the
  measurement.
- **KV shape** — Flash-Next has 1 KV head and 0.93 KB/token at fp16. MLA models pay a
  different, usually smaller, per-token cost, and compressed-KV models smaller still.
  Measure with M3; do not import the number.
- **Engram / embedding tables** — Flash-Next offloads PLE to host. A model with a large
  sparse lookup table pays **host** memory and I/O, not card memory. Report it
  separately and keep it out of the card budget entirely, or the plan double-counts.
  The asymmetry is deliberate: **host-side tables stay out of `fixed`, the drafter stays
  in.** One is a tier, the other is resident.
- **Drafter** — with speculative decoding the second model's static cost lands inside
  M4's residual unless it is named, and it *belongs* in the card budget, so it must be
  measured rather than excluded. Take a SPEC=0 arm on the new model specifically to size
  it: the difference between the two residuals is the drafter's static cost. Report
  which arm every residual belongs to.

## Invariants that make the numbers comparable

- One arm per window; restart the server between arms rather than reconfiguring.
- Discard the first request in every arm: the first call compiles Triton/TileLang
  kernels and inflates both time and, transiently, memory.
- The served configuration is the one to measure. A launcher flag left unset is a
  different model — record every export and say which ones were not set.
- Record tree identity with `git describe` in the SGLang tree, the repo commit, and the
  kernel `.so` hash. The banner lies; `git describe` does not.
- `nvidia-smi` attribution before every reading:
  `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory`. A co-tenant on the card
  invalidates the arm, and the mean over ranks will not tell you.
