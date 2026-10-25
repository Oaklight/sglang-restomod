# Engineering plan

The layer turns validated V100 behaviour into buildable, testable components. It
does not carry a fork.

## Layout

```
sglang_restomod/
├── plugins.py              component registry and selection
├── attention/v100/         FA-V100 backend and CUDA graph hooks
├── quant/sm70_fp8_moe/     block-FP8 expert GEMM binding
└── kvcache/                E5M2 write/partial hooks
tests/                      CPU unit tests and GPU-marked checks
recipes/                    validated launch profiles
patches/                    upstream edits only
```

## Rules

1. A capability that upstream lacks becomes a component, not a patch.
2. A patch exists only to change a file that upstream already has.
3. Components are selected by name through `SGLANG_RESTOMOD_COMPONENTS`;
   unset means baseline.
4. Component imports are cheap; kernels load on first use.

## Definition of done per component

- `PROVENANCE.md` with repository, commit, license and selected files;
- a CPU-runnable unit test;
- a GPU-marked numeric or smoke test (`pytest -m gpu`, `scripts/test_gpu.sh`);
- a `recipes/` entry and a signed receipt;
- a `COMPATIBILITY.md` row.

## Work packages

P0 (no GPU required):

1. Registry and launch-profile selection (done).
2. FP8 MoE component: loader, build cache, tests (done).
3. Attention component scaffold with the graph metadata fix (scaffolded).
4. Test layout, CPU CI step, GPU runner script (done).

P1 (hardware-dependent):

5. Wire `Fp8MoEMethod` to the component (#20) and run Flash-Next FP8 end to end.
6. Fix or disable the E5M2 partial decode kernel (#51).
7. Publish the graph-on 27B recipe with a receipt.
8. Extract the attention backend itself (E5.1) and delete the lab patch.
