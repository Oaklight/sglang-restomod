# Provenance: attention/lsa_sm70

## Source

| Field | Value |
|---|---|
| Repository | `Oaklight/lsa-sm70` |
| Commit | `3a4798093224933988175ee685084d521474da0b` |
| Version | `0.1.0` |
| License | Apache-2.0 |
| Upstream reference | `Prefrontal-AI/latent-sparse-attention` @ `97b82d695b74ca341965a16bbc2a951c8b0c567a`, Apache-2.0 |
| Paper | arXiv 2610.42147, *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention* |

`Oaklight/lsa-sm70` is the only implementation. Its `PROVENANCE.md` records the
file-level selection from the reference: the portable selection core, the
serving path written against it, and what was deliberately left out (the
Hopper/Blackwell CuTeDSL kernels, bucket shapes, the fixed/tree hierarchy, the
MLA route and training).

## What is here

Only the registration shim:

| file | what it does |
|---|---|
| `__init__.py` | `register()` imports `lsa_sm70` and delegates to the package's own `register()` |

No selection, Triton program, read or backend code is vendored in this
repository. The shim imports nothing heavy: the package loads only when
`register()` runs, and the package itself defers torch, triton and sglang to
first use.

The backend name stays `lsa_sm70`, so a launch that selects
`SGLANG_RESTOMOD_COMPONENTS=lsa_sm70` and the `launch_check` component check
behave exactly as before. Only the implementation source moved.

## Pin

`UPSTREAM_PIN.toml` carries the `[lsa]` section with the repository, commit and
version. `scripts/check_pin.py` compares the installed `lsa_sm70.__version__`
with the pin when the package is present, and skips when it is absent, because
the component is optional.

Install the pinned commit:

```sh
pip install 'lsa-sm70 @ git+https://github.com/Oaklight/lsa-sm70@3a4798093224933988175ee685084d521474da0b'
```

## Status

The package's README is the status of record. At this pin it is alpha: greedy
output is token-identical to the dense `tilelang_fa_v100` baseline on 32 tokens
at 2,048 and 16,384 prompt tokens; CUDA graph capture is refused because the
selection is host-orchestrated; speculative decoding, DP attention, context
parallel and an fp8 KV cache are not supported.
