# QPN2 NVFP4 weight layout (SM70)

Single definition for the repack (9b), the decode GEMV (9b), and the prefill
GEMM reader (df). Derived from 1Cat-vLLM `csrc/sm70_turbomind/ops/
nvfp4_qpn2_layout.cuh` and the prepack/reader in `nvfp4_qpn2_sm70.cu`.
Do not invent a second layout.

## 1. Buffer

For a weight tensor of shape `[N, K]` (`N % 32 == 0`, `K % 16 == 0`):

```
groups = K / 16
tiles  = N / 32
bytes  = tiles * groups * 288
```

Group `(tile t, group g)` is exactly **288 bytes**:

```
[  0, 256) : 32 lanes x 8 code bytes, lane-major:  lane*8 + byte
[256, 288) : 32 lane scale bytes:                  256 + lane
```

Addresses:

```
code byte b (0..7), storage lane l (0..31):
    off = ((t*groups + g) * 288) + l*8 + b
scale byte:
    off = ((t*groups + g) * 288) + 256 + l
```

The storage lane is **not** the logical column. The column of storage lane `l`
inside its 32-column tile is

```
qpn2_col_from_lane(l) = ((l >> 2) & 3) * 8 + (l & 3) + ((l & 16) ? 4 : 0)
column n = t*32 + qpn2_col_from_lane(l)
```

Inverse (logical column `c = n % 32` to storage lane):

```
l = ((c & 24) >> 1) | (c & 3) | ((c & 4) << 2)
```

Storage lane 0..31 maps to columns
`0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31`.
This order matches the HMMA fragment lane assignment. Both readers must use it;
do not relabel the lanes.

## 2. K order inside a lane

A lane's 8 code bytes hold 16 codes = the 16 K values of its column in one K16
group. Byte `b` holds physical codes `2b` (low nibble) and `2b+1` (high nibble).
Physical index `p` (0..15) is logical K

```
qpn2_logical_k(p) = (p & 8) + (((p & 7) & 3) << 1) + ((p & 7) >> 2)
logical K = g*16 + qpn2_logical_k(p)
```

The 16 values in storage order (nibble 0..15 of the two 32-bit words) are

```
logical K = [0, 2, 4, 6, 1, 3, 5, 7,  8, 10, 12, 14, 9, 11, 13, 15]
```

i.e. the nibble order `[0,2,4,6,1,3,5,7]` within each K8 half. **Both** the
decode GEMV and the prefill reader must consume the values in this storage
order and present the activation operand in the same order (or apply the
inverse permutation to exactly one side). If one side uses logical K 0..15 and
the other uses storage order, the dot product pairs the wrong K and the result
is wrong even though every byte is correct.

Simplification when expanding with the word trick in §3: the four half2 of one
32-bit word come out as logical pairs `(0,1) (2,3) (4,5) (6,7)` for word 0 and
`(8,9) ... (14,15)` for word 1. The permutation is absorbed by the expansion,
so a reader that uses the trick has no separate un-permute step and its
activation operand stays in plain logical K order.

## 3. Code values and scaling

Nibble `c` (0..15) is E2M1: bit 3 is the sign, magnitude
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`.

Exact dequant, matching 1Cat's FP16 RNE path (one 32-bit word -> four half2):

```c
const unsigned kSign = 0x80008000u, kExpMant = 0x0e000e00u;
v0 = ((w << 12) & kSign) | ((w <<  9) & kExpMant);
v1 = ((w <<  8) & kSign) | ((w <<  5) & kExpMant);
v2 = ((w <<  4) & kSign) | ((w <<  1) & kExpMant);
v3 = ( w        & kSign) | ((w >>  3) & kExpMant);
code[i]  = half2(v_i) * 16384.0h;      // 2^14, undo the FP4 exponent bias
out[i]   = code[i] * scale;            // scale applied after the bias removal
```

The group scale is one E4M3 byte per (column, K16 group). Its effective value:

```c
// exact E4M3 -> FP16
bits = ((byte & 0x80) << 8) | ((byte & 0x7f) << 7);
raw  = half(bits) * 256.0h;
// multiply-then-round: do NOT round the global factor first
scale = fp16_rne(float(raw) * global_scale);
```

`global_scale` is the per-expert ModelOpt `weight_scale_2` (FP32). The order
matters: round `raw * global_scale` once. 1Cat's `nvfp4_effective_scale` does
exactly this.

## 4. Repack (checkpoint -> QPN2)

Input: ModelOpt packed weight `[N, K/2]` uint8 (low nibble = even K, high nibble
= odd K), E4M3 scales `[N, K/16]`, and the per-expert FP32 global scale.

For every output code byte index `i` in `[0, tiles*groups*32*8)`:

```
physical_byte = i & 7
outer         = i >> 3
storage_lane  = outer & 31
outer       >>= 5
g             = outer % groups
t             = outer / groups
column        = t*32 + qpn2_col_from_lane(storage_lane)
k0            = g*16 + qpn2_logical_k(2*physical_byte)
k1            = g*16 + qpn2_logical_k(2*physical_byte + 1)
c0            = (weight[column, k0/2] >> ((k0 & 1)*4)) & 0xF
c1            = (weight[column, k1/2] >> ((k1 & 1)*4)) & 0xF
out[i]        = c0 | (c1 << 4)
```

For every `(t, g, storage_lane)`:

```
out[((t*groups + g)*288) + 256 + storage_lane] = scales[column, g]   // 0 if column >= N
```

This is a pure permutation of bytes: the buffer size is the same as the source
(0.5 B/weight code + 0.0625 B/weight scale = **0.5625 B/weight**). The global
scale is **not** baked into the buffer; the reader multiplies it in, as in §3.

The repack allocates the buffer zero-filled and writes every defined byte. A
padded column (`N % 32 != 0`) therefore has code bytes `0x00` and scale byte
`0`, so a coalesced load of a whole 288-byte group always reads defined data.
Zero codes decode to 0.0; the reader must still not emit padded columns.

## 4.1 Expert stacking (MoE)

Experts are contiguous, no header and no inter-expert padding:

```
expert e base = e * tiles * groups * 288
byte offset of (e, t, g, lane, code byte b)
    = ((e * tiles + t) * groups + g) * 288 + lane*8 + b
scale byte = that base + 256 + lane
```

`w13` 460,800 B/expert, `w2` 230,400 B/expert at TP4. The repack writes the
experts in this order and the reader indexes them with this stride.

## 5. Shapes

Per rank at TP4 for Qwen3.8-Flash-Next NVFP4:

| tensor | N | K | tiles | groups | bytes/expert |
|---|---:|---:|---:|---:|---:|
| `w13` gate_up | 320 | 2560 | 10 | 160 | 460,800 |
| `w2` down | 2560 | 160 | 80 | 10 | 230,400 |

`N % 32 == 0` and `K % 16 == 0` are required. For an `N` not divisible by 32,
pad the last column tile and write scale 0; the reader must not use the padded
columns.

## 6. What the prefill reader must not do

- Assume storage lane == logical column (use §1).
- Assume the 16 values are in logical K 0..15 order (use §2).
- Pre-multiply the global scale into the E4M3 byte or round it before the
  multiply (use §3).
- Interleave the scale bytes per lane with the code bytes (the scales are a
  separate 32-byte region at the end of each 288-byte group).

## 7. Verification

- CPU round trip: random packed weights + E4M3 scales -> QPN2 -> dequantized
  dense; compare bitwise to the source decode.
- GPU: GEMV vs an FP32 dequantized reference, relative error <= 1e-3 (the #224
  P1 probe).
- The prefill reader is accepted only if its output matches the GEMV/reference
  on a small `M > 1` case, not just `M == 1`.
