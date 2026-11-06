# Optimising Attention on a CPU

CS6886 Systems for Deep Learning, Assignment 5 (IIT Madras)
Aravindh Sriram Kumar A G (CS26S020)

This repository contains single-head attention, `O = softmax(QKᵀ/√d) V`, in FP32, optimised step by step in plain C on an AMD Ryzen 7 5800X. It uses no BLAS and no math library. Every variant does the same useful work of `4N²d` FLOPs. Full answers to Q1–Q5 are in [`report.pdf`](report.pdf).

## Machine

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5800X (Zen 3), 8 cores / 16 threads |
| Clock under load | 4.83 GHz (1 core), 4.66 GHz (8 cores) |
| SIMD | AVX2 + FMA3, 256-bit (8 FP32 lanes), 2 FMA units per core |
| Caches | L1d 32 KiB/core, L2 512 KiB/core, L3 32 MiB shared |
| Memory | 32 GB DDR4-3200, single channel |
| Compiler | clang 14, `-O3 -march=native` |

| | 1 core | 8 cores |
|---|---|---|
| Peak FP32, theoretical / measured (GFLOP/s) | 154.7 / 152.4 | 1192 / 1179.3 |
| DRAM bandwidth, theoretical / measured (GB/s) | 25.6 / 16.1 | 25.6 / 15.2 |
| Ridge point, measured (FLOP/byte) | 9.5 | 77.6 |

## Variants

| File | What it does |
|---|---|
| `attn_naive.c` | Given baseline: scalar loops, stores the full N×N score matrix |
| `attn_loops.c` | Loop reordering so the compiler can vectorise the inner loops |
| `attn_tiled.c` | Register tiles (4×16 and 5×16) and cache blocking |
| `attn_fused.c` | Score, softmax and output computed per block of rows; N×N matrix never stored |
| `attn_best.c` | Fused and tiled, plus a hand-written AVX2 exponential and vectorised softmax |
| `attn_omp.c` | `best` parallelised over row blocks with OpenMP |

## Results

Time per call in ms (speedup over `naive` in brackets). `omp` uses 8 threads, one per core.

| Variant | N=1024, d=64 | N=2048, d=64 | N=4096, d=64 | N=4096, d=128 |
|---|---|---|---|---|
| naive | 85.068 (1.0×) | 350.830 (1.0×) | 1441.053 (1.0×) | 3438.837 (1.0×) |
| loops | 11.505 (7.4×) | 46.994 (7.5×) | 222.167 (6.5×) | 337.870 (10.2×) |
| tiled | 5.242 (16.2×) | 20.814 (16.9×) | 113.522 (12.7×) | 148.317 (23.2×) |
| fused | 5.505 (15.5×) | 21.271 (16.5×) | 85.197 (16.9×) | 117.638 (29.2×) |
| best | 2.589 (32.9×) | 10.345 (33.9×) | 41.135 (35.0×) | 77.091 (44.6×) |
| **omp (8)** | **0.437 (194.7×)** | **1.527 (229.8×)** | **5.943 (242.5×)** | **10.942 (314.3×)** |

Key points:

- **naive** reaches only 2% of peak. Each output is one long scalar sum, and every multiply–add waits 4 cycles for the previous one.
- **loops** lets the compiler use 8-wide vector instructions (6.5–10×). Partial sums still go through memory.
- **tiled** keeps blocks of outputs in registers, so each loaded value is reused many times (about 2× over `loops`).
- **fused** cuts temporary memory from 65 MiB to 2.3 MiB. It is faster only at N=4096, where the score matrix no longer fits in the L3 cache.
- **best** makes the softmax 9× faster with a vectorised exponential. It reaches 104 GFLOP/s, 68.5% of single-core peak.
- **omp** scales almost linearly up to 4 threads. 8 threads give 6.9–7.1× at N=4096, which is 722.7 GFLOP/s (61% of the 8-core peak). 16 threads (SMT) do not help, because one thread already keeps both FMA units busy.

`make check` passes for every variant on every input. The largest error is 6.1×10⁻⁶, against the 10⁻⁴ limit.

## Roofline (N=4096, d=64)

| Variant | GFLOP/s | Arithmetic intensity (FLOP/byte) | % of compute roof |
|---|---|---|---|
| naive | 2.98 | 15.75 | 2.0 |
| loops | 19.33 | 15.63 | 12.7 |
| tiled | 37.83 | 15.63 | 24.8 |
| fused | 50.41 | 682.67 | 33.1 |
| best | 104.41 | 682.67 | 68.5 |
| omp (8) | 722.69 | 682.67 | 61.3 |

Fusion raises the arithmetic intensity from about 16 to `N/6 ≈ 683` FLOP/byte, because the 64 MiB score matrix no longer goes through DRAM. All variants are compute-bound on one core. Their remaining limits are inside the core. See [`plots/roofline.png`](plots/roofline.png).

## How to run

```sh
make naive      # build and time one variant on every input in input/
make all        # all variants
make check      # compare every variant against naive (tolerance 1e-4)
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads make omp
```
