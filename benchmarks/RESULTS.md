# Split-K GEMM: performance vs cuBLAS

Results for the parallel split-K path (`split_k_mode="parallel"`: per-split fp32
workspace partials + a separate deterministic reduce kernel, `gemm_splitk_reduce.py`)
against cuBLAS via `torch.bmm`.

## Setup

- **GPU**: NVIDIA B200 (148 SMs), bf16 inputs/outputs, fp32 accumulate.
- **Baseline**: `torch.bmm` (cuBLAS). For the K-heavy small-M/N shapes cuBLAS's own
  heuristic selects a **split-K** kernel (`nvjet_..._splitK_TNT` + a separate
  `cublasLt::splitKreduce_kernel`), confirmed via `benchmarks/inspect_cublas.py`. So
  for those shapes this is a **split-K vs split-K** comparison. For the large square
  (4096³) cuBLAS picks a plain (non-split) GEMM and split-K does not help.
- **Measurement**: `benchmarks/benchmark_gemm_splitk.py` (`triton.testing.do_bench`,
  warmup=5, iters=30), every point ref-checked against an fp32 `torch.bmm` (atol 3e-2
  for bf16). Tile 128×128, cluster 1×1, persistent. Clocks not locked (shared box), so
  absolute numbers carry a few-% run-to-run noise; ratios are stable.

## Parallel split-K vs cuBLAS (best over a `split_k ∈ {1,2,4,8,16,32}` sweep)

| shape (m×n×k) | output tiles | cuBLAS | best quack | best split_k | vs cuBLAS | vs split_k=1 |
|---|---|---|---|---|---|---|
| 128×128×16384 | 1  | 20.2 TF / 26.6 µs | 16.5 TF / 32.6 µs | 16 | **0.82×** | 2.73× |
| 128×128×65536 | 1  | 58.3 TF / 36.9 µs | 49.9 TF / 43.0 µs | 32 | **0.86×** | 7.42× |
| 256×256×32768 | 4  | 110.6 TF / 38.8 µs | 109.7 TF / 39.1 µs | 16 | **0.99×** | 4.30× |
| 512×512×16384 | 16 | 233.7 TF / 36.8 µs | 209.7 TF / 41.0 µs | 8 | **0.90×** | 2.24× |
| 4096×4096×4096 | 1024 | 826.0 TF / 166 µs | 709.7 TF / 194 µs | 1 | 0.86× | 1.00× |

The last row uses `split_k=1` (split-K is counter-productive once the GPU is already
full); its 0.86× reflects the dense mainloop, not split-K.

## Parallel vs serial split-K (best vs cuBLAS)

The earlier serial path (in-kernel turnstile reduction) is kept as
`split_k_mode="serial"` for reference. Parallel is uniformly better and scales with
`split_k` instead of plateauing:

| shape | serial best | parallel best |
|---|---|---|
| 128×128×16384 | 0.30× | **0.82×** |
| 128×128×65536 | 0.18× | **0.86×** |
| 256×256×32768 | 0.29× | **0.99×** |
| 512×512×16384 | 0.40× | **0.90×** |

## How we got here

1. **Parallel split-K mode** (per-split workspace + separate reduce) replacing the
   serial turnstile — scales with `split_k` and removes the high-`split_k` blow-up.
2. **Parallelized the reduce** over `(output tile, chunk)` — it had been one CTA per
   tile (266 µs at 128²×16384 sk=32); now ~µs-scale.
3. **Unrolled the reduce accumulation + smaller reduce CTAs** — the reduce was
   latency-bound (one outstanding load per thread); unrolling keeps many loads in
   flight and spreading across more SMs adds bandwidth at high `split_k`. This made a
   higher `split_k` worthwhile, which in turn fills the GEMM more.

## Remaining gap

On the single-output-tile shapes (128² rows, 0.82–0.86×) the bottleneck is now the
**GEMM**, not the reduce: at the optimal `split_k` it runs on only `1 tile × split_k`
CTAs (e.g. 16 CTAs at 128²×65536 sk=32) and under-fills the 148-SM GPU, while cuBLAS
spreads the same work across ~64 CTAs. Raising `split_k` further regrows the reduce
rather than helping. Closing this needs **Stream-K** (partition total MAC work evenly
across a fixed ~all-SMs grid with partial-tile fixup), not a larger `split_k`.
