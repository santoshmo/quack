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
| 128×128×16384 | 1  | 20.1 TF / 26.7 µs | 17.5 TF / 30.7 µs | 32 | **0.87×** | 2.91× |
| 128×128×65536 | 1  | 58.2 TF / 36.9 µs | 55.3 TF / 38.8 µs | 32 | **0.95×** | 8.20× |
| 256×256×32768 | 4  | 110.6 TF / 38.8 µs | 110.6 TF / 38.8 µs | 32 | **1.00×** | 4.33× |
| 512×512×16384 | 16 | 233.2 TF / 36.8 µs | 209.9 TF / 40.9 µs | 8 | **0.90×** | 2.25× |
| 4096×4096×4096 | 1024 | 825.7 TF / 166 µs | 710 TF / 194 µs | 1 | 0.86× | 1.00× |

The last row uses `split_k=1` (split-K is counter-productive once the GPU is already
full); its 0.86× reflects the dense mainloop, not split-K.

## Parallel vs serial split-K (best vs cuBLAS)

The earlier serial path (in-kernel turnstile reduction) is kept as
`split_k_mode="serial"` for reference. Parallel is uniformly better and scales with
`split_k` instead of plateauing:

| shape | serial best | parallel best |
|---|---|---|
| 128×128×16384 | 0.30× | **0.87×** |
| 128×128×65536 | 0.18× | **0.95×** |
| 256×256×32768 | 0.29× | **1.00×** |
| 512×512×16384 | 0.40× | **0.90×** |

## How we got here

1. **Parallel split-K mode** (per-split workspace + separate reduce) replacing the
   serial turnstile — scales with `split_k` and removes the high-`split_k` blow-up.
2. **Parallelized the reduce** over `(output tile, chunk)` — it had been one CTA per
   tile (266 µs at 128²×16384 sk=32); now ~µs-scale.
3. **Unrolled the reduce accumulation** — the reduce is latency-bound; unrolling keeps
   several of each thread's slot loads in flight.
4. **One element per reduce thread (`vec_width` 4→1)** — the decisive reduce win. The
   reduce throughput scales with outstanding memory transactions = thread count, and
   the thread count is `tile_mn / vec_width` per tile. Dropping the vector width to 1
   quadruples it to `tile_mn` threads (matching cuBLAS's reduce thread count): the
   reduce at 128²×65536 sk=32 went 13.1→8.6 µs (V=4/2/1 = 13.1/9.8/8.6 µs). A cheaper
   reduce both stops it dominating *and* makes a higher `split_k` profitable (it's no
   longer eaten by reduce growth), which fills the GEMM more — so the gain compounds.
5. **Adaptive `vec_width`** — V=1 over-decomposes into many tiny reduce CTAs for shapes
   with many output tiles (where the reduce isn't the bottleneck). `choose_reduce_vec_width`
   picks the smallest V whose total reduce-CTA count stays within ~8 GPU waves: V=1 for
   few tiles (max threads), larger V for many tiles. Recovers 512²×16384 (16 tiles)
   from 0.86× back to 0.90× while keeping every few-tile win.

## Why not Stream-K

An earlier hypothesis was that the single-tile shapes were GEMM-under-fill-limited and
needed Stream-K. A sweep disproved it: higher `split_k` (more GEMM fill) **monotonically
hurts** past the optimum (128²×65536: sk 16/32/64/128 = 6.70/7.08/6.23/4.73× vs sk=1),
because the reduce grows faster than the fill helps (sk32→sk64: GEMM 20.0→15.6 µs but
reduce 13.1→21.5 µs). The optimum sits at only 16–32 CTAs (11–22% of the 148 SMs), so
the GPU is *not* full there — the binding constraint was the **reduce cost**, not fill.
Stream-K on a single tile produces ~`G` contributors (≡ a huge `split_k`), landing deep
in the *worse* region, so it cannot beat the current path on these shapes. The fix was a
cheaper reduce (step 4 above), not Stream-K.

## Remaining gap

The single-tile reduce is now 8.6 µs vs cuBLAS's 3.8 µs; a split-axis (tree) reduction
that adds threads beyond `tile_mn` could close more, at the cost of a two-pass combine —
high effort for the last few percent, given the few-tile shapes are already at 0.87–1.00×.
