"""Inspect how cuBLAS implements the GEMMs in our split-K benchmark vs quack.

For each shape this script reports, side by side:
- cuBLAS (torch.bmm): exact kernel names (nvjet names encode tile/stage/cluster and
  split-K variant), grid/block dims, per-kernel and total kernel time, plus the
  cublasLt heuristic's chosen algorithm config (tile, stages, splitK factor,
  reduction scheme, cluster) captured via CUBLASLT_LOG_LEVEL=5 in a subprocess
  (the env var must be set before libcublasLt loads, hence the re-exec).
- quack: kernel names (GEMM + SplitKReduce in parallel mode), grid/block dims,
  per-kernel and total kernel time.
- For both: wall time per call vs summed kernel time -- the difference is the
  host/launch overhead floor, which is what split-K amortization runs into.

Usage:
    python benchmarks/inspect_cublas.py
    python benchmarks/inspect_cublas.py --shapes "128,128,65536,1" --split_k 8
    python benchmarks/inspect_cublas.py --tile_shape_mn 128,256 --split_k_mode parallel

For a deeper dive into the nvjet mainloop (SASS, tensor-pipe utilization, stall
reasons), run the printed ncu command for a shape of interest.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import torch

_LOG_SENTINEL = "--_cublaslt-log-run"

_TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}

_DEFAULT_SHAPES = [
    (128, 128, 16384, 1),
    (128, 128, 65536, 1),
    (256, 256, 32768, 1),
    (512, 512, 16384, 1),
    (4096, 4096, 4096, 1),
]

# Interesting fields in cublasLt API logs (format varies across CUDA versions, so we
# filter rather than parse).
_LOG_KEYWORDS = re.compile(r"algo|tile|stages|splitk|reduction|cluster|cga|matmul\[", re.IGNORECASE)


def _parse_shapes(s: str):
    shapes = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = tuple(int(x.strip()) for x in chunk.split(","))
        if len(vals) != 4:
            raise argparse.ArgumentTypeError(f"shape '{chunk}' must have 4 ints (m,n,k,l)")
        shapes.append(vals)
    return shapes


def _parse_ints(s: str):
    return tuple(int(x.strip()) for x in s.split(","))


def _make_inputs(m, n, k, l, dtype):
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=dtype, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=dtype, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=dtype, device="cuda")
    return A, B, D


def _annotate_kernel(name: str) -> str:
    notes = []
    lname = name.lower()
    if "splitk" in lname:
        notes.append("split-K variant")
    if "reduce" in lname:
        notes.append("reduction kernel")
    if "nvjet" in lname:
        notes.append("cuBLAS nvjet (name tokens encode tile/stages/cluster)")
    toks = re.findall(r"\d+x\d+(?:x\d+)?", name)
    if toks:
        notes.append("tokens: " + " ".join(toks[:4]))
    return "; ".join(notes)


def profile_kernels(fn, steps: int):
    """Run fn under torch.profiler; return (kernels, wall_us_per_call).

    kernels: list of dicts with name/grid/block/us_per_call/calls_per_step,
    aggregated from the chrome trace (kineto records grid/block per kernel).
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):  # warmup (includes JIT compile for quack)
        fn()
    torch.cuda.synchronize()

    # Wall time per call: includes host launch path; the gap vs summed kernel
    # time below is the per-call overhead floor.
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t0) / steps * 1e6

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    finally:
        os.unlink(path)

    kernels = {}
    for ev in events:
        if "ernel" not in str(ev.get("cat", "")):
            continue
        name = ev.get("name", "?")
        args = ev.get("args", {})
        rec = kernels.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "total_us": 0.0,
                "grid": args.get("grid"),
                "block": args.get("block"),
            },
        )
        rec["calls"] += 1
        rec["total_us"] += float(ev.get("dur", 0.0))
    out = sorted(kernels.values(), key=lambda r: -r["total_us"])
    for rec in out:
        rec["us_per_call"] = rec["total_us"] / steps
        rec["calls_per_step"] = rec["calls"] / steps
    return out, wall_us


def _print_kernel_table(title, kernels, wall_us):
    kernel_us = sum(k["us_per_call"] for k in kernels)
    print(
        f"  {title}: wall {wall_us:8.1f} us/call | kernels {kernel_us:8.1f} us/call "
        f"| host/launch gap {wall_us - kernel_us:8.1f} us"
    )
    for k in kernels:
        grid = f"grid={k['grid']}" if k["grid"] is not None else ""
        block = f"block={k['block']}" if k["block"] is not None else ""
        note = _annotate_kernel(k["name"])
        print(f"    {k['us_per_call']:9.1f} us x{k['calls_per_step']:.0f}  {grid} {block}")
        print(f"      {k['name'][:140]}")
        if note:
            print(f"      [{note}]")


def capture_cublaslt_log(m, n, k, l, dtype_name: str):
    """Re-exec this script with cublasLt logging enabled (the env var is read at
    library load, so it cannot be flipped inside an already-initialized process)."""
    fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    env = os.environ.copy()
    env["CUBLASLT_LOG_LEVEL"] = "5"
    env["CUBLASLT_LOG_FILE"] = log_path
    cmd = [sys.executable, os.path.abspath(__file__), _LOG_SENTINEL, f"{m},{n},{k},{l}", dtype_name]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        with open(log_path) as f:
            log = f.read()
    finally:
        os.unlink(log_path)
    if res.returncode != 0:
        return None, f"log subprocess failed: {res.stderr.strip()[-500:]}"
    if not log.strip():
        return None, (
            "empty cublasLt log (older CUDA?); try running manually with "
            "CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=..."
        )
    lines = [ln for ln in log.splitlines() if _LOG_KEYWORDS.search(ln)]
    seen, deduped = set(), []
    for ln in lines:
        key = ln.strip()
        if key not in seen:
            seen.add(key)
            deduped.append(ln.strip())
    return deduped[:30], None


def _log_run(shape_str: str, dtype_name: str):
    """Subprocess body: one bmm so cublasLt logs its heuristic choice."""
    m, n, k, l = (int(x) for x in shape_str.split(","))
    dtype = _TORCH_DTYPE_MAP[dtype_name]
    A = torch.randn(l, m, k, dtype=dtype, device="cuda")
    B = torch.randn(l, n, k, dtype=dtype, device="cuda")
    torch.bmm(A, B.mT)
    torch.cuda.synchronize()


def _auto_split_k(m, n, k, l, tile_m, tile_n):
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n) * l
    k_tiles = max(1, k // 64)  # bf16/fp16 k-tile is 64 deep
    return max(1, min(math.ceil(num_sms / tiles), k_tiles // 8, 32))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapes", type=_parse_shapes, default=_DEFAULT_SHAPES)
    p.add_argument(
        "--split_k",
        type=int,
        default=0,
        help="quack split factor; 0 = auto (ceil(SMs/tiles), capped)",
    )
    p.add_argument("--split_k_mode", type=str, default="parallel", choices=["parallel", "serial"])
    p.add_argument("--tile_shape_mn", type=_parse_ints, default=(128, 128))
    p.add_argument("--cluster_shape_mn", type=_parse_ints, default=(1, 1))
    p.add_argument("--dtype", type=str, default="bf16", choices=sorted(_TORCH_DTYPE_MAP))
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--skip_cublaslt_log", action="store_true")
    args = p.parse_args()

    from quack.gemm import gemm as quack_gemm

    dtype = _TORCH_DTYPE_MAP[args.dtype]
    tile_m, tile_n = args.tile_shape_mn
    cluster_m, cluster_n = args.cluster_shape_mn

    for m, n, k, l in args.shapes:
        sk = args.split_k or _auto_split_k(m, n, k, l, tile_m, tile_n)
        print(
            f"\n=== m={m} n={n} k={k} l={l} ({args.dtype}) | "
            f"quack: tile={tile_m}x{tile_n} cluster={cluster_m}x{cluster_n} "
            f"split_k={sk} ({args.split_k_mode}) ==="
        )
        A, B, D = _make_inputs(m, n, k, l, dtype)

        kernels, wall_us = profile_kernels(lambda: torch.bmm(A, B.mT), args.steps)
        _print_kernel_table("cuBLAS  ", kernels, wall_us)

        if not args.skip_cublaslt_log:
            log_lines, err = capture_cublaslt_log(m, n, k, l, args.dtype)
            if err:
                print(f"  cublasLt log: {err}")
            else:
                print("  cublasLt heuristic choice (filtered log):")
                for ln in log_lines:
                    print(f"    {ln[:160]}")

        def run_quack():
            quack_gemm(
                A,
                B,
                D,
                C=None,
                tile_count_semaphore=None,
                tile_M=tile_m,
                tile_N=tile_n,
                cluster_M=cluster_m,
                cluster_N=cluster_n,
                persistent=True,
                split_k=sk,
                split_k_mode=args.split_k_mode,
            )

        try:
            kernels_q, wall_us_q = profile_kernels(run_quack, args.steps)
            _print_kernel_table("quack   ", kernels_q, wall_us_q)
            kernel_us = sum(kk["us_per_call"] for kk in kernels)
            kernel_us_q = sum(kk["us_per_call"] for kk in kernels_q)
            if kernel_us_q > 0 and wall_us_q > 0:
                print(
                    f"  -> quack/cuBLAS: kernel-only {kernel_us / kernel_us_q:5.2f}x, "
                    f"wall {wall_us / wall_us_q:5.2f}x"
                )
        except Exception as e:
            print(f"  quack run failed: {type(e).__name__}: {e}")

        print("  deeper mainloop dive (SASS, tensor-pipe util, stalls):")
        print(
            f"    ncu --set full -k 'regex:nvjet|kernel' --launch-count 4 "
            f"{sys.executable} benchmarks/inspect_cublas.py "
            f"--shapes '{m},{n},{k},{l}' --steps 1 --skip_cublaslt_log"
        )

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _LOG_SENTINEL:
        _log_run(sys.argv[2], sys.argv[3])
        sys.exit(0)
    main()
