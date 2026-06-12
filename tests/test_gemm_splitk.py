"""Split-K GEMM (SM100 only).

Each output tile is computed by `split_k` work units covering disjoint K ranges.
Two reduction modes, both run-to-run deterministic:
- "parallel" (default): every split stores fp32 partials to its own workspace slice
  with no inter-CTA synchronization; a separate reduce kernel sums the slices in
  fixed ascending order and applies the epilogue (cuBLAS splitKreduce-style).
- "serial": fused in-kernel turnstile reduction; non-final splits serialize their
  partials into a shared slot (k-ascending) and the final split runs the epilogue.
"""

import math
import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="split-K GEMM is SM100 only",
)

ATOL = {torch.bfloat16: 3e-2, torch.float16: 1e-2}
RTOL = 1e-3
MODES = ["parallel", "serial"]


def _run_gemm(A, B, D, split_k, mode, tile_mn=(128, 128), cluster_mn=(1, 1), **kwargs):
    quack_gemm(
        A,
        B,
        D,
        C=kwargs.pop("C", None),
        tile_count_semaphore=None,
        tile_M=tile_mn[0],
        tile_N=tile_mn[1],
        cluster_M=cluster_mn[0],
        cluster_N=cluster_mn[1],
        persistent=True,
        split_k=split_k,
        split_k_mode=mode,
        **kwargs,
    )


def _make_inputs(l, m, n, k, dtype):
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=dtype, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=dtype, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=dtype, device="cuda")
    return A, B, D


# ── Correctness vs fp32 reference ────────────────────────────────────────────


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# K values exercise both divisible and ragged k-tile counts per split
@pytest.mark.parametrize(
    "m,n,k,l", [(128, 128, 16384, 1), (256, 384, 8192, 1), (128, 256, 4160, 3)]
)
def test_gemm_splitk(m, n, k, l, dtype, split_k, mode):
    A, B, D = _make_inputs(l, m, n, k, dtype)
    _run_gemm(A, B, D, split_k, mode)
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# Parallel mode is the one whose split count can scale to fill the GPU
@pytest.mark.parametrize("split_k", [16, 64])
def test_gemm_splitk_parallel_large_split(split_k):
    dtype = torch.bfloat16
    A, B, D = _make_inputs(1, 128, 128, 16384, dtype)
    _run_gemm(A, B, D, split_k, "parallel")
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# Edge output tiles (M/N not divisible by the tile shape): OOB lanes round-trip
# through the workspace and must be predicated away only by the final epilogue
# (serial) / the reduce kernel (parallel).
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_edge_tiles(split_k, mode):
    dtype = torch.bfloat16
    A, B, D = _make_inputs(2, 192, 320, 8192, dtype)
    _run_gemm(A, B, D, split_k, mode)
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# split_k larger than the number of k tiles: some splits own zero k tiles and must
# contribute zeros.
@pytest.mark.parametrize("mode", MODES)
def test_gemm_splitk_more_splits_than_k_tiles(mode):
    dtype = torch.bfloat16
    A, B, D = _make_inputs(1, 128, 128, 128, dtype)
    _run_gemm(A, B, D, 8, mode)
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_cluster(split_k, mode):
    dtype = torch.bfloat16
    A, B, D = _make_inputs(1, 256, 256, 16384, dtype)
    _run_gemm(A, B, D, split_k, mode, cluster_mn=(2, 1))
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# ── Epilogue ops must apply to the fully reduced accumulator ─────────────────


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_alpha_beta_C(split_k, mode):
    dtype = torch.bfloat16
    l, m, n, k = 2, 128, 256, 8192
    A, B, D = _make_inputs(l, m, n, k, dtype)
    C = torch.randn(l, m, n, dtype=dtype, device="cuda")
    alpha, beta = 0.5, 0.7
    _run_gemm(A, B, D, split_k, mode, C=C, alpha=alpha, beta=beta)
    ref = (alpha * torch.bmm(A.float(), B.float().mT) + beta * C.float()).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_bias(split_k, mode):
    dtype = torch.bfloat16
    l, m, n, k = 2, 128, 256, 8192
    A, B, D = _make_inputs(l, m, n, k, dtype)
    bias = torch.randn(l, n, dtype=dtype, device="cuda")
    _run_gemm(A, B, D, split_k, mode, rowvec_bias=bias)
    ref = (torch.bmm(A.float(), B.float().mT) + bias.float().unsqueeze(1)).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# ── Determinism: the point of split-K with an ordered reduction ──────────────


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [4, 8])
def test_gemm_splitk_deterministic(split_k, mode):
    dtype = torch.bfloat16
    A, B, D1 = _make_inputs(1, 128, 256, 16384, dtype)
    D2 = torch.empty_like(D1)
    _run_gemm(A, B, D1, split_k, mode)
    for _ in range(5):
        _run_gemm(A, B, D2, split_k, mode)
        assert torch.equal(D1, D2), "split-K reduction must be run-to-run deterministic"


# ── Table-driven reduce: variable contributors per tile (Stream-K precursor) ──
# Drive the reduce kernel DIRECTLY with a NON-uniform contributor layout to prove the
# per-tile (first_slot, count) prefix-sum indirection and the (m_idx, n_idx, l) tile
# enumeration are correct independently of the GEMM. The uniform split-K path uses
# first_slot = tile_idx * split_k (order-independent) and so cannot exercise this; it is
# the slot-index identity Stream-K will rely on once contributor counts vary per tile.
@pytest.mark.parametrize("d_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_C", [False, True])
@pytest.mark.parametrize("l", [1, 2])
def test_splitk_reduce_variable_contributors(l, with_C, d_dtype):
    from quack.gemm_splitk_reduce import splitk_reduce

    torch.manual_seed(0)
    tile_m = tile_n = 128
    ntile_m, ntile_n = 2, 3  # multi-tile, non-square raster (exact multiples, no edges)
    M, N = ntile_m * tile_m, ntile_n * tile_n
    num_tiles = ntile_m * ntile_n * l

    # Distinct per-tile contributor counts (>=1) so the prefix-sum offset actually matters.
    counts = torch.tensor([1, 5, 2, 4, 3, 1, 6, 2, 1, 3, 5, 2][:num_tiles], dtype=torch.int32)
    first = torch.zeros(num_tiles, dtype=torch.int32)
    if num_tiles > 1:
        first[1:] = torch.cumsum(counts.to(torch.int64), 0)[:-1].to(torch.int32)
    total = int(counts.sum().item())
    tile_first_slot, tile_count = first.cuda(), counts.cuda()

    ws = torch.randn(total, tile_m, tile_n, dtype=torch.float32, device="cuda")
    # n-major (M, N, L) layout, matching what perm3d hands the kernel in gemm().
    D = torch.empty(l, M, N, dtype=d_dtype, device="cuda").permute(1, 2, 0)
    C = torch.randn(l, M, N, dtype=d_dtype, device="cuda").permute(1, 2, 0) if with_C else None
    alpha, beta = (0.5, 0.7) if with_C else (1.0, 1.0)

    # Reference: sum each tile's own slots, then the kernel's epilogue order.
    ref = torch.zeros(M, N, l, dtype=torch.float32, device="cuda")
    for t in range(num_tiles):
        li, rem = t % l, t // l
        ni, mi = rem % ntile_n, rem // ntile_n
        f, c = int(first[t]), int(counts[t])
        part = ws[f : f + c].sum(0)
        if alpha != 1.0:
            part = part * alpha
        rs, re, cs, ce = mi * tile_m, (mi + 1) * tile_m, ni * tile_n, (ni + 1) * tile_n
        if with_C:
            part = part + beta * C[rs:re, cs:ce, li].float()
        ref[rs:re, cs:ce, li] = part

    splitk_reduce(
        ws.reshape(-1), D, C, alpha, beta, None, None,
        tile_first_slot, tile_count, tile_m, tile_n,
    )
    torch.testing.assert_close(D.float(), ref, atol=ATOL[d_dtype], rtol=RTOL)
