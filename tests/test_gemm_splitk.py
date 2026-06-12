"""Split-K GEMM with turnstile reduction (SM100 only).

Each output tile is computed by `split_k` work units covering disjoint K ranges.
Non-final splits serialize their fp32 partial accumulators into a gmem workspace
through a per-tile turnstile counter (deterministic, k-ascending order); the final
split adds the accumulated partials and runs the regular epilogue.
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


def _run_gemm(A, B, D, split_k, tile_mn=(128, 128), cluster_mn=(1, 1), **kwargs):
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
        **kwargs,
    )


def _make_inputs(l, m, n, k, dtype):
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=dtype, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=dtype, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=dtype, device="cuda")
    return A, B, D


# ── Correctness vs fp32 reference ────────────────────────────────────────────


@pytest.mark.parametrize("split_k", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# K values exercise both divisible and ragged k-tile counts per split
@pytest.mark.parametrize(
    "m,n,k,l", [(128, 128, 16384, 1), (256, 384, 8192, 1), (128, 256, 4160, 3)]
)
def test_gemm_splitk(m, n, k, l, dtype, split_k):
    A, B, D = _make_inputs(l, m, n, k, dtype)
    _run_gemm(A, B, D, split_k)
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# Edge output tiles (M/N not divisible by the tile shape): OOB lanes round-trip
# through the workspace and must be predicated away only by the final epilogue.
@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_edge_tiles(split_k):
    dtype = torch.bfloat16
    A, B, D = _make_inputs(2, 192, 320, 8192, dtype)
    _run_gemm(A, B, D, split_k)
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# split_k larger than the number of k tiles: some splits own zero k tiles and must
# contribute zeros through the turnstile.
def test_gemm_splitk_more_splits_than_k_tiles():
    dtype = torch.bfloat16
    A, B, D = _make_inputs(1, 128, 128, 128, dtype)
    _run_gemm(A, B, D, split_k=8)
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_cluster(split_k):
    dtype = torch.bfloat16
    A, B, D = _make_inputs(1, 256, 256, 16384, dtype)
    _run_gemm(A, B, D, split_k, cluster_mn=(2, 1))
    ref = torch.bmm(A.float(), B.float().mT).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# ── Epilogue ops must apply to the fully reduced accumulator ─────────────────


@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_alpha_beta_C(split_k):
    dtype = torch.bfloat16
    l, m, n, k = 2, 128, 256, 8192
    A, B, D = _make_inputs(l, m, n, k, dtype)
    C = torch.randn(l, m, n, dtype=dtype, device="cuda")
    alpha, beta = 0.5, 0.7
    _run_gemm(A, B, D, split_k, C=C, alpha=alpha, beta=beta)
    ref = (alpha * torch.bmm(A.float(), B.float().mT) + beta * C.float()).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_splitk_bias(split_k):
    dtype = torch.bfloat16
    l, m, n, k = 2, 128, 256, 8192
    A, B, D = _make_inputs(l, m, n, k, dtype)
    bias = torch.randn(l, n, dtype=dtype, device="cuda")
    _run_gemm(A, B, D, split_k, rowvec_bias=bias)
    ref = (torch.bmm(A.float(), B.float().mT) + bias.float().unsqueeze(1)).to(dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[dtype], rtol=RTOL)


# ── Determinism: the point of the turnstile vs atomic reduction ──────────────


@pytest.mark.parametrize("split_k", [4, 8])
def test_gemm_splitk_deterministic(split_k):
    dtype = torch.bfloat16
    A, B, D1 = _make_inputs(1, 128, 256, 16384, dtype)
    D2 = torch.empty_like(D1)
    _run_gemm(A, B, D1, split_k)
    for _ in range(5):
        _run_gemm(A, B, D2, split_k)
        assert torch.equal(D1, D2), "split-K turnstile reduction must be run-to-run deterministic"
