# Copyright (c) 2026, QuACK contributors.
# Second kernel of parallel split-K GEMM: sums the per-split fp32 partial tiles written
# by the GEMM kernel (fixed ascending split order -> run-to-run deterministic) and
# applies the default epilogue (alpha, beta*C, rowvec/colvec bias) while converting to
# the output dtype. Mirrors cuBLAS's splitKreduce_kernel.

from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import make_ptr

import torch
from torch import Tensor

import quack.utils as utils
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.gemm_tvm_ffi_utils import div_for_dtype


class SplitKReduce:
    """One CTA per (output tile, chunk); each CTA sums its chunk across the tile's slots.

    Workspace layout: tile `tile_idx` owns `tile_count[tile_idx]` consecutive slots
    starting at `tile_first_slot[tile_idx]` (a host-built prefix sum); each slot holds a
    row-major (tile_m, tile_n) fp32 partial, where tile_idx =
    (tile_m_idx * ntile_n + tile_n_idx) * L + l. The fixed split-K path passes uniform
    tables (count = split_k, first_slot = tile_idx * split_k); Stream-K will pass a
    variable, data-dependent count per tile into the same kernel. Edge tiles are padded
    inside the slot; this kernel predicates the final D store by (M, N).
    """

    # Small CTAs so a tile splits into many reduce CTAs (num_chunks = tile_mn/(64*4),
    # e.g. 64 for a 128x128 tile vs 16 at 256 threads). Once the unrolled accumulation
    # below hides per-thread load latency, spreading the reduce across more SMs pulls
    # more HBM bandwidth -- at high split_k this is measurably faster than 256 threads.
    num_threads = 64
    vec_width = 4  # fp32 elements per vectorized workspace load (16B)

    def __init__(self, tile_m: int, tile_n: int):
        self.tile_m, self.tile_n = tile_m, tile_n
        # A vector never straddles a row of the slot, and chunks tile the slot exactly
        assert tile_n % self.vec_width == 0
        assert (tile_m * tile_n) % (self.num_threads * self.vec_width) == 0

    @cute.jit
    def __call__(
        self,
        mWS: cute.Tensor,  # (total_contributors * tile_m * tile_n,) f32
        mD: cute.Tensor,  # (M, N, L)
        mC: Optional[cute.Tensor],  # (M, N, L)
        alpha: Optional[Float32 | cute.Pointer],
        beta: Optional[Float32 | cute.Pointer],
        mRowVec: Optional[cute.Tensor],  # (L, N)
        mColVec: Optional[cute.Tensor],  # (L, M)
        mTileFirstSlot: cute.Tensor,  # (num_tiles,) i32: first slot owned by each tile
        mTileCount: cute.Tensor,  # (num_tiles,) i32: contributors per tile
        stream: cuda.CUstream,
    ):
        ntile_m = cute.ceil_div(cute.size(mD, mode=[0]), self.tile_m)
        ntile_n = cute.ceil_div(cute.size(mD, mode=[1]), self.tile_n)
        # One CTA per (output tile, chunk): a tile's chunks reduce on separate CTAs
        # so the kernel fills the GPU instead of serializing on one CTA per tile.
        # block_idx.z packs (l, chunk) -> grid.z = L * num_chunks.
        num_chunks = (self.tile_m * self.tile_n) // (self.num_threads * self.vec_width)
        self.kernel(
            mWS, mD, mC, alpha, beta, mRowVec, mColVec, mTileFirstSlot, mTileCount, ntile_n
        ).launch(
            grid=[ntile_m, ntile_n, cute.size(mD, mode=[2]) * num_chunks],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mWS: cute.Tensor,
        mD: cute.Tensor,
        mC: Optional[cute.Tensor],
        alpha: Optional[Float32 | cute.Pointer],
        beta: Optional[Float32 | cute.Pointer],
        mRowVec: Optional[cute.Tensor],
        mColVec: Optional[cute.Tensor],
        mTileFirstSlot: cute.Tensor,
        mTileCount: cute.Tensor,
        ntile_n: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_lc = cute.arch.block_idx()
        tile_m = const_expr(self.tile_m)
        tile_n = const_expr(self.tile_n)
        tile_mn = const_expr(tile_m * tile_n)
        V = const_expr(self.vec_width)
        num_chunks = const_expr(tile_mn // (self.num_threads * V))
        # block_idx.z packs (l, chunk): this CTA reduces one chunk of one output tile.
        bid_l = bid_lc // num_chunks
        chunk = bid_lc % num_chunks

        len_m = cute.size(mD, mode=[0])
        len_n = cute.size(mD, mode=[1])
        num_l = cute.size(mD, mode=[2])
        tile_idx = (bid_m * ntile_n + bid_n) * num_l + bid_l
        # Variable per-tile contributor count + first-slot offset (a host-built prefix
        # sum) generalize the fixed `tile_idx * split_k` of the uniform split-K path, so
        # Stream-K (data-dependent contributors per tile) can reuse this reduce kernel.
        first_slot = mTileFirstSlot[tile_idx]
        count = mTileCount[tile_idx]
        slot_base = mWS.iterator + first_slot * tile_mn
        m0, n0 = bid_m * tile_m, bid_n * tile_n

        alpha_v, beta_v = Float32(1.0), Float32(1.0)
        if const_expr(alpha is not None):
            alpha_v = utils.load_scalar_or_pointer(alpha)
        if const_expr(beta is not None):
            beta_v = utils.load_scalar_or_pointer(beta)

        rAcc = cute.make_rmem_tensor(V, Float32)
        flat = (chunk * self.num_threads + tidx) * V
        # Sum the split slices in fixed ascending order (deterministic). The reduce is
        # latency- not bandwidth-bound (few threads per output element), so unroll the
        # accumulation: the slice loads are independent and issue ahead, keeping many
        # loads in flight per thread. Unrolling preserves the add order (numerics).
        tWS = cute.make_tensor(slot_base + flat, cute.make_layout(V))
        cute.autovec_copy(tWS, rAcc)
        for s in cutlass.range(1, count, unroll=8):
            tWS_s = cute.make_tensor(slot_base + s * tile_mn + flat, cute.make_layout(V))
            rAcc.store(rAcc.load() + tWS_s.load())
        # The vector spans one row of the slot (tile_n % V == 0)
        m = m0 + flat // tile_n
        n_base = n0 + flat % tile_n
        if m < len_m:
            colvec_val = Float32(0.0)
            if const_expr(mColVec is not None):
                colvec_val = Float32(mColVec[bid_l, m])
            # Epilogue + predicated store, elementwise (D/C layout-agnostic).
            # Same op order as GemmDefaultEpiMixin.epi_visit_subtile:
            # alpha * acc, then (+ beta * C | + C), then rowvec/colvec bias.
            for i in cutlass.range_constexpr(V):
                n = n_base + i
                if n < len_n:
                    val = Float32(rAcc[i])
                    if const_expr(alpha is not None):
                        val = val * alpha_v
                    if const_expr(mC is not None):
                        c_val = Float32(mC[m, n, bid_l])
                        if const_expr(beta is not None):
                            val += beta_v * c_val
                        else:
                            val += c_val
                    if const_expr(mRowVec is not None):
                        val += Float32(mRowVec[bid_l, n])
                    if const_expr(mColVec is not None):
                        val += colvec_val
                    mD[m, n, bid_l] = mD.element_type(val)


@jit_cache
def _compile_splitk_reduce(
    d_dtype,
    c_dtype,
    d_major,
    c_major,
    tile_m,
    tile_n,
    alpha_mode,
    beta_mode,
    rowvec_dtype,
    colvec_dtype,
):
    m, n, l = cute.sym_int(), cute.sym_int(), cute.sym_int()

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(1.0)
        else:
            return make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    mWS = fake_tensor(Float32, (cute.sym_int(),), leading_dim=0, divisibility=4)
    mD = fake_tensor(
        d_dtype,
        (m, n, l),
        leading_dim=1 if d_major == "n" else 0,
        divisibility=div_for_dtype(d_dtype),
    )
    mC = (
        fake_tensor(
            c_dtype,
            (m, n, l),
            leading_dim=1 if c_major == "n" else 0,
            divisibility=div_for_dtype(c_dtype),
        )
        if c_dtype is not None
        else None
    )
    mRowVec = (
        fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
        if rowvec_dtype is not None
        else None
    )
    mColVec = (
        fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
        if colvec_dtype is not None
        else None
    )
    mTileFirstSlot = fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    mTileCount = fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    return cute.compile(
        SplitKReduce(tile_m, tile_n),
        mWS,
        mD,
        mC,
        fake_scalar(alpha_mode),
        fake_scalar(beta_mode),
        mRowVec,
        mColVec,
        mTileFirstSlot,
        mTileCount,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _scalar_modes(alpha, beta):
    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)
    return alpha_mode, beta_mode


def compile_splitk_reduce(D, C, alpha, beta, rowvec, colvec, tile_m, tile_n):
    """Compile (or fetch from cache) the reduce kernel for these tensor properties."""
    alpha_mode, beta_mode = _scalar_modes(alpha, beta)
    return _compile_splitk_reduce(
        torch2cute_dtype_map[D.dtype],
        torch2cute_dtype_map[C.dtype] if C is not None else None,
        "n" if D.stride(1) == 1 else "m",
        ("n" if C.stride(1) == 1 else "m") if C is not None else None,
        tile_m,
        tile_n,
        alpha_mode,
        beta_mode,
        torch2cute_dtype_map[rowvec.dtype] if rowvec is not None else None,
        torch2cute_dtype_map[colvec.dtype] if colvec is not None else None,
    )


_uniform_tables_cache: dict = {}


def uniform_splitk_tables(num_tiles: int, split_k: int, device) -> tuple[Tensor, Tensor]:
    """(tile_first_slot, tile_count) for the fixed split-K layout: every tile has exactly
    `split_k` contributors at slots [t*split_k, (t+1)*split_k). Cached per
    (num_tiles, split_k, device) so the parallel split-K path pays no per-call build cost.
    Stream-K will instead build non-uniform tables and feed the same reduce kernel."""
    dev_key = device.index if device.type == "cuda" else -1
    key = (num_tiles, split_k, dev_key)
    cached = _uniform_tables_cache.get(key)
    if cached is None:
        first = torch.arange(0, num_tiles * split_k, split_k, dtype=torch.int32, device=device)
        count = torch.full((num_tiles,), split_k, dtype=torch.int32, device=device)
        cached = (first, count)
        _uniform_tables_cache[key] = cached
    return cached


def splitk_reduce(
    ws: Tensor,  # (total_contributors * tile_m * tile_n,) f32 partials
    D: Tensor,  # (M, N, L), post-perm3d layout
    C: Optional[Tensor],  # (M, N, L), post-perm3d layout
    alpha: float | Tensor,
    beta: float | Tensor,
    rowvec: Optional[Tensor],  # (L, N)
    colvec: Optional[Tensor],  # (L, M)
    tile_first_slot: Tensor,  # (num_tiles,) i32
    tile_count: Tensor,  # (num_tiles,) i32
    tile_m: int,
    tile_n: int,
) -> None:
    compiled_fn = compile_splitk_reduce(D, C, alpha, beta, rowvec, colvec, tile_m, tile_n)

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    alpha_mode, beta_mode = _scalar_modes(alpha, beta)

    def scalar_arg(scalar, mode):
        if mode == 0:
            return None
        elif mode == 1:
            return float(scalar)
        else:
            return scalar.data_ptr()

    compiled_fn(
        ws,
        D,
        C,
        scalar_arg(alpha, alpha_mode),
        scalar_arg(beta, beta_mode),
        rowvec,
        colvec,
        tile_first_slot,
        tile_count,
    )
