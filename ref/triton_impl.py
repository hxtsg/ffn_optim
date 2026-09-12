"""Ascend950-targeted Triton-Ascend inference reference

Exactly two compute launches for nonempty inputs after contiguous normalization
No Stream-K, atomics, full-load policy, original EagleFFN, or torch matmul calls
Device compilation/correctness must be validated on the target software stack
"""

import torch
import triton
import triton.language as tl
from .common import prepare


@triton.jit
def _up(X, W, B, A, M: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tiles_n = tl.cdiv(H, BN)
    tiles = tl.cdiv(M, BM) * tiles_n
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        rows = (tile // tiles_n) * BM + tl.arange(0, BM)
        cols = (tile % tiles_n) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        gate = tl.zeros((BM, BN), tl.float32)
        if ACT == 2:
            up = tl.zeros((BM, BN), tl.float32)
        for block_k in range(tl.cdiv(K, BK)):
            ks = block_k * BK + kk
            x = tl.load(X + rows[:, None] * K + ks[None, :],
                        (rows[:, None] < M) & (ks[None, :] < K), other=0)
            w = tl.load(W + cols[None, :] * K + ks[:, None],
                        (cols[None, :] < H) & (ks[:, None] < K), other=0)
            gate += tl.dot(x, w)
            if ACT == 2:
                wu = tl.load(W + (cols[None, :] + H) * K + ks[:, None],
                             (cols[None, :] < H) & (ks[:, None] < K), other=0)
                up += tl.dot(x, wu)
        if HAS_BIAS:
            gate += tl.load(B + cols, cols < H, other=0).to(tl.float32)[None, :]
            if ACT == 2:
                up += tl.load(B + H + cols, cols < H, other=0).to(tl.float32)[None, :]
        if ACT == 0:
            a = 0.5 * gate * (1.0 + tl.erf(gate * 0.7071067811865476))
        elif ACT == 1:
            a = gate * tl.sigmoid(gate)
        else:
            a = gate * tl.sigmoid(gate) * up
        a = a.to(A.dtype.element_ty)
        tl.store(A + rows[:, None] * H + cols[None, :], a,
                 (rows[:, None] < M) & (cols[None, :] < H))


@triton.jit
def _down(A, W, B, Y, M: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
          HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tiles_n = tl.cdiv(N, BN)
    tiles = tl.cdiv(M, BM) * tiles_n
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        rows = (tile // tiles_n) * BM + tl.arange(0, BM)
        cols = (tile % tiles_n) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block_k in range(tl.cdiv(H, BK)):
            ks = block_k * BK + kk
            a = tl.load(A + rows[:, None] * H + ks[None, :],
                        (rows[:, None] < M) & (ks[None, :] < H), other=0)
            w = tl.load(W + cols[None, :] * H + ks[:, None],
                        (cols[None, :] < N) & (ks[:, None] < H), other=0)
            acc += tl.dot(a, w)
        if HAS_BIAS:
            acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
        tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(Y.dtype.element_ty),
                 (rows[:, None] < M) & (cols[None, :] < N))


def ffn_triton(x, weight1, weight2, bias1=None, bias2=None,
               activation="gelu", layout="linear"):
    p = prepare(x, weight1, weight2, bias1, bias2, activation, layout)
    if p.x.device.type != "npu":
        raise ValueError("ffn_triton requires an Ascend NPU and Triton-Ascend")
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad
                                      for t in (x, weight1, weight2, bias1, bias2)):
        raise ValueError("ffn_triton is inference-only; call it under torch.no_grad()")
    output = torch.empty((p.m, p.n), device=p.x.device, dtype=p.x.dtype)
    if p.m == 0:
        return output.reshape(p.output_shape)
    hidden = torch.empty((p.m, p.h), device=p.x.device, dtype=p.x.dtype)
    # Small fixed tiles favor readability, not claimed optimal for Ascend950
    bm, bn, bk = 32, 64, 32
    with torch.npu.device(p.x.device):
        props = triton.runtime.driver.active.utils.get_device_properties(p.x.device.index)
        cores = int(props["num_aicore"])
        if cores <= 0:
            raise RuntimeError("Triton-Ascend reported an invalid num_aicore")
        up_grid = (min(cores, triton.cdiv(p.m, bm) * triton.cdiv(p.h, bn)),)
        down_grid = (min(cores, triton.cdiv(p.m, bm) * triton.cdiv(p.n, bn)),)
        # Both launches use the current NPU stream; no CPU synchronization here
        _up[up_grid](p.x, p.w1, p.b1 if p.b1 is not None else p.x, hidden,
                     p.m, p.k, p.h, p.b1 is not None,
                     {"gelu": 0, "silu": 1, "swiglu": 2}[p.activation], bm, bn, bk)
        _down[down_grid](hidden, p.w2, p.b2 if p.b2 is not None else p.x, output,
                         p.m, p.h, p.n, p.b2 is not None, bm, bn, bk)
    return output.reshape(p.output_shape)
