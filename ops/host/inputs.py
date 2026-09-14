"""Input contracts shared by reference, planning, and device execution."""
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class Problem:
    """功能：保存FFN的逻辑形状及类型，不持有Tensor或执行计算

    输入：input_shape为2～8维正整数形状，m为前导维乘积，k为最后一维
    h/n分别为隐藏/输出宽度，dtype为float16或bfloat16，layout为linear
    应由problem_from_shapes创建以校验维度关系和范围，直接构造不做校验
    输出：只读问题描述；output_shape返回(*input_shape[:-1], n)
    """

    input_shape: tuple[int, ...]
    m: int
    k: int
    h: int
    n: int
    dtype: str
    layout: str

    @property
    def output_shape(self):
        return (*self.input_shape[:-1], self.n)


def _shape(value, name, ranks):
    result = tuple(value)
    if len(result) not in ranks:
        raise ValueError(f"{name}: unsupported rank {len(result)}")
    if any(type(d) is not int or d <= 0 for d in result):
        raise ValueError(f"{name}: dimensions must be positive integers")
    return result


def problem_from_shapes(x_shape, w1_shape, w2_shape, dtype="float16", layout="linear"):
    x = _shape(x_shape, "x", range(2, 9))
    w1 = _shape(w1_shape, "weight1", (2,))
    w2 = _shape(w2_shape, "weight2", (2,))
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dtype must be float16 or bfloat16")
    # V1 did not confirm Canonical support: reject rather than infer from shape
    if layout != "linear":
        raise ValueError("Only explicit layout='linear' is enabled in this implementation")
    h, k = w1
    n, h2 = w2
    if x[-1] != k or h != h2:
        raise ValueError("Expected x[..., K], weight1[H,K], weight2[N,H]")
    if max(k, h, n) > 65535 or prod(x[:-1]) > 2**31 - 16:
        raise ValueError("Shape exceeds the V1 baseline-compatible bounds")
    return Problem(x, prod(x[:-1]), k, h, n, dtype, layout)


def validate_tensors(x, weight1, weight2, layout="linear", require_npu=False):
    import torch

    tensors = (x, weight1, weight2)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("x, weight1 and weight2 must be torch.Tensor")
    if any(t.dtype != x.dtype or t.device != x.device for t in tensors):
        raise ValueError("All tensors must have the same dtype and device")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("Only dense strided tensors are supported")
    if any(t.requires_grad for t in tensors):
        raise ValueError("V1 is inference-only; detach inputs explicitly")
    if require_npu and x.device.type != "npu":
        raise ValueError("FFN execution requires NPU tensors; no torch fallback is used")
    return problem_from_shapes(x.shape, weight1.shape, weight2.shape,
                               str(x.dtype).removeprefix("torch."), layout)


def prepare_storage(x, weight1, weight2, plan):
    """Contiguous zero-padding is an auxiliary operation, outside timed FFN."""
    import torch

    p = plan.problem
    xc = x.contiguous().view(p.m, p.k)
    w1c, w2c = weight1.contiguous(), weight2.contiguous()
    xp = torch.zeros((plan.mp, plan.kp), device=x.device, dtype=x.dtype)
    w1p = torch.zeros((plan.hp, plan.kp), device=x.device, dtype=x.dtype)
    w2p = torch.zeros((plan.np, plan.hp), device=x.device, dtype=x.dtype)
    xp[:p.m, :p.k].copy_(xc)
    w1p[:p.h, :p.k].copy_(w1c)
    w2p[:p.n, :p.h].copy_(w2c)
    hidden = torch.empty((plan.mp, plan.hp), device=x.device, dtype=x.dtype)
    out = torch.empty((plan.mp, plan.np), device=x.device, dtype=x.dtype)
    ws = torch.empty(plan.workspace_shape, device=x.device, dtype=torch.float32)
    return xp, w1p, w2p, hidden, out, ws
