"""Torch small-op reference; no fused EagleFFN calls."""
from ops.host.inputs import validate_tensors


def gelu_rational(z):
    """Baseline fusion_regbase_act.h rational erf approximation, FP32."""
    t = (z.float() * 0.70710678118654752).clamp(-3.92, 3.92)
    t2 = t * t
    p = (((((t2 * 0.053443748819 + 7.5517016694) * t2 + 101.62808918)
            * t2 + 1393.8061484) * t2 + 5063.7915060) * t2 + 29639.384698) * t
    q = ((((t2 + 31.212858877) * t2 + 398.56963806) * t2 + 3023.1248150)
         * t2 + 13243.365831) * t2 + 26267.224157
    return 0.5 * z.float() * (1 + p / q)


def ffn_torch(x, weight1, weight2, *, staged=False, rational=False):
    import torch.nn.functional as F
    p = validate_tensors(x, weight1, weight2)
    z = x.reshape(p.m, p.k).float() @ weight1.float().T
    hidden = gelu_rational(z) if rational else F.gelu(z, approximate='none')
    if staged:
        hidden = hidden.to(x.dtype).float()
    return (hidden @ weight2.float().T).to(x.dtype).reshape(p.output_shape)
