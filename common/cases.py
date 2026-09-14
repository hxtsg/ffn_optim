"""Case schema and deterministic inputs shared by validation and tuning."""
import json
from pathlib import Path
from ops.host.inputs import problem_from_shapes


def load_cases(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Expected case schema_version=1')
    cases = data.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a non-empty list')
    seen = set()
    for case in cases:
        name = case.get('case_id')
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError('case_id must be a unique non-empty string')
        seen.add(name)
        case_problem(case)
        if type(case.get('seed', 0)) is not int:
            raise ValueError('seed must be an integer')
        if type(case.get('noncontiguous', False)) is not bool:
            raise ValueError('noncontiguous must be boolean')
    return cases


def case_problem(case):
    return problem_from_shapes(case['shape'], case['weight1_shape'],
                               case['weight2_shape'], case['dtype'],
                               case.get('layout', 'linear'))


def make_inputs(case, device='cpu'):
    import torch
    p = case_problem(case)
    generator = torch.Generator(device='cpu').manual_seed(case.get('seed', 0))
    dtype = getattr(torch, p.dtype)
    def sample(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(dtype).to(device)
    x = sample(p.input_shape)
    w1 = sample((p.h, p.k), p.k ** -0.5)
    w2 = sample((p.n, p.h), p.h ** -0.5)
    if case.get('noncontiguous', False):
        def strided(t):
            buf = torch.empty((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype, device=t.device)
            buf[..., ::2].copy_(t)
            return buf[..., ::2]
        x, w1, w2 = map(strided, (x, w1, w2))
    return x, w1, w2
