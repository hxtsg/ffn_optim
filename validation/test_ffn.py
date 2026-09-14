"""Shared error metrics and CPU-only regressions (no NPU at import/discovery)."""
import ast
import unittest


def compare(output, golden):
    import torch
    result = {'passed': False, 'shape_ok': tuple(output.shape) == tuple(golden.shape),
              'dtype_ok': output.dtype == golden.dtype, 'finite': False,
              'max_abs': None, 'rms': None, 'mean_rel': None}
    if not result['shape_ok'] or not result['dtype_ok']:
        return result
    a, b = output.detach().float(), golden.detach().float()
    result['finite'] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not result['finite'] or not a.numel():
        return result
    d = (a - b).abs()
    result.update(max_abs=d.max().item(), rms=d.square().mean().sqrt().item(),
                  mean_rel=(d / (b.abs() + 1e-3)).mean().item())
    result['passed'] = result['mean_rel'] < 0.05
    return result


class InputTests(unittest.TestCase):
    def test_ranks_and_padding(self):
        import torch
        from ops.host.inputs import validate_tensors, prepare_storage
        from ops.host.tiling import make_plan
        from validation.reference import ffn_torch
        for dtype in (torch.float16, torch.bfloat16):
            for rank in range(2, 9):
                shape = (1,) * (rank - 2) + (3, 17)
                x = torch.randn((*shape[:-1], 34)).to(dtype)[..., ::2]
                w1 = torch.randn(19, 34).to(dtype)[:, ::2]
                w2 = torch.randn(23, 38).to(dtype)[:, ::2]
                p = validate_tensors(x, w1, w2)
                plan = make_plan(p)
                xp, a, b, *_ = prepare_storage(x, w1, w2, plan)
                got = ffn_torch(xp, a, b, staged=True)[:p.m, :p.n].view(p.output_shape)
                ref = ffn_torch(x, w1, w2, staged=True)
                self.assertTrue(compare(got, ref)['passed'])
                self.assertEqual(got.shape, (*shape[:-1], 23))

    def test_invalid_shapes(self):
        from ops.host.inputs import problem_from_shapes as p
        for args in [([3], [4, 3], [3, 4]), ([0, 3], [4, 3], [3, 4]),
                     ([2, 3], [4, 2], [3, 4]), ([1]*9, [4, 1], [3, 4]),
                     ([True, 3], [4, 3], [3, 4]), ([2, 65536], [4, 65536], [3, 4])]:
            with self.assertRaises(ValueError):
                p(*args)
        for kwargs in ({'dtype': 'float32'}, {'layout': 'canonical'}):
            with self.assertRaises(ValueError):
                p([2, 3], [4, 3], [3, 4], **kwargs)

    def test_no_device_fallback(self):
        import torch
        from ops.api import ffn
        with self.assertRaisesRegex(ValueError, 'requires NPU'):
            ffn(torch.zeros(2, 3).half(), torch.zeros(4, 3).half(), torch.zeros(3, 4).half())

    def test_invalid_tensors(self):
        import torch
        from ops.host.inputs import validate_tensors
        x, a, b = torch.zeros(2, 3).half(), torch.zeros(4, 3).half(), torch.zeros(3, 4).half()
        with self.assertRaises(ValueError):
            validate_tensors(x, a.float(), b)
        with self.assertRaises(ValueError):
            validate_tensors(x.requires_grad_(), a, b)


class MetricTests(unittest.TestCase):
    def test_bad_results(self):
        import torch
        a = torch.ones(2, 3).half()
        self.assertTrue(compare(a, a)['passed'])
        for bad in (a.float(), a.T, a * float('nan'), a * float('inf'), a * 2):
            self.assertFalse(compare(bad, a)['passed'])

    def test_gelu_approximation(self):
        import torch
        from validation.reference import gelu_rational
        x = torch.linspace(-10, 10, 10001)
        d = (gelu_rational(x) - torch.nn.functional.gelu(x, approximate='none')).abs()
        self.assertLess(d.max().item(), 2e-5)


class CompositionTests(unittest.TestCase):
    def test_generated_gelu_expression(self):
        """Evaluate the actual generated arithmetic AST on CPU, not a device emulator."""
        import torch
        from types import SimpleNamespace
        from ops.kernel.ffn import generate_source
        from validation.reference import gelu_rational
        tree = ast.parse(generate_source()[0])
        assignments = {n.targets[0].id: n for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'z'}
        self.assertEqual(len(assignments), 1)
        chunk_loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                          and isinstance(n.target, ast.Name) and n.target.id == 'chunk')
        operations = [n for n in chunk_loop.body if isinstance(n, ast.Assign)
                      and n.targets[0].id in ('t', 't2', 'p', 'q', 'y')]
        x = torch.linspace(-10, 10, 10001)
        namespace = {'z': x, 'mask32': None, 'tla': SimpleNamespace(
            add=lambda a, b, **kw: a+b, mul=lambda a, b, **kw: a*b,
            div=lambda a, b, **kw: a/b, min=lambda a, b, **kw: a.clamp(max=b),
            max=lambda a, b, **kw: a.clamp(min=b))}
        module = ast.fix_missing_locations(ast.Module(body=operations, type_ignores=[]))
        exec(compile(module, '<gelu-cpu-arithmetic>', 'exec'), namespace)
        torch.testing.assert_close(namespace['y'], gelu_rational(x), rtol=0, atol=0)

    def test_one_kernel_and_regions(self):
        from ops.kernel.ffn import generate_source
        for kind in ('basic', 'streamk'):
            source, provenance = generate_source(kind)
            tree = ast.parse(source)
            kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            self.assertEqual(len(kernels), 1)
            regions = [n.items[0].context_expr.func.attr for n in kernels[0].body
                       if isinstance(n, ast.With)]
            self.assertEqual(regions, ['cube', 'vector'])
            self.assertEqual(provenance['up_output_copy_adaptations'], 1)
            self.assertNotIn('torch', source)
            self.assertNotIn('CAST_FLOOR', source)
            self.assertNotIn('basic_mmad_kernel(', source)

    def test_stage_barrier_unconditional_and_ordered(self):
        from ops.kernel.ffn import generate_source
        for kind in ('basic', 'streamk'):
            fn = ast.parse(generate_source(kind)[0]).body[0]
            regions = [n for n in fn.body if isinstance(n, ast.With)]
            def calls(region):
                return [ast.unparse(n) for n in region.body if isinstance(n, ast.Expr)]
            cube, vector = map(calls, regions)
            expected_cube = ['tla.cross_core_wait_flag(up_ready, tla.arch.MTE2)',
                             'tla.cross_core_set_flag(all_aic, tla.arch.MTE2)',
                             'tla.cross_core_wait_flag(all_aic, tla.arch.MTE2)',
                             'tla.cross_core_set_flag(down_release, tla.arch.MTE2)']
            positions = [cube.index(s) for s in expected_cube]
            self.assertEqual(positions, sorted(positions))
            self.assertIn('tla.cross_core_set_flag(up_ready, tla.arch.MTE3)', vector)
            self.assertIn('tla.cross_core_wait_flag(down_release, tla.arch.MTE2)', vector)
            # Presence at region top-level includes no-tile cores and both AIVs
            self.assertEqual(generate_source(kind)[0], generate_source(kind)[0])

    def test_selection(self):
        from ops.host.dispatch import Selection, DependencyUnavailable, NotApplicable
        from ops.host.inputs import problem_from_shapes
        from ops.host.tiling import make_plan
        for kind in ('full_load_a', 'full_load_b'):
            with self.assertRaises(DependencyUnavailable):
                Selection(down_impl=kind).require_available()
        p = problem_from_shapes([64, 128], [256, 128], [64, 256])
        plan = make_plan(p, Selection(down_impl='streamk'), 8)
        self.assertEqual(plan.workspace_shape, (1024, 64))
        with self.assertRaises(NotApplicable):
            make_plan(p, Selection(down_impl='streamk'), 1)


class SearchTests(unittest.TestCase):
    def test_best_and_failure_filter(self):
        from performance.tune import search
        configs = [{'down_impl': k} for k in ('basic', 'streamk', 'full_load_a')]
        def evaluate(config):
            k = config['down_impl']
            return {'status': 'passed' if k != 'full_load_a' else 'unavailable',
                    'device_median_us': {'basic': 8, 'streamk': 4, 'full_load_a': None}[k]}
        records, best = search(configs, evaluate)
        self.assertTrue(best['complete'])
        self.assertEqual(best['configuration']['down_impl'], 'streamk')
        self.assertEqual(len(records), 3)

    def test_dry_run_no_best(self):
        from performance.tune import search
        _, best = search([{}], lambda c: {'status': 'planned'})
        self.assertIsNone(best['configuration'])

    def test_accuracy_failure_never_wins(self):
        from performance.tune import search
        def evaluate(c):
            return {'status': 'accuracy_failed' if c['down_impl'] == 'streamk' else 'passed',
                    'device_median_us': 1 if c['down_impl'] == 'streamk' else 10}
        _, best = search([{'down_impl': 'streamk'}, {'down_impl': 'basic'}], evaluate)
        self.assertEqual(best['configuration']['down_impl'], 'basic')

    def test_checkpoint_before_interruption(self):
        from performance.tune import search
        checkpoints = []
        def evaluate(c):
            if c['id'] == 2:
                raise KeyboardInterrupt()
            return {'status': 'passed', 'device_median_us': 10}
        with self.assertRaises(KeyboardInterrupt):
            search([{'id': 1}, {'id': 2}], evaluate,
                   lambda rows, best: checkpoints.append((list(rows), dict(best))))
        self.assertEqual(len(checkpoints[0][0]), 1)
        self.assertFalse(checkpoints[0][1]['complete'])


class ReportTests(unittest.TestCase):
    def test_summary_keeps_units_and_ignores_other_ops(self):
        import tempfile
        from pathlib import Path
        from performance.collect import extract_summary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'op_summary_0.csv'
            path.write_text('Op Name,Task Duration(us),aic_cube_ratio,aic_mte2_time(us)\n'
                            'ffn_kernel,10,0.7,4\nother,500,1,2\nffn_kernel,12,0.9,8\n')
            metrics = extract_summary(directory)
            self.assertEqual(metrics['task_duration_us'], 11)
            self.assertAlmostEqual(metrics['pipeline_utilization']['aic_cube_ratio'], 0.8)
            self.assertNotIn('aic_mte2_time(us)', metrics['pipeline_utilization'])

    def test_unknown_profiler_schema(self):
        import tempfile
        from performance.collect import extract_summary
        with tempfile.TemporaryDirectory() as directory:
            metrics = extract_summary(directory)
            self.assertIsNone(metrics['pipeline_utilization'])
            self.assertIsNone(metrics['task_duration_us'])

    def test_default_cli_is_cpu_only(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from ops.host.dispatch import ROOT
        script = '''
import sys
from performance.run import main
main(['--mode', 'search', '--output-root', sys.argv[1]])
assert not any(n.startswith(('torch_npu', 'catlass')) for n in sys.modules)
'''
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-c', script, directory], cwd=ROOT,
                           capture_output=True, text=True, check=True)
            run = next(Path(directory).iterdir())
            rows = json.loads((run / 'accuracy.json').read_text())
            self.assertEqual(len(rows), 12)
            self.assertFalse(json.loads((run / 'manifest.json').read_text())['device_executed'])
            best = json.loads((run / 'best_configs.json').read_text())
            self.assertTrue(all(value['configuration'] is None for value in best.values()))


if __name__ == '__main__':
    unittest.main()
