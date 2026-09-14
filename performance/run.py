"""Manual/search entry. Default is a CPU-only plan; --execute-npu opts in."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from common.cases import load_cases, case_problem, make_inputs
from common.environment import manifest
from ops.host.dispatch import Selection, DependencyUnavailable, NotApplicable
from ops.host.tiling import make_plan
from ops.kernel.ffn import generate_source
from performance.tune import combinations, search
from performance.analyze import report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', default='configs/v1_cases.json')
    p.add_argument('--case-id', action='append', help='Only selected case IDs (repeatable)')
    p.add_argument('--mode', choices=('manual', 'search'), default='manual')
    p.add_argument('--up-impl', default='basic')
    p.add_argument('--down-impl', default='basic')
    p.add_argument('--up-candidates', nargs='+', default=['basic'])
    p.add_argument('--down-candidates', nargs='+', default=['basic', 'full_load_a', 'full_load_b', 'streamk'])
    p.add_argument('--block-num', type=int, default=8)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--execute-npu', action='store_true', help='Authorize device preparation, compilation and execution')
    p.add_argument('--profile', action='store_true', help='Separate profiler pass after unprofiled timing')
    p.add_argument('--dry-run', action='store_true', help='Explicit CPU-only mode (also the default)')
    p.add_argument('--output-root', default='results')
    p.add_argument('--validate-only', action='store_true', help=argparse.SUPPRESS)
    return p


def _json(path, data):
    # Runtime output, not source edits. Atomic replacement supports checkpoints.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.execute_npu and args.dry_run:
        raise ValueError('--execute-npu and --dry-run are mutually exclusive')
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError('Invalid timing counts')
    cases = load_cases(args.cases)
    if args.case_id:
        unknown = set(args.case_id) - {c['case_id'] for c in cases}
        if unknown:
            raise ValueError(f'Unknown case IDs: {sorted(unknown)}')
        cases = [c for c in cases if c['case_id'] in args.case_id]
    configs = (combinations(args.up_candidates, args.down_candidates) if args.mode == 'search'
               else [{'up_impl': args.up_impl, 'down_impl': args.down_impl}])
    for config in configs:
        Selection(**config)  # Reject typos before creating a run
    info = manifest()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
    root = Path(args.output_root) / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / 'profiler').mkdir()
    info.update(arguments=vars(args), cases=cases, search_space=configs, complete=False,
                device_executed=False)
    records, bests = [], {}
    active_evaluation = None
    _json(root / 'manifest.json', info)

    def persist():
        _json(root / 'accuracy.json', records)
        _json(root / 'best_configs.json', bests)
        fields = ['case_id', 'up_impl', 'down_impl', 'status', 'device_median_us',
                  'task_duration_us', 'pipeline_utilization', 'reason']
        with (root / 'performance.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in records:
                entry = {key: row.get(key) for key in fields}
                entry.update(row['configuration'])
                writer.writerow(entry)
        (root / 'analysis.md').write_text(report(records, bests))
        _json(root / 'manifest.json', info)

    try:
        for case_index, case in enumerate(cases):
            inputs = golden = staged = None
            if args.execute_npu:
                import torch
                import torch_npu  # noqa: F401
                from validation.reference import ffn_torch
                # Generate/golden once on CPU, then move the same values for every candidate
                cpu_inputs = make_inputs(case)
                golden = ffn_torch(*cpu_inputs)
                staged = ffn_torch(*cpu_inputs, staged=True, rational=True)
                # Construct non-contiguous views AFTER transfer: Tensor.to may densify slices
                inputs = make_inputs(case, args.device)
                info.update(device_executed=True, device_name=torch.npu.get_device_name(args.device),
                            cann_version=getattr(torch.version, 'cann', None))
            start_index = len(records)
            bests[case['case_id']] = {'configuration': None, 'complete': False, 'scope': configs}
            persist()

            def evaluate(config):
                nonlocal active_evaluation
                active_evaluation = {'case_id': case['case_id'], 'configuration': config}
                row = {'case_id': case['case_id']}
                try:
                    plan = make_plan(case_problem(case), Selection(**config), args.block_num)
                except DependencyUnavailable as exc:
                    return dict(row, status='unavailable', reason=str(exc))
                except NotApplicable as exc:
                    return dict(row, status='not_applicable', reason=str(exc))
                source, provenance = generate_source(config['down_impl'])
                generated = f'kernel_{config["down_impl"]}.py'
                (root / generated).write_text(source)
                row.update(plan=plan.to_dict(), provenance=provenance, generated_source=generated)
                if not args.execute_npu:
                    return dict(row, status='planned', reason='CPU/static only; not compiled or executed on NPU')
                from ops.api import prepare_ffn, KernelCompilationError
                from validation.test_ffn import compare
                from performance.collect import time_prepared, capture_profile, extract_task_duration, extract_summary
                before = perf_counter()
                try:
                    prepared = prepare_ffn(*inputs, **config, block_num=args.block_num,
                                           layout=case.get('layout', 'linear'))
                except KernelCompilationError as exc:
                    return dict(row, status='compile_failed', reason=str(exc))
                except Exception as exc:
                    # May include allocation/copy faults: do not keep using a possibly bad device context
                    row.update(status='preparation_failed', reason=f'{type(exc).__name__}: {exc}')
                    records.append(dict(row, configuration=config))
                    persist()
                    raise
                torch.npu.synchronize(args.device)
                row['preparation_wall_ms'] = (perf_counter() - before) * 1000
                row.update(prepared.preparation)
                # Includes allocation, auxiliary copies and JIT, not called task duration
                output = prepared.run().detach().cpu()
                row['accuracy'] = compare(output, golden)
                row['staged_accuracy'] = compare(output, staged)
                if not row['accuracy']['passed']:
                    return dict(row, status='accuracy_failed', reason='FP32 golden comparison failed')
                # Reuse buffers and flags over repeated invocations; actual device regression only on opt-in
                for _ in range(2):
                    if not compare(prepared.run().detach().cpu(), golden)['passed']:
                        return dict(row, status='accuracy_failed', reason='Repeated invocation failed')
                if not args.validate_only:
                    row.update(time_prepared(prepared, args.warmup, args.repeats))
                    if args.profile:
                        directory = root / 'profiler' / f'case_{case_index}_{config["down_impl"]}'
                        try:
                            profile_info = capture_profile(prepared, directory)
                            profile_info['profiler_path'] = str(directory.relative_to(root))
                            row.update(profile_info)
                            row.update(extract_summary(directory))
                            traces = list(directory.rglob('trace_view.json'))
                            if len(traces) == 1 and row.get('task_duration_us') is None:
                                row.update(extract_task_duration(traces[0]))
                        except (AttributeError, ImportError) as exc:
                            row['profiler_note'] = f'Profiler unavailable: {exc}'
                return dict(row, status='passed')

            def checkpoint(current, best):
                nonlocal active_evaluation
                records[start_index:] = current
                bests[case['case_id']] = dict(best, measured=bool(args.execute_npu and not args.validate_only))
                active_evaluation = None
                persist()

            search(configs, evaluate, checkpoint)
        info['complete'] = True
    except BaseException as exc:
        info['interrupted_or_failed'] = f'{type(exc).__name__}: {exc}'
        if active_evaluation is not None and not any(
                r['case_id'] == active_evaluation['case_id']
                and r['configuration'] == active_evaluation['configuration'] for r in records):
            records.append(dict(active_evaluation, status='interrupted_or_failed',
                                reason=info['interrupted_or_failed']))
        persist()
        raise
    persist()
    print(f'Run saved: {root}')
    print('NPU execution enabled' if args.execute_npu else 'CPU/static only; no NPU calls')
    if args.execute_npu and any(row['status'] == 'accuracy_failed' for row in records):
        return 1
    if args.execute_npu and not any(row['status'] == 'passed' for row in records):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
