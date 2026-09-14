"""Future explicit NPU timings, distinct from profiler task duration."""
import statistics
import csv
import math
from pathlib import Path


def time_prepared(prepared, warmup=5, repeats=20):
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup >= 0 and repeats >= 1 required')
    import torch
    device = prepared.storage[0].device
    with torch.npu.device(device):
        for _ in range(warmup):
            prepared.run()
        torch.npu.synchronize(device)
        samples = []
        for _ in range(repeats):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            prepared.run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise RuntimeError('Invalid device timing; refusing to rank this candidate')
    return {'device_median_us': statistics.median(samples), 'device_samples_us': samples,
            'warmup': warmup, 'repeats': repeats, 'task_duration_us': None,
            'pipeline_utilization': None,
            'profiler_note': 'Unavailable until profiler trace is captured and inspected'}


def capture_profile(prepared, directory):
    """Separate pass: record a single FFN execution; never mix into event ranking.

    Raw trace is retained. Pipeline metrics depend on installed profiler support.
    """
    import torch
    import torch_npu
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    options = {}
    metric_note = 'Pipeline counter API unavailable; collect trace timing only'
    profiler = torch_npu.profiler
    experimental = getattr(profiler, '_ExperimentalConfig', None)
    metrics = getattr(profiler, 'AiCMetrics', None)
    levels = getattr(profiler, 'ProfilerLevel', None)
    metric = getattr(metrics, 'PipeUtilization', None)
    level = getattr(levels, 'Level1', None)
    if experimental is not None and metric is not None and level is not None:
        try:
            options['experimental_config'] = experimental(aic_metrics=metric, profiler_level=level)
            metric_note = 'PipeUtilization requested; inspect exported counters for availability'
        except TypeError as exc:
            metric_note = f'Installed experimental profiler API differs: {exc}'
    with torch.npu.device(prepared.storage[0].device):
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
            record_shapes=True,
            **options,
        ):
            prepared.run()
            torch.npu.synchronize()
    return {'profiler_path': str(directory), 'profiler_note': metric_note}


def extract_summary(directory, kernel_name='ffn_kernel'):
    """Extract only explicitly labelled task/us and pipeline ratios from op_summary.

    Export header names/units are preserved. Unknown schemas stay unavailable.
    """
    task_times, counters, files = [], {}, []
    for path in Path(directory).rglob('op_summary*.csv'):
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                names = [value for key, value in row.items()
                         if key and key.strip().lower() in ('op name', 'kernel name', 'name')]
                if not any(kernel_name.lower() in (name or '').lower() for name in names):
                    continue
                files.append(str(path.relative_to(directory)))
                for header, value in row.items():
                    if not header:
                        continue
                    key = header.strip().lower().replace(' ', '').replace('μ', 'u').replace('µ', 'u')
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(numeric):
                        continue
                    if key in ('taskduration(us)', 'task_duration(us)') and numeric >= 0:
                        task_times.append(numeric)
                    # Never label pipe time, bandwidth or rates as utilization
                    if (any(pipe in key for pipe in ('cube', 'vec', 'mte1', 'mte2', 'mte3', 'fixpipe'))
                            and any(label in key for label in ('ratio', 'utilization'))):
                        counters.setdefault(header, []).append(numeric)
    return {'task_duration_us': statistics.median(task_times) if task_times else None,
            'pipeline_utilization': {k: statistics.median(v) for k, v in counters.items()} or None,
            'summary_sources': sorted(set(files)),
            'summary_note': 'Medians of matching FFN rows; pipeline header units preserved, no inferred metrics'}


def extract_task_duration(trace_path, kernel_name='ffn_kernel'):
    """Read Chrome trace device events only; do not mistake Host API time for task time."""
    import json
    data = json.loads(Path(trace_path).read_text())
    events = data.get('traceEvents', [])
    durations = [e['dur'] for e in events if e.get('ph') == 'X'
                 and e.get('cat', '').lower() in ('kernel', 'npu', 'aicore')
                 and kernel_name in e.get('name', '') and isinstance(e.get('dur'), (float, int))]
    if not durations:
        return {'task_duration_us': None, 'task_duration_note': 'No matching device kernel trace event'}
    return {'task_duration_us': statistics.median(durations), 'profiled_task_count': len(durations),
            'task_duration_note': 'Median of matching Chrome trace device durations, microseconds'}
