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
from ops.host.dispatch import read_kernel_source
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


class PerformanceRunner:
    """功能：串行管理case、候选配置、精度验证、性能采集及结果保存

    输入：parser()解析得到的Namespace，case文件及依赖须可读，输出目录须可写
    默认仅做dry-run；execute_npu为真时才准备NPU输入并编译、执行算子
    输出：run()返回退出码0或1，保存报告并实时打印进度；异常保存现场后继续抛出
    实例持有当前case和评估状态，只支持调用一次run()，不支持并发复用
    """

    def __init__(self, args):
        """输入CLI配置，校验参数并初始化运行目录和状态，不执行CPU golden或NPU计算"""
        self.args = args
        self.validate_arguments()
        self.cases = load_cases(args.cases)
        if args.case_id:
            unknown = set(args.case_id) - {c['case_id'] for c in self.cases}
            if unknown:
                raise ValueError(f'Unknown case IDs: {sorted(unknown)}')
            self.cases = [c for c in self.cases if c['case_id'] in args.case_id]
        self.configs = (
            combinations(args.up_candidates, args.down_candidates) if args.mode == 'search'
            else [{'up_impl': args.up_impl, 'down_impl': args.down_impl}]
        )
        for config in self.configs:
            Selection(**config)
        self.info = manifest()
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
        self.root = Path(args.output_root) / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'profiler').mkdir()
        self.info.update(arguments=vars(args), cases=self.cases, search_space=self.configs,
                         complete=False, device_executed=False)
        self.records, self.bests = [], {}
        self.case = None
        self.case_index = self.config_index = 0
        self.start_index = 0
        self.inputs = self.golden = self.staged = None
        self.active_evaluation = None
        self.started = False
        _json(self.root / 'manifest.json', self.info)

    def validate_arguments(self):
        """校验执行模式和计时次数，非法配置抛出ValueError，无返回值"""
        if self.args.execute_npu and self.args.dry_run:
            raise ValueError('--execute-npu and --dry-run are mutually exclusive')
        if self.args.warmup < 0 or self.args.repeats < 1:
            raise ValueError('Invalid timing counts')

    def log(self, message):
        """输入阶段消息，打印时间、case/配置序号并立即刷新，不在设备计时区间调用"""
        prefix = datetime.now().strftime('%H:%M:%S')
        if self.case is not None:
            prefix += f' case {self.case_index + 1}/{len(self.cases)} {self.case["case_id"]}'
        if self.config_index:
            prefix += f' 配置 {self.config_index}/{len(self.configs)}'
        print(f'[{prefix}] {message}', flush=True)

    def persist(self):
        """将当前记录、最佳配置和环境快照写入报告，无返回值，不修改搜索状态"""
        _json(self.root / 'accuracy.json', self.records)
        _json(self.root / 'best_configs.json', self.bests)
        fields = ['case_id', 'up_impl', 'down_impl', 'status', 'device_median_us',
                  'task_duration_us', 'pipeline_utilization', 'reason']
        with (self.root / 'performance.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.records:
                entry = {key: row.get(key) for key in fields}
                entry.update(row['configuration'])
                writer.writerow(entry)
        (self.root / 'analysis.md').write_text(report(self.records, self.bests))
        _json(self.root / 'manifest.json', self.info)

    def prepare_case(self):
        """为当前case生成CPU参考和固定NPU输入供全部候选复用，dry-run明确跳过"""
        self.inputs = self.golden = self.staged = None
        if not self.args.execute_npu:
            self.log('dry-run：跳过CPU golden、NPU计算及性能采集')
            return
        from validation.reference import ffn_torch

        self.log('进入CPU golden计算阶段')
        cpu_inputs = make_inputs(self.case)
        self.golden = ffn_torch(*cpu_inputs)
        self.log('CPU golden计算完成，进入分阶段舍入参考计算')
        self.staged = ffn_torch(*cpu_inputs, staged=True, rational=True)
        self.log('CPU分阶段舍入参考计算完成')
        import torch
        import torch_npu  # noqa: F401

        self.log(f'进入NPU输入准备阶段：{self.args.device}')
        # 在设备转移后构造非连续view，所有候选使用同一组固定输入
        self.info['device_executed'] = True
        self.inputs = make_inputs(self.case, self.args.device)
        torch.npu.synchronize(self.args.device)
        self.info.update(device_name=torch.npu.get_device_name(self.args.device),
                         cann_version=getattr(torch.version, 'cann', None))
        self.log('NPU输入准备完成')

    def evaluate(self, config):
        """输入一个确定的up/down配置，返回状态、精度及可用性能字段，不自动回退"""
        self.config_index += 1
        self.active_evaluation = {'case_id': self.case['case_id'], 'configuration': config}
        self.log(f'开始评估：{config["up_impl"]}＋GELU＋{config["down_impl"]}')
        row = {'case_id': self.case['case_id']}
        self.log('进入策略适用性与Tiling检查')
        try:
            plan = make_plan(case_problem(self.case), Selection(**config), self.args.block_num)
        except DependencyUnavailable as exc:
            self.log(f'跳过：实现不可用，{exc}')
            return dict(row, status='unavailable', reason=str(exc))
        except NotApplicable as exc:
            self.log(f'跳过：当前case不适用，{exc}')
            return dict(row, status='not_applicable', reason=str(exc))
        source, provenance = read_kernel_source(config['down_impl'])
        snapshot = 'ffn_source_snapshot.py'
        (self.root / snapshot).write_text(source)
        row.update(plan=plan.to_dict(), provenance=provenance, source_snapshot=snapshot)
        self.log('Tiling检查完成，源码快照已保存')
        if not self.args.execute_npu:
            return dict(row, status='planned', reason='CPU/static only; not compiled or executed on NPU')

        from ops.api import prepare_ffn, KernelCompilationError
        import torch

        self.log('进入NPU缓冲准备及Kernel编译阶段')
        before = perf_counter()
        try:
            prepared = prepare_ffn(*self.inputs, **config, block_num=self.args.block_num,
                                   layout=self.case.get('layout', 'linear'))
        except KernelCompilationError as exc:
            self.log(f'Kernel编译失败：{exc}')
            return dict(row, status='compile_failed', reason=str(exc))
        except Exception as exc:
            # 准备失败可能涉及设备上下文，不在该上下文继续下发
            row.update(status='preparation_failed', reason=f'{type(exc).__name__}: {exc}')
            self.records.append(dict(row, configuration=config))
            self.persist()
            raise
        torch.npu.synchronize(self.args.device)
        row['preparation_wall_ms'] = (perf_counter() - before) * 1000
        row.update(prepared.preparation)
        self.log('NPU缓冲准备及Kernel编译完成')
        if not self.validate_prepared(prepared, row):
            return row
        if not self.args.validate_only:
            self.collect_performance(prepared, row, config, provenance)
        else:
            self.log('仅精度验证模式：跳过预热、计时和profiler采集')
        return dict(row, status='passed')

    def validate_prepared(self, prepared, row):
        """输入已编译算子和记录字典，执行NPU验证并更新row，返回是否通过主golden判据"""
        from validation.test_ffn import compare

        self.log('进入NPU计算阶段：首次FFN执行')
        output = prepared.run().detach().cpu()
        self.log('NPU计算完成，输出已回传CPU，进入精度比较')
        row['accuracy'] = compare(output, self.golden)
        row['staged_accuracy'] = compare(output, self.staged)
        if not row['accuracy']['passed']:
            row.update(status='accuracy_failed', reason='FP32 golden comparison failed')
            self.log(f'精度验证失败：{row["accuracy"]}')
            return False
        self.log(f'精度验证通过：mean_rel={row["accuracy"]["mean_rel"]:.6g}')
        for repeat in range(2):
            self.log(f'进入NPU重复调用验证 {repeat + 1}/2')
            if not compare(prepared.run().detach().cpu(), self.golden)['passed']:
                row.update(status='accuracy_failed', reason='Repeated invocation failed')
                self.log(f'NPU重复调用验证 {repeat + 1}/2失败')
                return False
            self.log(f'NPU重复调用验证 {repeat + 1}/2完成并通过')
        return True

    def collect_performance(self, prepared, row, config, provenance):
        """输入已通过精度验证的算子，更新row的设备计时及可选profiler指标，无返回值"""
        from performance.collect import time_prepared

        row.update(time_prepared(prepared, self.args.warmup, self.args.repeats,
                                 progress=self.log))
        self.log(f'设备计时完成：中位数={row["device_median_us"]:.3f}μs')
        if self.args.profile:
            self.collect_profile(prepared, row, config, provenance)
        else:
            self.log('未启用--profile：跳过profiler采集')

    def collect_profile(self, prepared, row, config, provenance):
        """独立采集当前候选的profiler并提取指标，更新row；缺失接口记录为不可用"""
        from performance.collect import capture_profile, extract_task_duration, extract_summary

        directory = self.root / 'profiler' / f'case_{self.case_index}_{config["down_impl"]}'
        self.log('进入profiler采集阶段')
        try:
            profile_info = capture_profile(prepared, directory)
            profile_info['profiler_path'] = str(directory.relative_to(self.root))
            row.update(profile_info)
            self.log('profiler采集完成，进入task duration和流水指标提取')
            row.update(extract_summary(directory, provenance['entry_point']))
            traces = list(directory.rglob('trace_view.json'))
            if len(traces) == 1 and row.get('task_duration_us') is None:
                row.update(extract_task_duration(traces[0], provenance['entry_point']))
            self.log('profiler指标提取完成，缺失字段保留为不可用')
        except (AttributeError, ImportError) as exc:
            row['profiler_note'] = f'Profiler unavailable: {exc}'
            self.log(f'profiler不可用：{exc}')

    def checkpoint(self, current, best):
        """接收search的当前case记录和最佳配置，替换该case结果并落盘，不重复追加"""
        previous_count = len(self.records) - self.start_index
        self.records[self.start_index:] = current
        self.bests[self.case['case_id']] = dict(
            best, measured=bool(self.args.execute_npu and not self.args.validate_only))
        self.active_evaluation = None
        self.persist()
        if len(current) > previous_count:
            row = current[-1]
            self.log(f'配置评估结束：{row["status"]}，结果已保存')
        elif best['complete']:
            self.log('当前case的候选遍历完成，最佳配置记录已保存')

    def run_case(self, index, case):
        """输入从0开始的序号和case字典，完成该case全部候选，异常交由run保存现场"""
        self.case_index, self.case = index, case
        self.config_index = 0
        self.start_index = len(self.records)
        self.bests[case['case_id']] = {'configuration': None, 'complete': False, 'scope': self.configs}
        self.log(f'开始case：shape={case["shape"]}，dtype={case["dtype"]}')
        self.persist()
        self.prepare_case()
        search(self.configs, self.evaluate, self.checkpoint)
        self.log('case执行完成')

    def save_failure(self, exc):
        """输入异常对象，保存未完成状态及当前候选原因，不吞掉异常或标记运行成功"""
        reason = f'{type(exc).__name__}: {exc}'
        self.info['interrupted_or_failed'] = reason
        if self.active_evaluation is not None and not any(
                r['case_id'] == self.active_evaluation['case_id']
                and r['configuration'] == self.active_evaluation['configuration'] for r in self.records):
            self.records.append(dict(self.active_evaluation, status='interrupted_or_failed', reason=reason))
        self.log(f'运行中断或失败：{reason}，正在保存现场')
        self.persist()

    def run(self):
        """顺序执行全部case并生成报告，正常返回0，精度失败或无可用设备结果返回1"""
        if self.started:
            raise RuntimeError('PerformanceRunner.run() may only be called once')
        self.started = True
        mode = 'NPU执行' if self.args.execute_npu else 'dry-run，不执行NPU'
        self.log(f'开始运行：{len(self.cases)}个case，每个case有{len(self.configs)}个候选，{mode}')
        try:
            for index, case in enumerate(self.cases):
                self.run_case(index, case)
            self.info['complete'] = True
        except BaseException as exc:
            self.save_failure(exc)
            raise
        self.log('进入最终报告保存阶段')
        self.persist()
        self.log(f'全部case处理完成，报告已保存：{self.root}')
        if self.args.execute_npu and any(row['status'] == 'accuracy_failed' for row in self.records):
            return 1
        if self.args.execute_npu and not any(row['status'] == 'passed' for row in self.records):
            return 1
        return 0


def main(argv=None):
    """解析CLI参数并调用运行管理类，返回退出码，兼容精度验证入口"""
    return PerformanceRunner(parser().parse_args(argv)).run()


if __name__ == '__main__':
    raise SystemExit(main())
