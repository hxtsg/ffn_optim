"""Evidence-limited reports: no invented pipeline utilization or stage timing."""


def report(records, best):
    lines = ['# 性能分析', '', '计时口径：不带profiler的设备Event中位数用于排序，task duration另列', '',
             '| case | 下投影 | 状态 | Event中位数/μs | task duration/μs |',
             '| --- | --- | --- | --- | --- |']
    for row in records:
        lines.append('| ' + ' | '.join(str(v) if v is not None else '不可用' for v in (
            row['case_id'], row['configuration']['down_impl'], row['status'],
            row.get('device_median_us'), row.get('task_duration_us'))) + ' |')
    lines += ['', '## 结论与待验证项', '',
              '- 最优配置仅限本次case、输入、设备、版本和搜索空间，不代表全局最优',
              '- 未采集或工具未提供的流水利用率标为不可用，不能从总时长推算',
              '- 单task总时长不能拆成上投影、GELU、同步和下投影耗时',
              '- 待结合trace确认：单AIV的GELU处理、逐tile握手、hidden读写和Stream-K归约是否为瓶颈',
              '- 后续候选：两AIV分担GELU、缓冲流水化、降低padding成本；尚无实测收益结论',
              '- 当前未接入A/B全载，case全集与性能标杆待补充', '']
    return '\n'.join(lines)
