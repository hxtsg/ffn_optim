"""Serial exhaustive search over an explicit implementation Cartesian product."""
import itertools
import math


def combinations(up, down):
    if not up or not down:
        raise ValueError('Search spaces must be non-empty')
    return [{'up_impl': u, 'down_impl': d} for u, d in itertools.product(dict.fromkeys(up), dict.fromkeys(down))]


def search(configurations, evaluate, checkpoint=None):
    records = []
    best = {'configuration': None, 'device_median_us': None, 'complete': False,
            'ranking_metric': 'unprofiled_device_event_median_us',
            'scope': configurations}
    for configuration in configurations:
        row = dict(evaluate(configuration), configuration=configuration)
        records.append(row)
        value = row.get('device_median_us')
        if (row['status'] == 'passed' and isinstance(value, (int, float))
                and math.isfinite(value) and value > 0
                and (best['device_median_us'] is None or value < best['device_median_us'])):
            best.update(configuration=configuration, device_median_us=value)
        if checkpoint:
            checkpoint(records, best)
    best['complete'] = True
    if checkpoint:
        checkpoint(records, best)
    return records, best
