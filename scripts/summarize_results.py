"""Collect results/<model>/metrics_<testset>.csv (written by srbench/test.py) into one benchmark table.

    python scripts/summarize_results.py                 # all runs under results/
    python scripts/summarize_results.py --set S2Maxar_test_clean --results results
"""
import argparse
import csv
import glob
import math
import os.path as osp
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'results'))
    parser.add_argument('--set', default=None, help='only this test set (e.g. S2Maxar_test_clean)')
    args = parser.parse_args()

    rows, metrics = [], []
    for path in sorted(glob.glob(osp.join(args.results, '*', 'metrics_*.csv'))):
        model = osp.basename(osp.dirname(path))
        test_set = osp.basename(path)[len('metrics_'):-len('.csv')]
        if args.set and test_set != args.set:
            continue
        with open(path) as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        names = [k for k in data[0] if k != 'key']
        metrics += [m for m in names if m not in metrics]
        row = {'model': model, 'set': test_set, 'n': len(data)}
        for m in names:
            vals = [float(r[m]) for r in data if r[m] not in ('', 'inf') and math.isfinite(float(r[m]))]
            row[m] = f'{statistics.fmean(vals):.4f} ± {statistics.pstdev(vals):.3f}' if vals else '-'
        rows.append(row)

    if not rows:
        print(f'no metrics_*.csv under {args.results}')
        return
    header = ['model', 'set', 'n'] + metrics
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(str(row.get(h, '-')) for h in header) + ' |')


if __name__ == '__main__':
    main()
