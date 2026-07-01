"""Report the best val metric of a run, without touching training code.

Reads `ckpt/best.ckpt`, which custom_train.py saves every time val
`metric_best` improves, and prints the `best_metric` value stored inside it.

Usage:
    python run/read_best_metric.py results/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego/42
"""
import argparse
import os

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='Path to a run dir, e.g. results/<dataset>/<exp>/<seed>')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, 'ckpt', 'best.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No best.ckpt found at {ckpt_path}')

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    best_metric = ckpt.get('best_metric')

    metric_name = 'f1'
    cfg_path = os.path.join(os.path.dirname(os.path.normpath(args.run_dir)), 'config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            metric_name = yaml.safe_load(f).get('metric_best', metric_name)

    print(f'Run dir: {args.run_dir}')
    print(f'Best val {metric_name} (from ckpt/best.ckpt): {best_metric}')


if __name__ == '__main__':
    main()
