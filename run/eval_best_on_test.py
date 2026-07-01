"""Evaluate a saved `best.ckpt` against the test set -- read-only inference.

Does not train or modify anything: it rebuilds the model architecture from
the run's config, loads the weights from `ckpt/best.ckpt`, and runs a single
forward pass over the test loader (no backward, no optimizer) to report the
real test metrics for that checkpoint.

Side effect: like every eval pass in this codebase, it appends one line to
`<run_dir>/test/stats.json` (tagged with epoch=-1 so it's easy to tell apart
from real training epochs). No checkpoint or training file is touched.

Usage:
    python run/eval_best_on_test.py \
        --cfg results/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego/config.yaml \
        --run_id 42
"""
import argparse
import os
import sys

# Allow running this script directly (e.g. `python run/eval_best_on_test.py`)
# from any working directory: the `fraudGT` package lives at the repo root,
# one level above this file, and is only importable via `python -m
# fraudGT.main` normally -- there is no `pip install -e .` for it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric import seed_everything

import fraudGT  # noqa: registers custom modules
from fraudGT.graphgym.checkpoint import MODEL_STATE, BEST_METRIC
from fraudGT.graphgym.config import cfg, set_cfg, assert_cfg
from fraudGT.graphgym.loader import create_loader
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.utils.device import auto_select_device
from fraudGT.logger import create_logger
from fraudGT.utils import custom_set_run_dir
from fraudGT.train.custom_train import eval_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', dest='cfg_file', required=True,
                         help="Path to the experiment's config.yaml "
                              '(results/<dataset>/<exp>/config.yaml).')
    parser.add_argument('--run_id', type=int, required=True,
                         help='Seed/run id whose ckpt/best.ckpt to evaluate, e.g. 42.')
    parser.add_argument('--gpu', type=int, default=-1,
                         help='GPU index, or -1 to auto-select (default).')
    args = parser.parse_args()

    set_cfg(cfg)
    cfg.merge_from_file(args.cfg_file)
    assert_cfg(cfg)
    custom_set_run_dir(cfg, args.run_id)
    cfg.seed = args.run_id
    cfg.run_id = args.run_id
    seed_everything(cfg.seed)

    if args.gpu == -1:
        auto_select_device(strategy='greedy')
    else:
        cfg.device = f'cuda:{args.gpu}'

    ckpt_path = os.path.join(cfg.run_dir, 'ckpt', 'best.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No best.ckpt found at {ckpt_path}')

    loaders, dataset = create_loader(returnDataset=True)
    model = create_model(dataset=dataset)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device(cfg.device))
    model.load_state_dict(ckpt[MODEL_STATE])
    print(f'Loaded weights from {ckpt_path} '
          f'(best val {cfg.metric_best} at save time: {ckpt.get(BEST_METRIC)})')

    loggers = create_logger()
    test_loader = loaders[-1]  # create_loader always returns [train, val, test]
    eval_epoch(loggers[-1], test_loader, model, split='test')
    test_stats = loggers[-1].write_epoch(-1)

    print('\nReal test metrics for this checkpoint:')
    for k, v in test_stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
