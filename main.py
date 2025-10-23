import os
import sys
import argparse
import yaml

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(os.path.join(config.model_save_path, config.dataset, str(config.win_size), 'checkpoint.pth'))) and (config.mode == 'train'):
        mkdir(os.path.join(config.model_save_path, config.dataset, str(config.win_size)))
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training & data
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--isLabelled', type=bool, default=False)

    # model hyperparameters (can be set via CLI or YAML config)
    parser.add_argument('--d_model', type=int, default=8, help='model hidden size')
    parser.add_argument('--n_heads', type=int, default=5, help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=8, help='feedforward hidden size')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')

    # experiment control
    parser.add_argument('--seed', type=int, default=0, help='random seed for experiments')
    parser.add_argument('--exp_name', type=str, default=None, help='optional experiment name to create folder under model_save_path/experiments/')

    # config file
    parser.add_argument('--config', type=str, default=None, help='path to YAML config file')

    args = parser.parse_args()

    # If a YAML config is provided, load it and merge: CLI args take precedence.
    if args.config:
        try:
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to read config file {args.config}: {e}")
            raise

        # Helper: find argparse Action for a dest
        def find_action(dest_name):
            for a in parser._actions:
                if a.dest == dest_name:
                    return a
            return None

        # For each item in YAML: set it only when user did NOT provide it on CLI.
        for k, v in cfg.items():
            action = find_action(k)
            # If action exists, check whether user provided any of its option strings on the CLI
            provided_on_cli = False
            if action is not None and hasattr(action, 'option_strings'):
                for opt in action.option_strings:
                    if opt in sys.argv:
                        provided_on_cli = True
                        break

            if provided_on_cli:
                # CLI value wins; skip YAML value
                continue

            # Try to cast YAML value to the argparse type (if action.type is provided)
            cast_val = v
            if action is not None and getattr(action, 'type', None) is not None:
                try:
                    cast_val = action.type(v)
                except Exception:
                    # if casting fails, keep original YAML value
                    cast_val = v

            # Finally set into args (even if dest didn't previously exist)
            try:
                setattr(args, k, cast_val)
            except Exception:
                # ignore invalid keys
                pass

    config = args

    # Print final configuration
    args_dict = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args_dict.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # If an experiment name was provided, nest the model_save_path under experiments/<exp_name>
    if getattr(config, 'exp_name', None):
        config.model_save_path = os.path.join(config.model_save_path, 'experiments', config.exp_name)

    # Ensure reproducible seed is set before data loaders and model are created
    import random as _random
    _random.seed(config.seed)
    import numpy as _np
    _np.random.seed(config.seed)
    import torch as _torch
    _torch.manual_seed(config.seed)
    try:
        _torch.cuda.manual_seed_all(config.seed)
    except Exception:
        pass

    # Save the final merged config to the model folder for reproducibility
    cfg_dir = os.path.join(config.model_save_path, config.dataset, str(config.win_size))
    try:
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, 'config_used.yaml'), 'w') as cf:
            yaml.safe_dump(args_dict, cf, default_flow_style=False)
    except Exception as e:
        print(f"Warning: failed to write config_used.yaml to {cfg_dir}: {e}")

    main(config)
