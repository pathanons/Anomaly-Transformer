#!/usr/bin/env python3
"""
Small experiment launcher to sweep combinations of hyperparameters.
It will:
 - create a configs/generated/ YAML per experiment (so you have a record)
 - create an experiment name based on the varied params
 - call `python main.py --config <generated_yaml> --exp_name <exp_name> --seed <seed>`

This launcher runs experiments sequentially. If you want parallel runs, wrap it with GNU parallel or modify it.
"""
import itertools
import os
import subprocess
import yaml
from datetime import datetime

# ---- User settings: modify these lists to sweep ----
SWEEPS = {
    'win_size': [110, 210],
    'd_model': [8, 16, 32],
    'n_heads': [4, 8],
    'e_layers': [2, 3],
    'd_ff': [32, 64]
}

# common base config to merge with
BASE_CONFIG = 'configs/example.yaml'
GENERATED_DIR = 'configs/generated'
SEED = 42
PYTHON = 'python'
MAIN = 'main.py'

os.makedirs(GENERATED_DIR, exist_ok=True)

# produce cartesian product of sweep values
keys = list(SWEEPS.keys())
values = [SWEEPS[k] for k in keys]
combos = list(itertools.product(*values))

print(f"Found {len(combos)} combinations")

for combo in combos:
    params = dict(zip(keys, combo))
    # create experiment name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = 'sweep_' + '_'.join([f"{k}{v}" for k, v in params.items()]) + '_' + timestamp

    # load base config and update with params
    with open(BASE_CONFIG, 'r') as f:
        base = yaml.safe_load(f) or {}
    base.update(params)
    base['seed'] = SEED

    # write generated config
    gen_path = os.path.join(GENERATED_DIR, f"{exp_name}.yaml")
    with open(gen_path, 'w') as f:
        yaml.safe_dump(base, f)

    # build command
    cmd = [PYTHON, MAIN, '--config', gen_path, '--exp_name', exp_name, '--seed', str(SEED)]
    print('\nRunning:', ' '.join(cmd))
    # run synchronously; replace subprocess.run with Popen if you want background
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"Experiment {exp_name} exited with code {proc.returncode}. Stopping further runs.")
        break

print('Done')
