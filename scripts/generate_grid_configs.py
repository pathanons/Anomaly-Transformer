import itertools
import yaml
import os

win_sizes = [50, 100, 150, 200, 250]
d_models = [32, 128, 512]
d_ffs = [32, 128, 512]
n_heads = [2, 4, 8, 12, 16]
e_layers = [2, 4, 6]
seed = 42
outdir = 'configs/generated'
os.makedirs(outdir, exist_ok=True)


# Default config values from example.yaml/main.py
default_config = {
    'dataset': 'SAW',
    'data_path': './dataset/SAW2',
    'mode': 'train',
    'input_c': 1,
    'output_c': 1,
    'batch_size': 64,
    'num_epochs': 10,
    'lr': 1e-3,
    'anormly_ratio': 0.5,
    'isLabelled': True,
    'dropout': 0.1,
    'activation': 'gelu',
    'pretrained_model': None,
    'model_save_path': '/content/drive/MyDrive/anomaly_runs/winsize_sweep'
}

for win_size, d_model, d_ff, n_head, e_layer in itertools.product(win_sizes, d_models, d_ffs, n_heads, e_layers):
    fname = f'win{win_size}_dm{d_model}_df{d_ff}_el{e_layer}_nh{n_head}_seed{seed}.yaml'
    fpath = os.path.join(outdir, fname)
    config = default_config.copy()
    config.update({
        'win_size': win_size,
        'd_model': d_model,
        'd_ff': d_ff,
        'n_heads': n_head,
        'e_layers': e_layer,
        'seed': seed
    })
    with open(fpath, 'w') as f:
        yaml.dump(config, f)
print('Grid configs generated:', len(win_sizes)*len(d_models)*len(d_ffs)*len(n_heads)*len(e_layers))
