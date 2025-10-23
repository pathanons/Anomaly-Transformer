This folder contains grid search configs for all combinations of:
- win_size: [50, 100, 150, 200, 250]
- d_model: [32, 64, 128, 256, 512]
- d_ff: [32, 64, 128, 256, 512]
- e_layers: [2, 3, 4, 5, 6]
- n_heads: [2, 4, 8, 12, 16]
- seed: 42
- model_save_path: /content/drive/MyDrive/anomaly_runs/winsize_sweep

File naming: win{win_size}_dm{d_model}_df{d_ff}_el{e_layers}_nh{n_heads}_seed42.yaml
Total configs: 3125
