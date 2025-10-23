#!/usr/bin/env bash
set -euo pipefail

SEED=42

for f in configs/generated/winsize*_seed${SEED}.yaml; do
  expname=$(basename "$f" .yaml)
  echo "===== Running experiment: $expname ====="
  python main.py --config "$f" --exp_name "$expname" --seed $SEED
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "Experiment $expname failed with exit code $rc"
    exit $rc
  fi
done

echo "All window size experiments finished."
