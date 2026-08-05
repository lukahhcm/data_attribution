#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally scheduler-agnostic. On Slurm, map each line to
# one array task/GPU. Override DATA_ROOT and OUTPUT_ROOT for the cluster.
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

for dataset in mnist cifar10 cifar100; do
  for noise in 0.0 0.1; do
    if [[ "${dataset}" == "cifar100" ]]; then
      retentions=(0.2 0.5)
    else
      retentions=(0.1 0.2 0.5)
    fi
    for retention in "${retentions[@]}"; do
      for method in rho_vf_one_step iterative_vf; do
        for seed in 1 2 3 4 5; do
          da-select --config "configs/${dataset}.yaml" \
            "experiment.method=${method}" \
            "experiment.seed=${seed}" \
            "experiment.output_root=${OUTPUT_ROOT}" \
            "dataset.root=${DATA_ROOT}" \
            "dataset.noise_rate=${noise}" \
            "selection.retention=${retention}"
        done
      done
    done
  done
done

