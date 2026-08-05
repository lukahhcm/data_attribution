#!/usr/bin/env bash
#SBATCH --job-name=data-attr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.out

set -euo pipefail

if [[ -z "${MANIFEST:-}" ]]; then
  echo "Set MANIFEST to a one-command-per-line file." >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must run as a Slurm array task." >&2
  exit 2
fi

COMMAND="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")"
if [[ -z "${COMMAND}" ]]; then
  echo "No command at zero-based manifest index ${SLURM_ARRAY_TASK_ID}." >&2
  exit 2
fi

echo "${COMMAND}"
bash -lc "${COMMAND}"

