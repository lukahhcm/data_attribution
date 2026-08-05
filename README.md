# Data attribution experiments

This package implements the frozen protocol in [PLAN.md](PLAN.md). It uses plain
PyTorch and supports both one-process runs and `torchrun` DDP.

## Installation

Install PyTorch using the command appropriate for the cluster CUDA version, then
install this package:

```bash
python -m pip install -e '.[dev]'
```

## Selector

```bash
da-select --config configs/cifar10.yaml \
  experiment.method=iterative_vf \
  dataset.noise_rate=0.1 \
  selection.retention=0.2 \
  experiment.seed=1
```

DDP:

```bash
torchrun --standalone --nproc-per-node=4 -m data_attribution_exp.train_selector \
  --config configs/cifar10.yaml experiment.method=iterative_vf
```

The selector writes `selected_indices.pt`. Train the independent evaluator with:

```bash
da-eval --config configs/cifar10.yaml \
  --selection-run outputs/cifar10/iterative_vf/seed_1
```

Original online RHO:

```bash
da-rho --config configs/cifar10.yaml experiment.method=original_rho
```

Configuration overrides use dotted `key=value` syntax. Values are parsed as
YAML, so numbers, booleans, lists, and null work naturally.

## Cluster sweep

The first cluster pass contains both clean and 10% symmetric-label-noise runs
for MNIST, CIFAR-10, and CIFAR-100. Generate separate manifests so evaluator
jobs are submitted only after selector jobs finish:

```bash
mkdir -p manifests logs
python scripts/make_manifest.py --stage selectors --output manifests/selectors.txt
python scripts/make_manifest.py --stage evaluators --output manifests/evaluators.txt
python scripts/make_manifest.py --stage baselines --output manifests/baselines.txt
python scripts/make_manifest.py --stage rho --output manifests/rho.txt
python scripts/make_manifest.py --stage ablations --seeds 1 2 3 --output manifests/ablations.txt
```

Submit a zero-based array whose upper bound is `line_count - 1`, for example:

```bash
N=$(wc -l < manifests/selectors.txt)
sbatch --array="0-$((N-1))%32" --export=ALL,MANIFEST=manifests/selectors.txt scripts/slurm_array.sh
```

Use the scheduler dependency returned by that selector submission for the
evaluator array. The preferred scaling mode is one independent configuration
per GPU. For an unusually slow single selector, replace `da-select` with a
site-specific `torchrun --nproc-per-node=... -m data_attribution_exp.train_selector`
launcher and ensure all configured global batch sizes divide the GPU count.

Selector runs checkpoint after every weight-update boundary. Resume with
`experiment.resume=/absolute/path/to/selector_checkpoint.pt`; the model,
optimizer, scheduler, AMP scaler, weights, and round counters are restored.
