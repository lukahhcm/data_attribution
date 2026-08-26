# V2 architecture and provenance

## New method layer

- `omega.py`: the only budget-state implementation;
- `f2sa.py`: penalty schedule, score sign, and convergence criterion;
- `selector.py`: neural two-inner optimization and global VF scoring;
- `rho.py`: faithful practical RHO batch scoring;
- `protocol.py`: common warm start, one-shot, and multi-round boundaries;
- `toy.py`: exact strongly-convex correctness problem.

No V2 module imports `data_attribution_main.main_study`, `vf_k_study`, or
`selection.update_weights`.

## Reused stable layer

The following files were copied from the old package on 2026-08-26 and should
be changed only for a documented data/model/training reason:

- `config.py`, `data.py`, `models.py`;
- `training.py`, `distributed.py`, `io.py`.

The only immediate data-layer change is the CIFAR-100 default split:
25k candidate / 20k clean holdout / 5k development. This preserves the old
candidate scale while removing the old holdout/development reuse.

## State boundary

The future runner owns target training and round orchestration. It must obtain
continuous updates from `OmegaState`, inner scores from `selector.py`, and
round boundaries from `ProtocolSpec`. It must never overwrite continuous
weights with a Top-B mask.
