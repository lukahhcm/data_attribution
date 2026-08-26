# Data Attribution V2 experiments

This repository is the clean implementation root for the revised paper. It
does not import the retired `online-K`, `block-K`, U1/UB, or fixed-alpha VF
state machines.

The frozen experimental specification is [PLAN.md](PLAN.md). The ICLR LaTeX
draft and local PDF instructions are under [paper/](paper/).

## What is ready

- budget-feasible persistent `OmegaState` with projected and calibrated-sigmoid variants;
- the correctly signed finite-penalty F2SA loss-difference direction;
- a polynomial/fixed penalty schedule (`penalty` in config is manuscript $\alpha_r$);
- two neural inner steps for `theta_hat` and `theta_tilde`, with frozen BN buffers;
- the preregistered inner convergence tracker;
- shared warm-start and update-interval protocol primitives;
- faithful practical RHO large-batch scoring with noncommitting BN statistics;
- a strongly-convex exact-gradient gate and unit tests;
- curated historical results with explicit reuse status.

Stable dataset/model/training utilities were retained from the earlier
experiment package. New method logic lives only in `omega.py`, `f2sa.py`,
`selector.py`, `rho.py`, and `protocol.py`.

## First commands

```bash
python3.11 -m pip install -e '.[dev]'
pytest -q
python3.11 scripts/run_toy_gate.py
python3.11 scripts/make_manifest.py --config configs/preflight/mnist.yaml
```

The neural end-to-end runner is intentionally the next layer to build after
the correctness gate and final cluster/runtime interface are agreed. The
current package contains the tested mathematical and protocol primitives it
must call, preventing another monolithic runner from redefining VF semantics.

The remaining server-side implementation work and its acceptance checklist
are specified in Section 11 of [PLAN.md](PLAN.md).
