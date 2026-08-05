#!/usr/bin/env python3
"""Generate one-command-per-line manifests for a Slurm/LSF job array."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


RETENTIONS = {
    "mnist": (0.1, 0.2, 0.5),
    "cifar10": (0.1, 0.2, 0.5),
    "cifar100": (0.2, 0.5),
}


def command(parts: list[str]) -> str:
    return shlex.join(parts)


def selector_commands(args: argparse.Namespace) -> list[str]:
    commands = []
    for dataset in args.datasets:
        for noise in args.noise_rates:
            for retention in RETENTIONS[dataset]:
                for method in ("rho_vf_one_step", "iterative_vf"):
                    for seed in args.seeds:
                        commands.append(
                            command(
                                [
                                    args.launcher,
                                    "--config",
                                    f"configs/{dataset}.yaml",
                                    f"experiment.method={method}",
                                    f"experiment.seed={seed}",
                                    f"experiment.output_root={args.output_root}",
                                    f"dataset.root={args.data_root}",
                                    f"dataset.noise_rate={noise}",
                                    f"selection.retention={retention}",
                                ]
                            )
                        )
    return commands


def baseline_commands(args: argparse.Namespace) -> list[str]:
    commands = []
    for dataset in args.datasets:
        for noise in args.noise_rates:
            for retention in RETENTIONS[dataset]:
                methods = ("full", "uniform") if retention == RETENTIONS[dataset][0] else ("uniform",)
                for method in methods:
                    for seed in args.seeds:
                        commands.append(
                            command(
                                [
                                    "da-eval",
                                    "--config",
                                    f"configs/{dataset}.yaml",
                                    f"experiment.method={method}",
                                    f"experiment.seed={seed}",
                                    f"experiment.output_root={args.output_root}",
                                    f"dataset.root={args.data_root}",
                                    f"dataset.noise_rate={noise}",
                                    f"selection.retention={retention}",
                                ]
                            )
                        )
    return commands


def rho_commands(args: argparse.Namespace) -> list[str]:
    commands = []
    for dataset in args.datasets:
        for noise in args.noise_rates:
            for method in ("uniform_online", "original_rho"):
                for seed in args.seeds:
                    commands.append(
                        command(
                            [
                                "da-rho",
                                "--config",
                                f"configs/{dataset}.yaml",
                                f"experiment.method={method}",
                                f"experiment.seed={seed}",
                                f"experiment.output_root={args.output_root}",
                                f"dataset.root={args.data_root}",
                                f"dataset.noise_rate={noise}",
                            ]
                        )
                    )
    return commands


def evaluator_commands(args: argparse.Namespace) -> list[str]:
    commands = []
    for dataset in args.datasets:
        for noise in args.noise_rates:
            for retention in RETENTIONS[dataset]:
                for method in ("rho_vf_one_step", "iterative_vf"):
                    for seed in args.seeds:
                        run = (
                            Path(args.output_root)
                            / dataset
                            / f"noise_{noise:g}"
                            / f"retention_{retention:g}"
                            / method
                            / f"seed_{seed}"
                        )
                        commands.append(
                            command(
                                [
                                    "da-eval",
                                    "--config",
                                    f"configs/{dataset}.yaml",
                                    "--selection-run",
                                    str(run),
                                    f"experiment.method={method}",
                                    f"experiment.seed={seed}",
                                    f"experiment.output_root={args.output_root}",
                                    f"dataset.root={args.data_root}",
                                    f"dataset.noise_rate={noise}",
                                    f"selection.retention={retention}",
                                ]
                            )
                        )
    return commands


def ablation_commands(args: argparse.Namespace) -> list[str]:
    """One-factor-at-a-time CIFAR-10 ablations; the R=10 run is the control."""
    variants: list[tuple[str, list[str]]] = []
    variants.extend(
        (f"rounds_{value}", [f"selection.rounds={value}"])
        for value in (1, 2, 5, 10, 20)
    )
    variants.extend(
        (f"alpha_schedule_{value}", [f"selection.alpha.schedule={value}"])
        for value in ("constant", "geometric")
    )
    variants.extend(
        (f"alpha_max_{value:g}", [f"selection.alpha.max_multiplier={value}"])
        for value in (1.0, 10.0)
    )
    variants.extend(
        (f"omega_lr_{value:g}", [f"selection.omega_lr={value}"])
        for value in (0.03, 0.3)
    )
    variants.extend(
        (f"epochs_{value}", [f"selector.epochs={value}"])
        for value in (50, 200)
    )
    commands = []
    for run_name, overrides in variants:
        for seed in args.seeds:
            commands.append(
                command(
                    [
                        args.launcher,
                        "--config",
                        "configs/cifar10.yaml",
                        "experiment.method=iterative_vf",
                        f"experiment.run_name={run_name}",
                        f"experiment.seed={seed}",
                        f"experiment.output_root={args.output_root}",
                        f"dataset.root={args.data_root}",
                        "dataset.noise_rate=0.1",
                        "selection.retention=0.2",
                        *overrides,
                    ]
                )
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("selectors", "evaluators", "baselines", "rho", "ablations"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(RETENTIONS), default=list(RETENTIONS))
    parser.add_argument("--noise-rates", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--launcher", default="da-select", help="Replace with a torchrun wrapper if desired")
    args = parser.parse_args()
    builders = {
        "selectors": selector_commands,
        "evaluators": evaluator_commands,
        "baselines": baseline_commands,
        "rho": rho_commands,
        "ablations": ablation_commands,
    }
    commands = builders[args.stage](args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"wrote {len(commands)} commands to {args.output}")


if __name__ == "__main__":
    main()
