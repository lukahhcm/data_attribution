#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ONE_SHOT = {"uniform_fixed", "rho_one_shot", "vf_one_shot"}
MULTI_ROUND = {"uniform_round", "rho_multi", "vf_multi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an auditable paired-run manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--intervals", default="1,5,10,20")
    parser.add_argument(
        "--methods",
        default="uniform_fixed,uniform_round,rho_one_shot,vf_one_shot,rho_multi,vf_multi",
    )
    return parser.parse_args()


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    methods = comma_strings(args.methods)
    unknown = set(methods) - ONE_SHOT - MULTI_ROUND
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    rows = []
    for seed in comma_ints(args.seeds):
        shared_subset_seed = seed
        for method in methods:
            intervals: list[int | None] = [None]
            if method in MULTI_ROUND:
                intervals = comma_ints(args.intervals)
            for interval in intervals:
                run_id = f"{config['dataset']['name']}_s{seed}_{method}"
                if interval is not None:
                    run_id += f"_tau{interval}"
                rows.append(
                    {
                        "run_id": run_id,
                        "base_config": str(args.config),
                        "method": method,
                        "seed": seed,
                        "shared_initial_subset_seed": shared_subset_seed,
                        "update_interval": interval,
                    }
                )
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
