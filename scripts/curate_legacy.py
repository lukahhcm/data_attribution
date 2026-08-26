#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASELINE_NOTES = {
    "uniform_epoch": (
        "sanity_only",
        "Valid historical budget baseline; rerun for shared-S0 paired tests.",
    ),
    "uniform_batch": (
        "sanity_only",
        "Valid historical batch baseline; rerun for shared-S0 paired tests.",
    ),
    "rho_batch_matched": (
        "rerun_required",
        "Historical RHO reference only; selection_batch_stats=false was not faithful to "
        "paper BN protocol.",
    ),
}

DIAGNOSTIC_NOTES = {
    "rho_epoch_persistent_u1": "Motivates update-interval study; not a V2 conclusion.",
    "rho_epoch_persistent_ub": "Old high-frequency update diagnostic; retired protocol.",
    "vf_epoch_persistent_u1": (
        "Motivates VF correctness gate; old fixed-alpha/one-pass protocol retired."
    ),
    "vf_epoch_persistent_ub": "Old high-frequency VF diagnostic; retired protocol.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild labelled legacy evidence extracts.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../data_attribution_main/artifacts/results_strict_v4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legacy_results"),
    )
    return parser.parse_args()


def write_extract(
    rows: list[dict[str, str]],
    methods: dict[str, tuple[str, str]],
    output: Path,
    source_label: str,
) -> None:
    base_fields = list(rows[0])
    fields = base_fields + ["reuse_status", "note", "source_report"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row["method"] not in methods:
                continue
            status, note = methods[row["method"]]
            writer.writerow(
                {**row, "reuse_status": status, "note": note, "source_report": source_label}
            )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.source / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("legacy summary is empty")

    source_label = "data_attribution_main/artifacts/results_strict_v4"
    write_extract(rows, BASELINE_NOTES, args.output / "baseline_summary.csv", source_label)
    diagnostics = {
        method: ("diagnostic_only", note) for method, note in DIAGNOSTIC_NOTES.items()
    }
    write_extract(rows, diagnostics, args.output / "failure_diagnostics.csv", source_label)

    audit = json.loads((args.source / "audit.json").read_text(encoding="utf-8"))
    source_audit = {
        "source": f"{source_label}/audit.json",
        "historical_audit": audit,
        "interpretation": (
            "Budget/metadata audit only; it does not validate the VF estimator or V2 protocol."
        ),
    }
    (args.output / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
