# Curated legacy evidence

These files are small, labelled extracts of
`data_attribution_main/artifacts/results_strict_v4`.

- `baseline_summary.csv`: historical uniform baselines are sanity ranges, not
  paired V2 results. Historical `rho_batch_matched` rows require rerunning
  because the saved configurations used `selection_batch_stats: false`.
- `failure_diagnostics.csv`: persistent RHO/VF drops retained only as
  motivation for the update-interval and correctness studies.
- `source_audit.json`: preserves the old 288-run audit result and explicitly
  limits its interpretation to budget/metadata consistency.

No historical number in this directory may be pooled with V2 seeds for a
paired interval or significance test. Original 660MB run artifacts stay in
place and are referenced rather than duplicated.

The extracts are reproducible from the V2 root with:

```bash
python3.11 scripts/curate_legacy.py
```
