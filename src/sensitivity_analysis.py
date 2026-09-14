"""Summarize split-seed sensitivity without treating seeds as participants."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from src.analysis import (
    calibration_difference,
    equal_dataset_point,
    protocol_difference,
    selective_difference,
)


def _seed_estimands(metrics: pd.DataFrame, selective: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for seed in sorted(metrics["seed"].unique()):
        seed_metrics = metrics[metrics["seed"].eq(seed)]
        seed_selective = selective[selective["seed"].eq(seed)]
        specs = [
            (
                "H1_disjoint_minus_mixed_macro_f1",
                protocol_difference(seed_metrics, "macro_f1", "raw"),
            ),
            (
                "H2_disjoint_minus_mixed_brier",
                protocol_difference(seed_metrics, "brier", "raw"),
            ),
            (
                "H3_calibrated_minus_raw_disjoint_brier",
                calibration_difference(seed_metrics, "brier"),
            ),
            (
                "H4_selective80_minus_full_error_disjoint",
                selective_difference(seed_metrics, seed_selective, 0.8),
            ),
        ]
        for name, differences in specs:
            rows.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "estimand": name,
                    "estimate": equal_dataset_point(differences),
                    "n_subjects": int(differences[["dataset", "subject"]].drop_duplicates().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _seed_ranks(metrics: pd.DataFrame, source: str) -> pd.DataFrame:
    raw = metrics[metrics["calibration"].eq("raw")]
    model_mean = (
        raw.groupby(["seed", "dataset", "protocol", "model"], observed=True, as_index=False)[
            "macro_f1"
        ]
        .mean()
    )
    model_mean["rank"] = model_mean.groupby(
        ["seed", "dataset", "protocol"], observed=True
    )["macro_f1"].rank(ascending=False, method="average")
    rows = []
    for (seed, dataset), group in model_mean.groupby(["seed", "dataset"], observed=True):
        wide = group.pivot(index="model", columns="protocol", values="rank")
        tau = kendalltau(wide["sample_mixed"], wide["subject_disjoint"])
        rows.append(
            {
                "source": source,
                "seed": int(seed),
                "dataset": dataset,
                "kendall_tau": float(tau.statistic),
                "kendall_p": float(tau.pvalue),
            }
        )
    return pd.DataFrame(rows)


def summarize_sensitivity(
    primary_dir: str | Path,
    sensitivity_dir: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    ranks = []
    for source, directory in (("primary", Path(primary_dir)), ("sensitivity", Path(sensitivity_dir))):
        metrics = pd.read_csv(directory / "subject_metrics.csv")
        selective = pd.read_csv(directory / "subject_selective.csv")
        frames.append(_seed_estimands(metrics, selective, source))
        ranks.append(_seed_ranks(metrics, source))
    estimates = pd.concat(frames, ignore_index=True)
    estimates.to_csv(output_dir / "sensitivity_estimands_by_seed.csv", index=False)
    ranges = (
        estimates.groupby("estimand", observed=True)["estimate"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )
    ranges.to_csv(output_dir / "sensitivity_estimand_ranges.csv", index=False)
    pd.concat(ranks, ignore_index=True).to_csv(
        output_dir / "sensitivity_rank_correlations.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", default="results/processed/primary")
    parser.add_argument("--sensitivity-dir", default="results/processed/sensitivity")
    parser.add_argument("--output-dir", default="results/processed/sensitivity_summary")
    args = parser.parse_args()
    summarize_sensitivity(args.primary_dir, args.sensitivity_dir, args.output_dir)


if __name__ == "__main__":
    main()
