"""Participant-level analysis for cross-fitted HAR probability predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.experiment import evaluate_probabilities
from src.metrics import expected_calibration_error


GROUP_COLUMNS = ["dataset", "protocol", "seed", "model", "subject"]


def _probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = sorted(column for column in frame if column.startswith(prefix))
    if len(columns) < 2:
        raise ValueError(f"fewer than two probability columns with prefix {prefix!r}")
    return columns


def compute_subject_metrics(
    predictions: pd.DataFrame, coverages: Iterable[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute discrimination/reliability and abstention metrics per participant."""
    all_raw_columns = _probability_columns(predictions, "p_raw_")
    all_calibrated_columns = _probability_columns(predictions, "p_cal_")
    metric_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []

    for group_values, group in predictions.groupby(GROUP_COLUMNS, sort=True, observed=True):
        identity = dict(zip(GROUP_COLUMNS, group_values))
        raw_columns = [column for column in all_raw_columns if group[column].notna().all()]
        calibrated_columns = [
            column for column in all_calibrated_columns if group[column].notna().all()
        ]
        if len(raw_columns) != len(calibrated_columns) or len(raw_columns) < 2:
            raise ValueError(f"invalid probability schema for group {identity}")
        n_classes = len(raw_columns)
        labels = group["y_true"].to_numpy(dtype=int)
        for calibration, columns in (
            ("raw", raw_columns),
            ("calibrated", calibrated_columns),
        ):
            probabilities = group[columns].to_numpy(dtype=float)
            values = evaluate_probabilities(labels, probabilities, n_classes)
            values["ece_10"] = expected_calibration_error(labels, probabilities, n_bins=10)
            metric_rows.append(
                {
                    **identity,
                    "calibration": calibration,
                    "n_windows": len(group),
                    **values,
                    "error": 1.0 - values["accuracy"],
                }
            )

        calibrated = group[calibrated_columns].to_numpy(dtype=float)
        confidence = calibrated.max(axis=1)
        predictions_class = calibrated.argmax(axis=1)
        for coverage in coverages:
            threshold_column = f"threshold_{coverage:g}"
            if threshold_column not in group:
                raise ValueError(f"missing threshold column {threshold_column}")
            thresholds = group[threshold_column].to_numpy(dtype=float)
            retained = confidence >= thresholds
            n_retained = int(retained.sum())
            selective_rows.append(
                {
                    **identity,
                    "target_coverage": float(coverage),
                    "n_windows": len(group),
                    "n_retained": n_retained,
                    "coverage": float(retained.mean()),
                    "selective_error": (
                        float(np.mean(predictions_class[retained] != labels[retained]))
                        if n_retained
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(selective_rows)


def compute_window_weighted_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute metric summaries in which every test window has equal weight."""
    all_raw_columns = _probability_columns(predictions, "p_raw_")
    all_calibrated_columns = _probability_columns(predictions, "p_cal_")
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "protocol", "seed", "model"]
    for group_values, group in predictions.groupby(
        group_columns, sort=True, observed=True
    ):
        identity = dict(zip(group_columns, group_values))
        raw_columns = [column for column in all_raw_columns if group[column].notna().all()]
        calibrated_columns = [
            column for column in all_calibrated_columns if group[column].notna().all()
        ]
        if len(raw_columns) != len(calibrated_columns) or len(raw_columns) < 2:
            raise ValueError(f"invalid probability schema for group {identity}")
        labels = group["y_true"].to_numpy(dtype=int)
        n_classes = len(raw_columns)
        for calibration, columns in (
            ("raw", raw_columns),
            ("calibrated", calibrated_columns),
        ):
            probabilities = group[columns].to_numpy(dtype=float)
            values = evaluate_probabilities(labels, probabilities, n_classes)
            values["ece_10"] = expected_calibration_error(
                labels, probabilities, n_bins=10
            )
            rows.append(
                {
                    **identity,
                    "calibration": calibration,
                    "n_windows": int(len(group)),
                    "n_subjects": int(group["subject"].nunique()),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_equal_dataset(
    differences: pd.DataFrame,
    *,
    draws: int = 5000,
    seed: int = 20260911,
) -> dict[str, float | int]:
    """Bootstrap participants within dataset, then weight datasets equally."""
    required = {"dataset", "subject", "value"}
    if not required.issubset(differences):
        raise ValueError(f"differences must contain {sorted(required)}")
    participant = (
        differences.dropna(subset=["value"])
        .groupby(["dataset", "subject"], observed=True, as_index=False)["value"]
        .mean()
    )
    if participant.empty:
        raise ValueError("no finite participant differences")
    arrays = [
        group["value"].to_numpy(dtype=float)
        for _, group in participant.groupby("dataset", sort=True, observed=True)
    ]
    point = equal_dataset_point(participant)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    for draw in range(draws):
        bootstrap[draw] = np.mean(
            [rng.choice(values, size=len(values), replace=True).mean() for values in arrays]
        )
    return {
        "estimate": point,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "draws": int(draws),
        "n_subjects": int(len(participant)),
        "n_datasets": int(participant["dataset"].nunique()),
    }


def equal_dataset_point(differences: pd.DataFrame) -> float:
    """Equal-person within dataset and equal-dataset pooled point estimate."""
    participant = (
        differences.dropna(subset=["value"])
        .groupby(["dataset", "subject"], observed=True, as_index=False)["value"]
        .mean()
    )
    if participant.empty:
        raise ValueError("no finite participant differences")
    return float(
        participant.groupby("dataset", observed=True)["value"].mean().mean()
    )


def cluster_sign_flip_p(
    differences: pd.DataFrame,
    *,
    draws: int = 5000,
    seed: int = 20260911,
) -> float:
    """Monte Carlo paired sign-flip test, preserving equal dataset weights."""
    participant = (
        differences.dropna(subset=["value"])
        .groupby(["dataset", "subject"], observed=True, as_index=False)["value"]
        .mean()
    )
    arrays = [
        group["value"].to_numpy(dtype=float)
        for _, group in participant.groupby("dataset", sort=True, observed=True)
    ]
    if not arrays:
        raise ValueError("no finite participant differences")
    observed = abs(float(np.mean([values.mean() for values in arrays])))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(draws):
        permuted = np.mean(
            [np.mean(values * rng.choice((-1.0, 1.0), size=len(values))) for values in arrays]
        )
        extreme += abs(float(permuted)) >= observed - 1e-15
    return float((extreme + 1) / (draws + 1))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, values[order] * (len(values) - np.arange(len(values))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def load_run_predictions(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in {"complete", "partial_failure"}:
        raise RuntimeError(f"run is not complete: status={manifest['status']}")
    frames: list[pd.DataFrame] = []
    for fit_path in sorted((run_dir / "fits").glob("*.json")):
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        if fit.get("status") != "success":
            continue
        prediction_path = run_dir.parent.parent / Path(fit["prediction_file"]).relative_to(
            "results/raw"
        )
        if not prediction_path.exists():
            prediction_path = run_dir.parents[2] / fit["prediction_file"]
        frame = pd.read_csv(prediction_path)
        for coverage, threshold in fit["thresholds"].items():
            frame[f"threshold_{float(coverage):g}"] = float(threshold)
        frames.append(frame)
    if not frames:
        raise ValueError("no successful prediction files found")
    predictions = pd.concat(frames, ignore_index=True)
    identity = ["dataset", "protocol", "seed", "model", "sample_id"]
    if predictions.duplicated(identity).any():
        raise ValueError("duplicate cross-fitted predictions detected")
    return predictions, manifest


def protocol_difference(
    subject_metrics: pd.DataFrame, metric: str, calibration: str
) -> pd.DataFrame:
    filtered = subject_metrics[subject_metrics["calibration"].eq(calibration)]
    wide = filtered.pivot(
        index=["dataset", "seed", "model", "subject"],
        columns="protocol",
        values=metric,
    )
    required = {"sample_mixed", "subject_disjoint"}
    if not required.issubset(wide.columns):
        raise ValueError("both evaluation protocols are required")
    result = wide.reset_index()
    result["value"] = result["subject_disjoint"] - result["sample_mixed"]
    return result


def calibration_difference(subject_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    filtered = subject_metrics[subject_metrics["protocol"].eq("subject_disjoint")]
    wide = filtered.pivot(
        index=["dataset", "seed", "model", "subject"],
        columns="calibration",
        values=metric,
    )
    result = wide.reset_index()
    result["value"] = result["calibrated"] - result["raw"]
    return result


def selective_difference(
    subject_metrics: pd.DataFrame,
    subject_selective: pd.DataFrame,
    target_coverage: float = 0.8,
) -> pd.DataFrame:
    full = subject_metrics[
        subject_metrics["protocol"].eq("subject_disjoint")
        & subject_metrics["calibration"].eq("calibrated")
    ][["dataset", "seed", "model", "subject", "error"]]
    selected = subject_selective[
        subject_selective["protocol"].eq("subject_disjoint")
        & np.isclose(subject_selective["target_coverage"], target_coverage)
    ][["dataset", "seed", "model", "subject", "selective_error"]]
    result = full.merge(selected, on=["dataset", "seed", "model", "subject"], validate="one_to_one")
    result["value"] = result["selective_error"] - result["error"]
    return result


def analyze_run(
    run_dir: str | Path,
    processed_dir: str | Path,
    tables_dir: str | Path,
    *,
    draws: int = 5000,
) -> None:
    run_dir = Path(run_dir)
    processed_dir = Path(processed_dir)
    tables_dir = Path(tables_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions, manifest = load_run_predictions(run_dir)
    coverages = sorted(
        {
            float(column.removeprefix("threshold_"))
            for column in predictions
            if column.startswith("threshold_")
        },
        reverse=True,
    )
    subject_metrics, subject_selective = compute_subject_metrics(predictions, coverages)
    subject_metrics.to_csv(processed_dir / "subject_metrics.csv", index=False)
    subject_selective.to_csv(processed_dir / "subject_selective.csv", index=False)

    window_weighted = compute_window_weighted_summary(predictions)
    window_weighted.to_csv(processed_dir / "model_summary_window_weighted.csv", index=False)
    window_weighted.to_csv(tables_dir / "table_model_summary_window_weighted.csv", index=False)

    summary = (
        subject_metrics.groupby(
            ["dataset", "protocol", "model", "calibration"], observed=True, as_index=False
        )[["accuracy", "macro_f1", "brier", "nll", "ece_10", "ece_15", "confidence_gap", "aurc"]]
        .mean()
    )
    summary.to_csv(processed_dir / "model_summary_subject_equal.csv", index=False)
    summary.to_csv(tables_dir / "table_model_summary.csv", index=False)

    seed = int(min(manifest.get("completed_fits") and predictions["seed"].unique()))
    primary_metrics = subject_metrics[subject_metrics["seed"].eq(seed)]
    primary_selective = subject_selective[subject_selective["seed"].eq(seed)]
    estimand_specs = [
        ("H1_disjoint_minus_mixed_macro_f1", protocol_difference(primary_metrics, "macro_f1", "raw")),
        ("H2_disjoint_minus_mixed_brier", protocol_difference(primary_metrics, "brier", "raw")),
        ("H3_calibrated_minus_raw_disjoint_brier", calibration_difference(primary_metrics, "brier")),
        ("H4_selective80_minus_full_error_disjoint", selective_difference(primary_metrics, primary_selective, 0.8)),
    ]
    estimand_rows: list[dict[str, Any]] = []
    difference_frames: list[pd.DataFrame] = []
    for index, (name, differences) in enumerate(estimand_specs):
        differences = differences.copy()
        differences["estimand"] = name
        difference_frames.append(differences)
        estimate = cluster_bootstrap_equal_dataset(
            differences, draws=draws, seed=20260911 + index
        )
        sign_flip_p = cluster_sign_flip_p(
            differences, draws=draws, seed=20261911 + index
        )
        estimand_rows.append({"estimand": name, **estimate, "sign_flip_two_sided_p": sign_flip_p})
    estimands = pd.DataFrame(estimand_rows)
    estimands["holm_adjusted_p"] = holm_adjust(estimands["sign_flip_two_sided_p"])
    estimands.to_csv(processed_dir / "primary_estimands.csv", index=False)
    estimands.to_csv(tables_dir / "table_primary_estimands.csv", index=False)
    pd.concat(difference_frames, ignore_index=True).to_csv(
        processed_dir / "participant_differences.csv", index=False
    )

    ranks = summary[summary["calibration"].eq("raw")].copy()
    ranks["rank_macro_f1"] = ranks.groupby(["dataset", "protocol"], observed=True)[
        "macro_f1"
    ].rank(ascending=False, method="average")
    rank_rows: list[dict[str, Any]] = []
    for dataset, group in ranks.groupby("dataset", observed=True):
        wide = group.pivot(index="model", columns="protocol", values="rank_macro_f1")
        tau = kendalltau(wide["sample_mixed"], wide["subject_disjoint"])
        rank_rows.append(
            {
                "dataset": dataset,
                "kendall_tau": float(tau.statistic),
                "kendall_p": float(tau.pvalue),
            }
        )
    ranks.to_csv(processed_dir / "model_ranks.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(processed_dir / "rank_correlations.csv", index=False)
    ranks.to_csv(tables_dir / "table_model_ranks.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processed-dir", default="results/processed/primary")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    analyze_run(args.run_dir, args.processed_dir, args.tables_dir, draws=args.draws)


if __name__ == "__main__":
    main()
