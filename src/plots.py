"""Generate publication figures and their auditable source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import load_run_predictions


MODEL_ORDER = ["logistic", "random_forest", "extra_trees", "hist_gradient_boosting"]
MODEL_LABELS = {
    "logistic": "Logistic",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist. boosting",
}
COLORS = dict(zip(MODEL_ORDER, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))


def weighted_reliability_bins(
    frame: pd.DataFrame, probability_columns: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """Reliability bins where each participant has total weight one."""
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == frame["y_true"].to_numpy(dtype=int)
    subject_counts = frame.groupby("subject", observed=True)["subject"].transform("size")
    weights = 1.0 / subject_counts.to_numpy(dtype=float)
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    working = pd.DataFrame(
        {
            "bin": bin_index,
            "confidence": confidence,
            "correct": correct.astype(float),
            "weight": weights,
            "subject": frame["subject"].to_numpy(),
        }
    )
    rows = []
    for bin_number, group in working.groupby("bin", sort=True, observed=True):
        weight = group["weight"].to_numpy(dtype=float)
        rows.append(
            {
                "bin": int(bin_number),
                "bin_lower": bin_number / n_bins,
                "bin_upper": (bin_number + 1) / n_bins,
                "mean_confidence": float(np.average(group["confidence"], weights=weight)),
                "observed_accuracy": float(np.average(group["correct"], weights=weight)),
                "subject_weight": float(weight.sum()),
                "n_windows": int(len(group)),
                "n_subjects": int(group["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_protocol_shift(summary: pd.DataFrame, output_dir: Path) -> None:
    raw = summary[summary["calibration"].eq("raw")]
    datasets = list(raw["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for axis, (metric, label) in zip(
        axes, (("macro_f1", "Subject-equal macro-F1"), ("brier", "Multiclass Brier score"))
    ):
        for dataset_index, dataset in enumerate(datasets):
            for model in MODEL_ORDER:
                group = raw[(raw["dataset"].eq(dataset)) & (raw["model"].eq(model))]
                values = []
                for protocol in ("sample_mixed", "subject_disjoint"):
                    value = group.loc[group["protocol"].eq(protocol), metric]
                    values.append(float(value.iloc[0]) if len(value) else np.nan)
                axis.plot(
                    [0, 1],
                    values,
                    color=COLORS[model],
                    marker="o" if dataset_index == 0 else "s",
                    linestyle="-" if dataset_index == 0 else "--",
                    linewidth=1.7,
                    alpha=0.9,
                    label=f"{dataset} · {MODEL_LABELS[model]}",
                )
        axis.set_xticks([0, 1], ["Sample-mixed", "Subject-disjoint"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Evaluation protocol changes discrimination and probability quality")
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure1_protocol_shift")


def plot_calibration_effect(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary["protocol"].eq("subject_disjoint")]
    fig, axis = plt.subplots(figsize=(6.2, 4.3))
    for dataset_index, dataset in enumerate(data["dataset"].drop_duplicates()):
        for model in MODEL_ORDER:
            group = data[(data["dataset"].eq(dataset)) & (data["model"].eq(model))]
            values = []
            for calibration in ("raw", "calibrated"):
                value = group.loc[group["calibration"].eq(calibration), "brier"]
                values.append(float(value.iloc[0]) if len(value) else np.nan)
            axis.plot(
                [0, 1],
                values,
                color=COLORS[model],
                marker="o" if dataset_index == 0 else "s",
                linestyle="-" if dataset_index == 0 else "--",
                linewidth=1.7,
                label=f"{dataset} · {MODEL_LABELS[model]}",
            )
    axis.set_xticks([0, 1], ["Raw", "Temperature-scaled"])
    axis.set_ylabel("Subject-equal multiclass Brier score")
    axis.set_title("Calibration effect for unseen users")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure2_calibration_effect")


def plot_risk_coverage(selective: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    data = selective[selective["protocol"].eq("subject_disjoint")]
    source = (
        data.groupby(["dataset", "model", "target_coverage"], observed=True, as_index=False)[
            ["coverage", "selective_error"]
        ]
        .mean()
        .sort_values(["dataset", "model", "coverage"])
    )
    datasets = list(source["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.1), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for model in MODEL_ORDER:
            group = source[(source["dataset"].eq(dataset)) & (source["model"].eq(model))]
            axis.plot(
                group["coverage"],
                group["selective_error"],
                marker="o",
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_title(dataset)
        axis.set_xlabel("Achieved coverage")
        axis.set_ylabel("Subject-equal selective error")
        axis.grid(alpha=0.25)
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Risk–coverage transfer to unseen users")
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure3_risk_coverage")
    return source


def plot_reliability(
    predictions: pd.DataFrame, output_dir: Path, n_bins: int = 10
) -> pd.DataFrame:
    data = predictions[predictions["protocol"].eq("subject_disjoint")]
    datasets = list(data["dataset"].drop_duplicates())
    all_raw_columns = sorted(column for column in data if column.startswith("p_raw_"))
    all_calibrated_columns = sorted(column for column in data if column.startswith("p_cal_"))
    rows = []
    fig, axes = plt.subplots(
        len(datasets), len(MODEL_ORDER), figsize=(13.5, 3.3 * len(datasets)), squeeze=False
    )
    for row_index, dataset in enumerate(datasets):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            group = data[(data["dataset"].eq(dataset)) & (data["model"].eq(model))]
            raw_columns = [column for column in all_raw_columns if group[column].notna().all()]
            calibrated_columns = [
                column for column in all_calibrated_columns if group[column].notna().all()
            ]
            for calibration, columns, style in (
                ("raw", raw_columns, "--"),
                ("calibrated", calibrated_columns, "-"),
            ):
                bins = weighted_reliability_bins(group, columns, n_bins=n_bins)
                bins["dataset"] = dataset
                bins["model"] = model
                bins["calibration"] = calibration
                rows.append(bins)
                axis.plot(
                    bins["mean_confidence"],
                    bins["observed_accuracy"],
                    marker="o",
                    linestyle=style,
                    label=calibration.capitalize(),
                )
            axis.plot([0, 1], [0, 1], color="0.5", linewidth=1, linestyle=":")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title(f"{dataset}\n{MODEL_LABELS[model]}")
            if row_index == len(datasets) - 1:
                axis.set_xlabel("Mean confidence")
            if column_index == 0:
                axis.set_ylabel("Observed accuracy")
            axis.grid(alpha=0.2)
    axes[0, -1].legend(frameon=False, loc="lower right")
    fig.suptitle("Participant-equal reliability for subject-disjoint evaluation")
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure4_reliability")
    return pd.concat(rows, ignore_index=True)


def generate_figures(
    run_dir: str | Path, processed_dir: str | Path, output_dir: str | Path
) -> None:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(processed_dir / "model_summary_subject_equal.csv")
    selective = pd.read_csv(processed_dir / "subject_selective.csv")
    predictions, _ = load_run_predictions(run_dir)
    plot_protocol_shift(summary, output_dir)
    plot_calibration_effect(summary, output_dir)
    risk_source = plot_risk_coverage(selective, output_dir)
    risk_source.to_csv(processed_dir / "figure3_risk_coverage_source.csv", index=False)
    reliability_source = plot_reliability(predictions, output_dir)
    reliability_source.to_csv(processed_dir / "figure4_reliability_source.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processed-dir", default="results/processed/primary")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()
    generate_figures(args.run_dir, args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
