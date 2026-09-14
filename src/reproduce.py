"""Independent numerical comparison for complete saved experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import load_run_predictions


IDENTITY_COLUMNS = ["dataset", "protocol", "seed", "model", "sample_id"]
IGNORED_COLUMNS = {"run_id"}
EXACT_VALUE_COLUMNS = {
    "fold",
    "split_id",
    "subject",
    "y_true",
    "y_true_name",
    "y_pred_raw",
    "y_pred_calibrated",
}


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing prediction identity columns: {missing}")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("duplicate prediction identities")
    return frame.sort_values(IDENTITY_COLUMNS, kind="stable").reset_index(drop=True)


def compare_prediction_frames(
    reference: pd.DataFrame, candidate: pd.DataFrame, atol: float = 1e-12
) -> dict[str, Any]:
    """Compare predictions while allowing a new immutable run identifier."""
    reference = _normalise(reference)
    candidate = _normalise(candidate)

    if not reference[IDENTITY_COLUMNS].equals(candidate[IDENTITY_COLUMNS]):
        raise ValueError("prediction identities differ")

    reference_columns = set(reference.columns) - IGNORED_COLUMNS
    candidate_columns = set(candidate.columns) - IGNORED_COLUMNS
    if reference_columns != candidate_columns:
        missing = sorted(reference_columns - candidate_columns)
        extra = sorted(candidate_columns - reference_columns)
        raise ValueError(f"prediction schemas differ: missing={missing}, extra={extra}")

    columns = sorted(reference_columns - set(IDENTITY_COLUMNS))
    numerical_columns: list[str] = []
    exact_columns: list[str] = []
    differences: dict[str, float] = {}
    exact_mismatches: dict[str, int] = {}

    for column in columns:
        ref_column = reference[column]
        candidate_column = candidate[column]
        if (
            column not in EXACT_VALUE_COLUMNS
            and pd.api.types.is_numeric_dtype(ref_column)
            and pd.api.types.is_numeric_dtype(candidate_column)
        ):
            numerical_columns.append(column)
            delta = np.abs(
                ref_column.to_numpy(dtype=float)
                - candidate_column.to_numpy(dtype=float)
            )
            differences[column] = (
                float(np.nanmax(delta))
                if delta.size and np.isfinite(delta).any()
                else 0.0
            )
        else:
            exact_columns.append(column)
            equal = (ref_column == candidate_column) | (
                ref_column.isna() & candidate_column.isna()
            )
            mismatch_count = int((~equal).sum())
            if mismatch_count:
                exact_mismatches[column] = mismatch_count

    numerical_passed = all(
        np.allclose(
            reference[column].to_numpy(dtype=float),
            candidate[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
        for column in numerical_columns
    )
    max_abs_difference = max(differences.values(), default=0.0)
    return {
        "passed": bool(numerical_passed and not exact_mismatches),
        "atol": float(atol),
        "n_predictions": int(len(reference)),
        "n_numerical_columns": len(numerical_columns),
        "n_exact_columns": len(exact_columns),
        "max_abs_difference": float(max_abs_difference),
        "column_max_abs_difference": differences,
        "exact_mismatches": exact_mismatches,
    }


def compare_runs(
    reference_dir: str | Path, candidate_dir: str | Path, atol: float = 1e-12
) -> dict[str, Any]:
    """Load, validate, and compare two completed run directories."""
    reference_dir = Path(reference_dir).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    reference, reference_manifest = load_run_predictions(reference_dir)
    candidate, candidate_manifest = load_run_predictions(candidate_dir)
    comparison = compare_prediction_frames(reference, candidate, atol=atol)
    config_hash_match = (
        reference_manifest.get("config_sha256")
        == candidate_manifest.get("config_sha256")
    )
    comparison.update(
        {
            "passed": bool(comparison["passed"] and config_hash_match),
            "reference_run_id": reference_manifest.get("run_id"),
            "candidate_run_id": candidate_manifest.get("run_id"),
            "reference_status": reference_manifest.get("status"),
            "candidate_status": candidate_manifest.get("status"),
            "reference_config_sha256": reference_manifest.get("config_sha256"),
            "candidate_config_sha256": candidate_manifest.get("config_sha256"),
            "config_hash_match": bool(config_hash_match),
        }
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = compare_runs(arguments.reference, arguments.candidate, arguments.atol)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
