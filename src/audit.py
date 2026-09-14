"""Independent reconstruction and integrity checks for saved experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.analysis import load_run_predictions
from src.experiment import _load_dataset, split_digest
from src.splits import make_rotating_splits


def audit_run(run_dir: str | Path, project_root: str | Path = ".") -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    requested_folds = config.get("folds", list(range(int(config["n_folds"]))))
    expected_fits = (
        len(config["datasets"])
        * len(config["protocols"])
        * len(config["seeds"])
        * len(requested_folds)
        * len(config["models"])
    )
    checks: dict[str, Any] = {
        "manifest_complete": manifest["status"] == "complete",
        "expected_fit_count": len(manifest["completed_fits"]) == expected_fits,
        "no_failed_fits": len(manifest["failed_fits"]) == 0,
        "split_digests_reproduced": True,
        "prediction_counts_complete": True,
        "probabilities_finite_and_normalized": True,
        "temperature_preserves_argmax": True,
        "crossfit_predictions_unique": True,
    }

    registry = {
        (item["dataset"], item["protocol"], int(item["seed"]), int(item["fold"])): item
        for item in manifest["split_registry"]
    }
    dataset_sizes: dict[str, int] = {}
    for dataset in config["datasets"]:
        bundle = _load_dataset(project_root, dataset, config["dataset_specs"][dataset])
        dataset_sizes[dataset] = bundle.n_samples
        classes = {label: index for index, label in enumerate(sorted(np.unique(bundle.y)))}
        labels = np.asarray([classes[label] for label in bundle.y], dtype=int)
        for seed in config["seeds"]:
            for protocol in config["protocols"]:
                splits = make_rotating_splits(
                    labels,
                    bundle.subjects,
                    n_folds=int(config["n_folds"]),
                    seed=int(seed),
                    protocol=protocol,
                )
                for split in splits:
                    if split.fold not in requested_folds:
                        continue
                    saved = registry.get((dataset, protocol, int(seed), split.fold))
                    reproduced = split_digest(split.train, split.calibration, split.test)
                    if saved is None or saved["split_id"] != reproduced:
                        checks["split_digests_reproduced"] = False

    predictions, _ = load_run_predictions(run_dir)
    identity = ["dataset", "protocol", "seed", "model", "sample_id"]
    checks["crossfit_predictions_unique"] = not bool(predictions.duplicated(identity).any())
    counts = predictions.groupby(["dataset", "protocol", "seed", "model"], observed=True).size()
    for (dataset, _protocol, _seed, _model), count in counts.items():
        if len(requested_folds) == int(config["n_folds"]) and count != dataset_sizes[dataset]:
            checks["prediction_counts_complete"] = False

    for dataset, group in predictions.groupby("dataset", observed=True):
        raw_columns = [
            column
            for column in sorted(c for c in predictions if c.startswith("p_raw_"))
            if group[column].notna().all()
        ]
        calibrated_columns = [
            column
            for column in sorted(c for c in predictions if c.startswith("p_cal_"))
            if group[column].notna().all()
        ]
        raw = group[raw_columns].to_numpy(dtype=float)
        calibrated = group[calibrated_columns].to_numpy(dtype=float)
        if (
            not np.isfinite(raw).all()
            or not np.isfinite(calibrated).all()
            or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-7)
            or not np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-7)
        ):
            checks["probabilities_finite_and_normalized"] = False
        if not np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1)):
            checks["temperature_preserves_argmax"] = False

    passed = all(value is True for value in checks.values())
    return {
        "run_id": manifest["run_id"],
        "passed": passed,
        "expected_fits": expected_fits,
        "observed_successful_fits": len(manifest["completed_fits"]),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    audit = audit_run(args.run_dir, args.project_root)
    serialized = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
