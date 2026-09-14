"""Leakage-controlled experiment runner with immutable per-fit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from src.calibration import TemperatureScaler, confidence_threshold
from src.data import DatasetBundle, load_uci_har, load_wisdm_arff
from src.metrics import (
    area_under_risk_coverage,
    expected_calibration_error,
    multiclass_brier_score,
    selective_metrics,
)
from src.models import build_model
from src.splits import make_rotating_splits


def aligned_predict_proba(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Align a classifier's probability columns to the global integer classes."""
    partial = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)
    if partial.shape != (len(X), len(model_classes)):
        raise ValueError("predict_proba shape is inconsistent with model.classes_")
    if np.any(model_classes < 0) or np.any(model_classes >= n_classes):
        raise ValueError("model class falls outside global class range")
    probabilities = np.zeros((len(X), n_classes), dtype=float)
    probabilities[:, model_classes] = partial
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("model produced an all-zero probability row")
    return probabilities / row_sums


def fit_with_thread_limit(
    model: Any, X: np.ndarray, y: np.ndarray, *, limit: int = 1
) -> Any:
    """Fit under an explicit numerical-library thread limit."""
    if limit < 1:
        raise ValueError("numeric thread limit must be at least one")
    with threadpool_limits(limits=limit):
        return model.fit(X, y)


def evaluate_probabilities(
    labels: np.ndarray, probabilities: np.ndarray, n_classes: int
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=np.arange(n_classes), average="macro", zero_division=0)
        ),
        "brier": multiclass_brier_score(labels, probabilities, n_classes),
        "nll": float(-np.log(clipped).mean()),
        "ece_15": expected_calibration_error(labels, probabilities, n_bins=15),
        "confidence_gap": float(confidence.mean() - (predictions == labels).mean()),
        "aurc": area_under_risk_coverage(labels, probabilities),
    }


def split_digest(train: np.ndarray, calibration: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train), ("calibration", calibration), ("test", test)):
        digest.update(name.encode())
        digest.update(np.asarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _load_dataset(project_root: Path, name: str, spec: dict[str, Any]) -> DatasetBundle:
    path = project_root / spec["path"]
    if name == "uci_har":
        return load_uci_har(path)
    if name == "wisdm_watch_accel":
        return load_wisdm_arff(
            path,
            device=spec.get("device", "watch"),
            sensor=spec.get("sensor", "accel"),
        )
    raise ValueError(f"unknown dataset: {name}")


def _subject_list(values: np.ndarray, indices: np.ndarray) -> list[str]:
    return sorted(str(value) for value in np.unique(values[indices]))


def _prediction_frame(
    bundle: DatasetBundle,
    test_indices: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": bundle.sample_ids[test_indices],
            "subject": bundle.subjects[test_indices],
            "y_true": labels,
            "y_true_name": label_names[labels],
            "y_pred_raw": raw.argmax(axis=1),
            "y_pred_calibrated": calibrated.argmax(axis=1),
            "confidence_raw": raw.max(axis=1),
            "confidence_calibrated": calibrated.max(axis=1),
        }
    )
    for key, value in metadata.items():
        frame[key] = value
    for class_index, class_name in enumerate(label_names):
        frame[f"p_raw_{class_index:02d}"] = raw[:, class_index]
        frame[f"p_cal_{class_index:02d}"] = calibrated[:, class_index]
    return frame


def run_experiment(config_path: str | Path, project_root: str | Path = ".") -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_digest = _config_hash(config)
    numeric_thread_limit = int(config.get("numeric_thread_limit", 1))
    run_id = config.get("run_id") or (
        f"{config['kind']}_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}_{config_digest[:8]}"
    )
    run_dir = project_root / "results" / "raw" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    (run_dir / "fits").mkdir()
    shutil.copyfile(config_path, run_dir / "config.yaml")
    log_path = project_root / "logs" / "runs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = datetime.fromisoformat(config["deadline_at"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": config["kind"],
        "status": "running",
        "started_at": _now(),
        "ended_at": None,
        "pid": os.getpid(),
        "config_sha256": config_digest,
        "command": " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "numeric_thread_limit": numeric_thread_limit,
        },
        "completed_fits": [],
        "failed_fits": [],
        "split_registry": [],
        "dataset_registry": {},
    }
    _write_json_atomic(run_dir / "manifest.json", manifest)

    metrics_rows: list[dict[str, Any]] = []
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"{_now()} START {run_id}\n")
        for dataset_name in config["datasets"]:
            bundle = _load_dataset(project_root, dataset_name, config["dataset_specs"][dataset_name])
            encoder = LabelEncoder().fit(bundle.y)
            encoded_y = encoder.transform(bundle.y)
            label_names = np.asarray(encoder.classes_, dtype=str)
            manifest["dataset_registry"][dataset_name] = {
                "n_samples": bundle.n_samples,
                "n_features": bundle.n_features,
                "n_subjects": int(len(np.unique(bundle.subjects))),
                "n_classes": int(len(label_names)),
                "classes": label_names.tolist(),
                "metadata": bundle.metadata,
                "source_archive_sha256": config["dataset_specs"][dataset_name].get(
                    "source_archive_sha256"
                ),
            }
            _write_json_atomic(run_dir / "manifest.json", manifest)

            for seed in config["seeds"]:
                for protocol in config["protocols"]:
                    splits = make_rotating_splits(
                        encoded_y,
                        bundle.subjects,
                        n_folds=int(config["n_folds"]),
                        seed=int(seed),
                        protocol=protocol,
                    )
                    requested_folds = config.get("folds")
                    if requested_folds is not None:
                        splits = [split for split in splits if split.fold in requested_folds]
                    for split in splits:
                        split_info = {
                            "dataset": dataset_name,
                            "protocol": protocol,
                            "seed": int(seed),
                            "fold": split.fold,
                            "split_id": split_digest(split.train, split.calibration, split.test),
                            "n_train": len(split.train),
                            "n_calibration": len(split.calibration),
                            "n_test": len(split.test),
                            "train_subjects": _subject_list(bundle.subjects, split.train),
                            "calibration_subjects": _subject_list(
                                bundle.subjects, split.calibration
                            ),
                            "test_subjects": _subject_list(bundle.subjects, split.test),
                        }
                        manifest["split_registry"].append(split_info)

                        for model_name in config["models"]:
                            if datetime.now().astimezone() >= deadline:
                                manifest["status"] = "deadline_stopped"
                                manifest["ended_at"] = _now()
                                _write_json_atomic(run_dir / "manifest.json", manifest)
                                log.write(f"{_now()} DEADLINE before next fit\n")
                                return run_dir
                            fit_id = (
                                f"{dataset_name}__{protocol}__s{seed}__f{split.fold}__{model_name}"
                            )
                            fit_path = run_dir / "fits" / f"{fit_id}.json"
                            prediction_path = run_dir / "predictions" / f"{fit_id}.csv.gz"
                            started = _now()
                            started_clock = time.perf_counter()
                            log.write(f"{started} FIT_START {fit_id}\n")
                            try:
                                model = build_model(
                                    model_name,
                                    seed=int(seed) + split.fold,
                                    n_jobs=int(config.get("n_jobs", 1)),
                                    pilot=config["kind"] == "pilot",
                                )
                                fit_with_thread_limit(
                                    model,
                                    bundle.X[split.train],
                                    encoded_y[split.train],
                                    limit=numeric_thread_limit,
                                )
                                fit_seconds = time.perf_counter() - started_clock
                                calibration_raw = aligned_predict_proba(
                                    model, bundle.X[split.calibration], len(label_names)
                                )
                                test_raw = aligned_predict_proba(
                                    model, bundle.X[split.test], len(label_names)
                                )
                                scaler = TemperatureScaler().fit(
                                    calibration_raw, encoded_y[split.calibration]
                                )
                                calibration_scaled = scaler.transform(calibration_raw)
                                test_scaled = scaler.transform(test_raw)
                                thresholds = {
                                    str(coverage): confidence_threshold(
                                        calibration_scaled, float(coverage)
                                    )
                                    for coverage in config["target_coverages"]
                                }
                                raw_metrics = evaluate_probabilities(
                                    encoded_y[split.test], test_raw, len(label_names)
                                )
                                calibrated_metrics = evaluate_probabilities(
                                    encoded_y[split.test], test_scaled, len(label_names)
                                )
                                selective = {
                                    coverage: selective_metrics(
                                        encoded_y[split.test], test_scaled, threshold
                                    )
                                    for coverage, threshold in thresholds.items()
                                }
                                fit_metadata = {
                                    "run_id": run_id,
                                    "dataset": dataset_name,
                                    "protocol": protocol,
                                    "seed": int(seed),
                                    "fold": split.fold,
                                    "model": model_name,
                                    "split_id": split_info["split_id"],
                                }
                                frame = _prediction_frame(
                                    bundle,
                                    split.test,
                                    encoded_y[split.test],
                                    label_names,
                                    test_raw,
                                    test_scaled,
                                    fit_metadata,
                                )
                                if prediction_path.exists():
                                    raise FileExistsError(f"refusing to overwrite {prediction_path}")
                                frame.to_csv(prediction_path, index=False, compression="gzip")
                                result = {
                                    **fit_metadata,
                                    "status": "success",
                                    "started_at": started,
                                    "ended_at": _now(),
                                    "fit_seconds": fit_seconds,
                                    "temperature": scaler.temperature_,
                                    "thresholds": thresholds,
                                    "raw_metrics": raw_metrics,
                                    "calibrated_metrics": calibrated_metrics,
                                    "selective_metrics": selective,
                                    "prediction_file": str(prediction_path.relative_to(project_root)),
                                }
                                _write_json_atomic(fit_path, result)
                                manifest["completed_fits"].append(fit_id)
                                for calibration_state, values in (
                                    ("raw", raw_metrics),
                                    ("calibrated", calibrated_metrics),
                                ):
                                    metrics_rows.append(
                                        {
                                            **fit_metadata,
                                            "calibration": calibration_state,
                                            "temperature": scaler.temperature_,
                                            "fit_seconds": fit_seconds,
                                            **values,
                                        }
                                    )
                                log.write(
                                    f"{_now()} FIT_OK {fit_id} seconds={fit_seconds:.3f} "
                                    f"temperature={scaler.temperature_:.6g}\n"
                                )
                            except Exception as exc:  # preserve individual failures and continue
                                failure = {
                                    "fit_id": fit_id,
                                    "status": "failed",
                                    "started_at": started,
                                    "ended_at": _now(),
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                }
                                _write_json_atomic(fit_path, failure)
                                manifest["failed_fits"].append(fit_id)
                                log.write(f"{_now()} FIT_FAILED {fit_id} {type(exc).__name__}: {exc}\n")
                            _write_json_atomic(run_dir / "manifest.json", manifest)

        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(run_dir / "fit_metrics.csv", index=False)
        manifest["status"] = "complete" if not manifest["failed_fits"] else "partial_failure"
        manifest["ended_at"] = _now()
        _write_json_atomic(run_dir / "manifest.json", manifest)
        log.write(f"{_now()} END status={manifest['status']}\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    arguments = parser.parse_args()
    run_dir = run_experiment(arguments.config, arguments.project_root)
    print(run_dir)


if __name__ == "__main__":
    main()
