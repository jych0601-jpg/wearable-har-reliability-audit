"""Probability-quality and selective-prediction metrics."""

import numpy as np


def multiclass_brier_score(labels: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), n_classes):
        raise ValueError("probability shape does not match labels and n_classes")
    targets = np.eye(n_classes, dtype=float)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or n_bins < 1:
        raise ValueError("invalid labels, probabilities, or n_bins")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    order = np.argsort(confidence, kind="stable")
    bins = [indices for indices in np.array_split(order, min(n_bins, len(order))) if len(indices)]
    return float(
        sum(len(indices) * abs(confidence[indices].mean() - correct[indices].mean()) for indices in bins)
        / len(labels)
    )


def selective_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    retained = confidence >= threshold
    n_retained = int(retained.sum())
    result: dict[str, float | int] = {
        "n_total": int(len(labels)),
        "n_retained": n_retained,
        "coverage": float(retained.mean()),
        "threshold": float(threshold),
    }
    if n_retained:
        predictions = probabilities[retained].argmax(axis=1)
        result["selective_error"] = float(np.mean(predictions != labels[retained]))
    else:
        result["selective_error"] = float("nan")
    return result


def area_under_risk_coverage(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Discrete AURC from most to least confident, including every retained prefix."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    errors = (probabilities.argmax(axis=1) != labels).astype(float)
    order = np.argsort(-confidence, kind="stable")
    risks = np.cumsum(errors[order]) / np.arange(1, len(labels) + 1)
    return float(risks.mean())
