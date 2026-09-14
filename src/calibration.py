"""Post-hoc calibration fitted solely on a dedicated calibration partition."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be n_samples by n_classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-7):
        raise ValueError("probability rows must sum to one")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


class TemperatureScaler:
    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.bounds = bounds
        self.temperature_: float | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        p = _validate_probabilities(probabilities)
        labels = np.asarray(labels, dtype=int)
        if len(labels) != len(p) or labels.min() < 0 or labels.max() >= p.shape[1]:
            raise ValueError("labels are incompatible with probability columns")
        log_p = np.log(p)

        def objective(log_temperature: float) -> float:
            scaled = log_p / np.exp(log_temperature)
            log_normalizer = logsumexp(scaled, axis=1)
            return float(np.mean(log_normalizer - scaled[np.arange(len(labels)), labels]))

        result = minimize_scalar(
            objective,
            bounds=(np.log(self.bounds[0]), np.log(self.bounds[1])),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"temperature optimization failed: {result.message}")
        self.temperature_ = float(np.exp(result.x))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        p = _validate_probabilities(probabilities)
        scaled = np.log(p) / self.temperature_
        scaled -= logsumexp(scaled, axis=1, keepdims=True)
        return np.exp(scaled)

    def fit_transform(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(probabilities, labels).transform(probabilities)


def confidence_threshold(probabilities: np.ndarray, target_coverage: float) -> float:
    p = _validate_probabilities(probabilities)
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    confidence = p.max(axis=1)
    return float(np.quantile(confidence, 1.0 - target_coverage, method="higher"))
