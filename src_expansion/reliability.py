"""Independent calibration, conformal prediction, and split-audit kernels.

APS follows the adaptive cumulative-probability score family of Romano, Sesia,
and Candes (2020), "Classification with Valid and Adaptive Coverage":
https://arxiv.org/abs/2006.02544 . This module uses a deterministic,
nonrandomized cumulative-INCLUSIVE score, with ascending class indices
breaking equal-probability ties. It is not the paper's randomized score.

Conformal coverage is marginal under exchangeability of calibration and future
scores conditional on the already fitted predictor/calibrator. These kernels
cannot establish exchangeability of dependent windows or shifted subjects.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
from sklearn.isotonic import IsotonicRegression


PROBABILITY_ATOL = 1e-6


def validate_probabilities(p) -> np.ndarray:
    """Return floating probabilities after validation, without repairing rows.

    Inputs must be nonempty N-by-K real arrays with finite nonnegative entries
    and row sums within absolute 1e-6 of one (relative tolerance is zero).
    The tolerance accommodates float32 softmax roundoff. No input is mutated.
    """
    values = np.asarray(p)
    if values.dtype.kind not in "biuf":
        raise ValueError("Probabilities must be real numeric values")
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Probabilities must have nonempty shape (n_rows, n_classes)")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities must be finite")
    if np.any(values < 0):
        raise ValueError("Probabilities must be nonnegative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=PROBABILITY_ATOL, rtol=0.0):
        raise ValueError("Probability row sums must equal one within absolute 1e-6")
    return values


def _validate_labels(y, n_rows: int, n_classes: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n_rows:
        raise ValueError("Labels must be a one-dimensional array matching probability rows")
    if labels.dtype.kind not in "iuf" or not np.all(np.isfinite(labels)):
        raise ValueError("Labels must be finite integer class indices")
    if np.any(labels != np.floor(labels)) or np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("Labels must be integer class indices in [0, n_classes)")
    return labels.astype(np.int64, copy=False)


class OVRIsotonic:
    """Fit one isotonic map per class on a designated probability-calibration set.

    Classes absent from calibration get a constant zero map. Independent map
    outputs are renormalized across classes. If every map returns zero for a
    row, the raw probability row supplies the fallback before normalization.
    Transform consumes probabilities only and never refits on prediction rows.
    """

    def fit(self, p, y) -> OVRIsotonic:
        probabilities = validate_probabilities(p)
        labels = _validate_labels(y, *probabilities.shape)
        models = []
        for class_index in range(probabilities.shape[1]):
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(probabilities[:, class_index], (labels == class_index).astype(float))
            models.append(model)
        self.models_ = models
        self.n_classes_ = probabilities.shape[1]
        return self

    def transform(self, p) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("OVRIsotonic must be fitted before transform")
        probabilities = validate_probabilities(p)
        if probabilities.shape[1] != self.n_classes_:
            raise ValueError("Prediction class count differs from calibration class count")
        calibrated = np.column_stack([
            model.predict(probabilities[:, class_index])
            for class_index, model in enumerate(self.models_)
        ])
        zero_rows = calibrated.sum(axis=1) == 0.0
        calibrated[zero_rows] = probabilities[zero_rows]
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return validate_probabilities(calibrated)


def _aps_scores(probabilities: np.ndarray) -> np.ndarray:
    """Return the cumulative-inclusive score for every candidate class."""
    # Validation has already enforced unit row mass. Remove only the admitted
    # floating-point roundoff so full-mass scores are exactly one for all rows.
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-normalized, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(normalized, order, axis=1)
    cumulative = np.minimum(np.cumsum(sorted_probabilities, axis=1), 1.0)
    cumulative[:, -1] = 1.0
    scores = np.empty_like(cumulative)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores


class APS:
    """Split conformal sets from nonrandomized cumulative-inclusive APS scores.

    For a candidate class k, score(x, k) sums all probabilities through k in
    descending stable order. Fitting uses only the true-label scores on the
    separate conformal calibration set. With n scores, the threshold is the
    ceil((n+1)*(1-alpha))-th smallest score, or +infinity when that rank exceeds
    n. Prediction returns {k: score(x,k) <= threshold}; empty sets are permitted.
    Callers should report empty-set frequency alongside coverage and set size.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, p, y) -> APS:
        if isinstance(self.alpha, (bool, np.bool_)) or not isinstance(self.alpha, Real):
            raise ValueError("alpha must be a finite real number strictly between zero and one")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be a finite real number strictly between zero and one")
        probabilities = validate_probabilities(p)
        labels = _validate_labels(y, *probabilities.shape)
        n_rows, n_classes = probabilities.shape
        scores = _aps_scores(probabilities)[np.arange(n_rows), labels]
        rank = math.ceil((n_rows + 1) * (1.0 - float(self.alpha)))
        threshold = math.inf if rank > n_rows else float(np.partition(scores, rank - 1)[rank - 1])
        self.threshold_ = threshold
        self.n_calibration_ = n_rows
        self.n_classes_ = n_classes
        return self

    def predict(self, p) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("APS must be fitted before predict")
        probabilities = validate_probabilities(p)
        if probabilities.shape[1] != self.n_classes_:
            raise ValueError("Prediction class count differs from conformal calibration class count")
        return _aps_scores(probabilities) <= self.threshold_


def assert_roles(subjects, roles: dict[str, np.ndarray], subject_disjoint: bool = True) -> None:
    """Reject row leakage and optionally participant leakage across all roles.

    ``subjects`` has one participant identifier per row in the source dataset;
    every value in ``roles`` contains global integer row indices. Include the
    internal validation split as its own role. Duplicate rows inside one role
    are invalid. Disjointness is checked pairwise for every supplied role;
    passing ``subject_disjoint=False`` still enforces all row-level checks.
    No exhaustiveness requirement is imposed on the supplied subset of rows.
    """
    participant_ids = np.asarray(subjects)
    if participant_ids.ndim != 1 or len(participant_ids) == 0:
        raise ValueError("subjects must be a nonempty one-dimensional array")
    for participant in participant_ids.tolist():
        if participant is None or (isinstance(participant, Real) and not math.isfinite(participant)):
            raise ValueError("Participant identifiers must not be missing or nonfinite")
        try:
            hash(participant)
        except TypeError as exc:
            raise ValueError("Participant identifiers must be hashable scalars") from exc
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError("roles must be a nonempty mapping from role names to row indices")
    row_owners: dict[int, str] = {}
    participant_owners: dict[object, str] = {}
    for role, raw_indices in roles.items():
        indices = np.asarray(raw_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"Role {role!r} must contain one-dimensional integer row indices")
        if np.any(indices < 0) or np.any(indices >= len(participant_ids)):
            raise ValueError(f"Role {role!r} has out-of-range row indices")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Role {role!r} contains duplicate rows")
        for row in indices.tolist():
            if row in row_owners:
                raise ValueError(f"Row overlap between roles {row_owners[row]!r} and {role!r}")
            row_owners[row] = role
        if subject_disjoint:
            for participant in set(participant_ids[indices].tolist()):
                if participant in participant_owners:
                    raise ValueError(
                        f"Participant overlap between roles {participant_owners[participant]!r} and {role!r}"
                    )
                participant_owners[participant] = role
