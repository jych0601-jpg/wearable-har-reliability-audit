"""Common validated representation for subject-labelled HAR datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import arff


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sample_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        subjects = np.asarray(self.subjects)
        sample_ids = np.asarray(self.sample_ids)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric features")
        if not np.isfinite(X).all():
            raise ValueError("X features must all be finite")
        n = X.shape[0]
        if any(len(values) != n for values in (y, subjects, sample_ids)):
            raise ValueError("X, y, subjects, and sample_ids must have equal length")
        if X.shape[1] != len(self.feature_names):
            raise ValueError("feature_names length must match X columns")
        if len(np.unique(sample_ids)) != n:
            raise ValueError("sample_ids must be unique")
        if n == 0:
            raise ValueError("dataset must contain at least one sample")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "sample_ids", sample_ids)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _read_vector(path: Path, dtype: type) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype, ndmin=1)


def load_uci_har(root: str | Path) -> DatasetBundle:
    """Load and concatenate the official supplied UCI HAR feature splits."""
    root = Path(root)
    activity_map = {
        int(row.split(maxsplit=1)[0]): row.split(maxsplit=1)[1].strip()
        for row in (root / "activity_labels.txt").read_text().splitlines()
        if row.strip()
    }
    features = []
    for row in (root / "features.txt").read_text().splitlines():
        if row.strip():
            features.append(row.split(maxsplit=1)[1].strip())

    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    sample_ids: list[str] = []
    for original_split in ("train", "test"):
        X_part = np.loadtxt(root / original_split / f"X_{original_split}.txt", dtype=float)
        y_codes = _read_vector(root / original_split / f"y_{original_split}.txt", int)
        subject_codes = _read_vector(root / original_split / f"subject_{original_split}.txt", int)
        matrices.append(X_part)
        labels.append(np.array([activity_map[int(code)] for code in y_codes], dtype=str))
        subjects.append(np.array([f"uci_{int(code):02d}" for code in subject_codes], dtype=str))
        sample_ids.extend(f"uci_{original_split}_{i:05d}" for i in range(len(y_codes)))

    return DatasetBundle(
        name="uci_har",
        X=np.vstack(matrices),
        y=np.concatenate(labels),
        subjects=np.concatenate(subjects),
        sample_ids=np.asarray(sample_ids),
        feature_names=tuple(features),
        metadata={
            "uci_id": 240,
            "doi": "10.24432/C54S4K",
            "license": "CC BY 4.0",
            "representation": "official 561-feature, 2.56-second windows, 50% overlap",
        },
    )


def _decode_arff_text(values: np.ndarray) -> np.ndarray:
    """Return nominal ARFF values as ordinary Unicode strings."""
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def load_wisdm_arff(
    root: str | Path,
    *,
    device: str = "watch",
    sensor: str = "accel",
) -> DatasetBundle:
    """Load official WISDM per-participant transformed-feature ARFF files.

    The WISDM archive stores one ARFF per participant and device/sensor pair. The
    ``class`` field is retained as the participant identifier; ``ACTIVITY`` is
    the target and every remaining field must be numeric.
    """
    root = Path(root)
    files = sorted((root / "arff_files" / device / sensor).glob("*.arff"))
    if not files:
        raise FileNotFoundError(
            f"No WISDM ARFF files found for device={device!r}, sensor={sensor!r} under {root}"
        )

    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    sample_ids: list[str] = []
    feature_names: tuple[str, ...] | None = None

    for path in files:
        records, _ = arff.loadarff(path)
        names = tuple(records.dtype.names or ())
        normalized_names = {name.strip("\"'"): name for name in names}
        required = {"ACTIVITY", "class"}
        if not required.issubset(normalized_names):
            missing = sorted(required - set(normalized_names))
            raise ValueError(f"{path} is missing required ARFF fields {missing}")
        original_features = tuple(
            name for name in names if name not in {normalized_names[key] for key in required}
        )
        current_features = tuple(name.strip("\"'") for name in original_features)
        if feature_names is None:
            feature_names = current_features
        elif current_features != feature_names:
            raise ValueError(f"Inconsistent feature schema in {path}")

        try:
            X_part = np.column_stack(
                [np.asarray(records[name], dtype=float) for name in original_features]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric feature encountered in {path}") from exc
        y_part = _decode_arff_text(records[normalized_names["ACTIVITY"]])
        subject_part = _decode_arff_text(records[normalized_names["class"]])
        subject_part = np.asarray([f"wisdm_{value}" for value in subject_part], dtype=str)

        matrices.append(X_part)
        labels.append(y_part)
        subjects.append(subject_part)
        sample_ids.extend(
            f"wisdm_{device}_{sensor}_{path.stem}_{row_index:05d}"
            for row_index in range(len(records))
        )

    activity_map = {
        "A": "walking",
        "B": "jogging",
        "C": "stairs",
        "D": "sitting",
        "E": "standing",
        "F": "typing",
        "G": "brushing_teeth",
        "H": "eating_soup",
        "I": "eating_chips",
        "J": "eating_pasta",
        "K": "drinking_from_cup",
        "L": "eating_sandwich",
        "M": "kicking",
        "O": "catching",
        "P": "dribbling",
        "Q": "writing",
        "R": "clapping",
        "S": "folding_clothes",
    }
    assert feature_names is not None
    return DatasetBundle(
        name=f"wisdm_{device}_{sensor}",
        X=np.vstack(matrices),
        y=np.concatenate(labels),
        subjects=np.concatenate(subjects),
        sample_ids=np.asarray(sample_ids),
        feature_names=feature_names,
        metadata={
            "uci_id": 507,
            "doi": "10.24432/C5HK59",
            "license": "CC BY 4.0",
            "device": device,
            "sensor": sensor,
            "representation": "official transformed 10-second feature windows",
            "activity_map": activity_map,
        },
    )
