"""Deterministic rotating train/calibration/test splits with leakage checks."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class SplitIndices:
    fold: int
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def make_rotating_splits(
    y: np.ndarray,
    subjects: np.ndarray,
    n_folds: int,
    seed: int,
    protocol: str,
) -> list[SplitIndices]:
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    if len(y) != len(subjects):
        raise ValueError("y and subjects must have equal length")
    if protocol == "subject_disjoint":
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_tests = [test for _, test in splitter.split(np.zeros(len(y)), y, groups=subjects)]
    elif protocol == "sample_mixed":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_tests = [test for _, test in splitter.split(np.zeros(len(y)), y)]
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    all_rows = np.arange(len(y))
    splits: list[SplitIndices] = []
    for fold in range(n_folds):
        test = np.sort(np.asarray(fold_tests[fold], dtype=int))
        calibration = np.sort(np.asarray(fold_tests[(fold + 1) % n_folds], dtype=int))
        held_out = np.union1d(test, calibration)
        train = np.setdiff1d(all_rows, held_out, assume_unique=True)
        split = SplitIndices(fold=fold, train=train, calibration=calibration, test=test)
        validate_split(split, subjects, protocol)
        splits.append(split)
    return splits


def validate_split(split: SplitIndices, subjects: np.ndarray, protocol: str) -> None:
    subjects = np.asarray(subjects)
    partitions = {
        "train": np.asarray(split.train, dtype=int),
        "calibration": np.asarray(split.calibration, dtype=int),
        "test": np.asarray(split.test, dtype=int),
    }
    for name, indices in partitions.items():
        if len(indices) == 0:
            raise ValueError(f"{name} partition must not be empty")
        if indices.min() < 0 or indices.max() >= len(subjects):
            raise ValueError(f"{name} contains an out-of-range row index")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} contains duplicate row indices")
    names = tuple(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if np.intersect1d(partitions[left], partitions[right]).size:
                raise ValueError(f"row overlap between {left} and {right}")
    if protocol == "subject_disjoint":
        user_sets = {name: set(subjects[idx]) for name, idx in partitions.items()}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                if not user_sets[left].isdisjoint(user_sets[right]):
                    raise ValueError(f"subject overlap between {left} and {right}")
    elif protocol != "sample_mixed":
        raise ValueError(f"unknown protocol: {protocol}")
