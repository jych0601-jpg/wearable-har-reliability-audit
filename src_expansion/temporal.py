"""UCI overlap-component leakage sensitivity, isolated from active base runs.

run(False) verifies the raw graph and delegates 20 classical fits. run(True)
additionally delegates five CNN fits. Components identify exact shared sensor
samples; they are neither sessions nor a claim of general temporal independence.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "leakage_reduced_mixed"
PRIMARY_SEED = 20260911
CLASSICAL_MODELS = ("logistic", "random_forest", "extra_trees", "hist_gradient_boosting")
GRAPH_PATH = ROOT / "06_robustness" / "uci_verified_overlap_graph.npz"
CONFIG_PATH = ROOT / "configs" / "temporal_blocking.json"


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _row_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _edge_array(edges, n_rows):
    values = np.asarray(edges)
    if values.ndim != 2 or values.shape[1] != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Graph edges must be an integer E,2 array")
    if len(values) and (values.min() < 0 or values.max() >= n_rows or np.any(values[:, 0] == values[:, 1])):
        raise ValueError("Graph edge rows are invalid")
    if len({tuple(pair) for pair in values}) != len(values):
        raise ValueError("Graph contains duplicate edges")
    return values.astype(np.int64, copy=False)


def audit_overlap_graph(total_acc, subjects, edges, components):
    """Recompute exact three-axis half-window matches and graph connectivity.

    Besides the directed suffix-to-prefix edges, every identical 64-sample
    prefix/prefix, prefix/suffix, or suffix/suffix pair within a participant must
    belong to one supplied component. Equality is tested on float64 source values.
    """
    X = np.asarray(total_acc)
    subjects, components = np.asarray(subjects), np.asarray(components)
    if X.ndim != 3 or X.shape[1:] != (3, 128) or not len(X) or not np.isfinite(X).all():
        raise ValueError("Exact graph audit requires finite N,3,128 total acceleration")
    n_rows = len(X)
    if subjects.shape != (n_rows,) or components.shape != (n_rows,):
        raise ValueError("Graph subjects/components are misaligned")
    edges = _edge_array(edges, n_rows)
    starts, half_blocks = {}, {}
    for index in range(n_rows):
        for half, selection in enumerate((slice(0, 64), slice(64, 128))):
            values = X[index, :, selection]
            key = (str(subjects[index]), hashlib.sha256(values.tobytes()).digest())
            if half == 0:
                starts.setdefault(key, []).append(index)
            if key in half_blocks:
                previous, previous_half = half_blocks[key]
                expected = X[previous, :, :64] if previous_half == 0 else X[previous, :, 64:]
                if not np.array_equal(values, expected):
                    raise ValueError("Hash equality did not imply exact raw sample equality")
                if components[index] != components[previous]:
                    raise ValueError("An exact shared 64-sample block crosses supplied components")
            else:
                half_blocks[key] = (index, half)
    actual_edges, ambiguous = set(), 0
    for index in range(n_rows):
        key = (str(subjects[index]), hashlib.sha256(X[index, :, 64:].tobytes()).digest())
        candidates = [other for other in starts.get(key, []) if other != index]
        ambiguous += len(candidates) > 1
        for other in candidates:
            if not np.array_equal(X[index, :, 64:], X[other, :, :64]):
                raise ValueError("False exact raw overlap edge")
            actual_edges.add((index, other))
    provided = {tuple(map(int, edge)) for edge in edges}
    if provided != actual_edges:
        raise ValueError(f"Saved graph is false or incomplete: missing={len(actual_edges - provided)}, extra={len(provided - actual_edges)}")
    parent = np.arange(n_rows)
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)
    for left, right in actual_edges:
        parent[find(right)] = find(left)
    root_to_component, component_to_root = {}, {}
    for index in range(n_rows):
        actual, supplied = find(index), components[index].item() if hasattr(components[index], "item") else components[index]
        if actual in root_to_component and root_to_component[actual] != supplied:
            raise ValueError("A connected graph component was split")
        if supplied in component_to_root and component_to_root[supplied] != actual:
            raise ValueError("Disconnected graph components were merged")
        root_to_component[actual], component_to_root[supplied] = supplied, actual
    sizes = np.unique(components, return_counts=True)[1]
    return {"n_windows": n_rows, "verified_edges": len(actual_edges),
            "ambiguous_successors": int(ambiguous), "connected_components": len(sizes),
            "component_size_min": int(sizes.min()), "component_size_median": float(np.median(sizes)),
            "component_size_max": int(sizes.max()),
            "all_matching_half_blocks_within_components": True,
            "exact_source_dtype": str(X.dtype)}


def assert_component_separation(roles, components, edges=None):
    """Require complete row assignment, disjoint components, and no crossing edge."""
    components = np.asarray(components)
    n_rows = len(components)
    if components.ndim != 1 or not n_rows or set(roles) not in ({"train", "calibration", "test"}, {"train", "validation", "calibration", "test"}):
        raise ValueError("Invalid component vector or partition roles")
    assignment = np.full(n_rows, -1, dtype=int)
    group_owner = {}
    counts, group_counts = {}, {}
    for role_index, (role, supplied) in enumerate(roles.items()):
        indices = np.asarray(supplied)
        if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{role} requires nonempty integer row indices")
        if indices.min() < 0 or indices.max() >= n_rows or len(np.unique(indices)) != len(indices):
            raise ValueError(f"{role} contains duplicate/out-of-range rows")
        if np.any(assignment[indices] >= 0):
            raise ValueError("Row overlap between temporal partitions")
        assignment[indices] = role_index
        groups = np.unique(components[indices])
        for group in groups:
            if group in group_owner:
                raise ValueError("An overlap component crosses temporal partitions")
            group_owner[group] = role
        counts[role], group_counts[role] = len(indices), len(groups)
    if np.any(assignment < 0):
        raise ValueError("Temporal partitions do not cover every row")
    if edges is not None:
        edges = _edge_array(edges, n_rows)
        if np.any(assignment[edges[:, 0]] != assignment[edges[:, 1]]):
            raise ValueError("A verified overlap edge crosses temporal partitions")
    return {"cross_partition_edges": 0, "cross_partition_components": 0,
            "rows": counts, "components": group_counts}


def make_component_splits(y, components, seed=PRIMARY_SEED, n_folds=5):
    """Rotate test=k, calibration=k+1, train=the remaining component-held folds."""
    y, groups = np.asarray(y), np.asarray(components)
    if y.ndim != 1 or groups.shape != y.shape or len(np.unique(groups)) < n_folds or n_folds < 3:
        raise ValueError("Insufficient or misaligned labels and overlap groups")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    tests = [np.sort(test) for _, test in splitter.split(np.zeros(len(y)), y, groups)]
    all_rows, result = np.arange(len(y)), []
    domain = np.unique(y)
    for fold in range(n_folds):
        test, calibration = tests[fold], tests[(fold + 1) % n_folds]
        train = np.setdiff1d(all_rows, np.union1d(test, calibration), assume_unique=True)
        roles = {"train": train, "calibration": calibration.copy(), "test": test.copy()}
        assert_component_separation(roles, groups)
        if not np.array_equal(np.unique(y[train]), domain):
            raise ValueError("An outer training partition lacks a globally declared class")
        result.append(roles)
    return result


def component_internal_validation(roles, y, components, seed=PRIMARY_SEED):
    """Hold out the first of five stratified component folds of outer training."""
    y, groups = np.asarray(y), np.asarray(components)
    assert_component_separation(roles, groups)
    outer_train = np.asarray(roles["train"])
    if len(np.unique(groups[outer_train])) < 5:
        raise ValueError("Insufficient outer training components for validation")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    fit_local, validation_local = next(splitter.split(np.zeros(len(outer_train)), y[outer_train], groups[outer_train]))
    result = {name: np.asarray(indices).copy() for name, indices in roles.items()}
    result.update(train=np.sort(outer_train[fit_local]), validation=np.sort(outer_train[validation_local]))
    assert_component_separation(result, groups)
    if not np.array_equal(np.unique(y[result["train"]]), np.unique(y)):
        raise ValueError("Internal training partition lacks a globally declared class")
    return result


@contextmanager
def installed_protocol(orchestration, state):
    """Install this named protocol only in the calling process, restoring on exit."""
    old_folds, old_validation = orchestration.folds, orchestration.internal_validation
    cache = {}
    def folds(dataset, model, protocol, seed):
        if protocol != PROTOCOL:
            return old_folds(dataset, model, protocol, seed)
        if dataset != "uci_har" or seed != PRIMARY_SEED:
            raise ValueError("Temporal sensitivity is frozen to UCI and the primary seed")
        if seed not in cache:
            cache[seed] = make_component_splits(state["y"], state["components"], seed)
        from types import SimpleNamespace
        result = []
        for index, supplied in enumerate(cache[seed]):
            roles = {name: rows.copy() for name, rows in supplied.items()}
            assert_component_separation(roles, state["components"], state["edges"])
            result.append((SimpleNamespace(fold=index, **roles), roles))
        return result
    def validation(roles, subjects, y, protocol, seed):
        if protocol != PROTOCOL:
            return old_validation(roles, subjects, y, protocol, seed)
        if not np.array_equal(subjects, state["subjects"]) or not np.array_equal(y, state["y"]):
            raise ValueError("CNN rows do not align with the verified UCI graph")
        result = component_internal_validation(roles, y, state["components"], seed)
        assert_component_separation(result, state["components"], state["edges"])
        return result
    orchestration.folds, orchestration.internal_validation = folds, validation
    try:
        yield
    finally:
        orchestration.folds, orchestration.internal_validation = old_folds, old_validation


def _frozen_configuration():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    fixed = {"schema_version": 1, "protocol": PROTOCOL, "dataset": "uci_har",
             "seed": PRIMARY_SEED, "n_folds": 5, "splitter": "StratifiedGroupKFold",
             "shuffle": True, "calibration_fold_offset": 1,
             "inner_validation_n_folds": 5, "inner_validation_fold": 0,
             "overlap_samples": 64, "graph_n_windows": 10299,
             "graph_n_edges": 9442, "graph_n_components": 857,
             "subject_disjoint": False, "sessions_inferred": False}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen temporal protocol differs from the implementation: {key}")
    if config.get("graph_sha256") != _sha256(GRAPH_PATH):
        raise ValueError("Frozen verified overlap graph changed")
    if config.get("base_config_sha256") != _sha256(ROOT / "configs" / "expanded.json"):
        raise ValueError("Frozen base model configuration changed")
    return config


def prepare():
    """Verify official row alignment and every overlap before any temporal fit."""
    import experiments
    config = _frozen_configuration()
    features, raw = experiments.features("uci_har"), experiments.raw_data("uci_har")
    for name in ("sample_ids", "y", "subjects"):
        if not np.array_equal(raw[name], getattr(features, name)):
            raise ValueError(f"UCI raw/V1 {name} alignment failed")
    with np.load(GRAPH_PATH, allow_pickle=False) as saved:
        edges, components, numeric_subjects = (saved[key].copy() for key in ("edges", "components", "subjects"))
    expected_subjects = np.asarray([f"uci_{int(subject):02d}" for subject in numeric_subjects])
    if not np.array_equal(expected_subjects, features.subjects):
        raise ValueError("Graph participant order differs from V1")
    root = ROOT / "work" / "data" / "uci_har_240" / "UCI HAR Dataset"
    arrays = []
    for split in ("train", "test"):
        arrays.append(np.stack([np.loadtxt(root / split / "Inertial Signals" / f"total_acc_{axis}_{split}.txt", dtype=np.float64, ndmin=2) for axis in "xyz"], axis=1))
    total_acc = np.concatenate(arrays)
    graph_audit = audit_overlap_graph(total_acc, numeric_subjects, edges, components)
    if (graph_audit["n_windows"], graph_audit["verified_edges"], graph_audit["connected_components"], graph_audit["ambiguous_successors"]) != (10299, 9442, 857, 0):
        raise ValueError("Graph audit differs from frozen pre-fit evidence")
    y = LabelEncoder().fit_transform(features.y)
    state = {"y": y, "subjects": features.subjects, "components": components, "edges": edges,
             "config": config, "config_sha256": _sha256(CONFIG_PATH), "graph_audit": graph_audit}
    records = []
    for fold, outer in enumerate(make_component_splits(y, components)):
        inner = component_internal_validation(outer, y, components, PRIMARY_SEED + fold)
        record = {"fold": fold, "outer": assert_component_separation(outer, components, edges),
                  "cnn_inner": assert_component_separation(inner, components, edges),
                  "outer_row_hashes": {role: _row_hash(rows) for role, rows in outer.items()},
                  "cnn_row_hashes": {role: _row_hash(rows) for role, rows in inner.items()},
                  "outer_subject_counts": {role: len(np.unique(features.subjects[rows])) for role, rows in outer.items()}}
        records.append(record)
    from common import atomic_json, now
    evidence = {"verified_utc": now(), "protocol": PROTOCOL, "config_sha256": state["config_sha256"],
                "graph_sha256": config["graph_sha256"], "raw_v1_alignment": True,
                "graph_audit": graph_audit, "folds": records, "passed": True}
    atomic_json(ROOT / "logs" / "temporal_partition_audit.json", evidence)
    return state


def run(include_deep=False):
    """Run/resume 20 classical fits; include_deep=True also runs/resumes five CNNs."""
    if not isinstance(include_deep, bool):
        raise ValueError("include_deep must be boolean")
    import experiments
    from common import atomic_json, now, log
    state = prepare()
    models = list(CLASSICAL_MODELS) + (["cnn1d"] if include_deep else [])
    specs = [("uci_har", model, PROTOCOL, PRIMARY_SEED, fold) for fold in range(5) for model in models]
    completed, failures, resumed = [], [], 0
    started = now()
    with installed_protocol(experiments, state):
        for spec in specs:
            dataset, model, protocol, seed, fold = spec
            try:
                _, expected_roles = experiments.folds(dataset, model, protocol, seed)[fold]
                if model == "cnn1d":
                    expected_roles = experiments.internal_validation(expected_roles, state["subjects"], state["y"], protocol, seed + fold)
                partition_audit = assert_component_separation(expected_roles, state["components"], state["edges"])
                destination = experiments.fit_directory(*spec)
                provenance_path = destination / "temporal_provenance.json"
                expected = {"schema_version": 1, "protocol": PROTOCOL, "dataset": dataset, "model": model,
                            "seed": seed, "fold": fold, "temporal_config": str(CONFIG_PATH.relative_to(ROOT)),
                            "temporal_config_sha256": state["config_sha256"],
                            "graph_sha256": state["config"]["graph_sha256"],
                            "source_sha256": _sha256(Path(__file__)),
                            "role_row_hashes": {role: _row_hash(rows) for role, rows in expected_roles.items()},
                            "partition_audit": partition_audit,
                            "shared_64_sample_blocks_cross_partitions": 0,
                            "evidence_scope": "Exact shared half-window samples within participant; components are not sessions"}
                if provenance_path.exists():
                    previous = json.loads(provenance_path.read_text(encoding="utf-8"))
                    if any(previous.get(key) != value for key, value in expected.items()):
                        raise RuntimeError("Temporal resume provenance changed: " + str(destination))
                was_complete = experiments.valid_completed(destination)
                experiments.fit_one(spec)
                marker_path = destination / "completed.json"
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                attempt = ROOT / marker["attempt"]
                with np.load(attempt / "roles.npz", allow_pickle=False) as stored:
                    actual_roles = {name: stored[name].copy() for name in stored.files}
                if set(actual_roles) != set(expected_roles) or any(not np.array_equal(actual_roles[name], expected_roles[name]) for name in expected_roles):
                    raise RuntimeError("Actual fitted rows differ from verified temporal partitions")
                assert_component_separation(actual_roles, state["components"], state["edges"])
                expected.update(completed_marker_sha256=_sha256(marker_path), attempt=marker["attempt"])
                if provenance_path.exists():
                    if json.loads(provenance_path.read_text(encoding="utf-8")) != expected:
                        raise RuntimeError("Temporal completion linkage changed")
                else:
                    atomic_json(provenance_path, expected)
                resumed += int(was_complete)
                completed.append({"spec": list(spec), "directory": str(destination.relative_to(ROOT)),
                                  "attempt": marker["attempt"], "temporal_provenance_sha256": _sha256(provenance_path)})
            except Exception as error:
                import traceback
                failures.append({"spec": list(spec), "error": str(error), "traceback": traceback.format_exc()})
                log("TEMPORAL_FIT_FAILED", spec=spec, error=str(error))
    result = {"protocol": PROTOCOL, "dataset": "uci_har", "seed": PRIMARY_SEED,
              "include_deep": include_deep, "models": models, "expected_fits": len(specs),
              "completed_fits": len(completed), "resumed_fits": resumed, "new_fits": len(completed) - resumed,
              "started_utc": started, "ended_utc": now(), "config_sha256": state["config_sha256"],
              "graph_audit": state["graph_audit"], "fits": completed, "failures": failures,
              "passed": not failures and len(completed) == len(specs)}
    output = ROOT / "logs" / ("temporal_run_all.json" if include_deep else "temporal_run_classical.json")
    atomic_json(output, result)
    if failures:
        raise RuntimeError(f"{len(failures)} temporal fits failed; detailed attempts and errors retained in {output}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-deep", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.audit_only:
        checked = prepare()
        print(json.dumps({"passed": True, "config_sha256": checked["config_sha256"], "graph_audit": checked["graph_audit"]}), flush=True)
    else:
        print(json.dumps(run(include_deep=arguments.include_deep), ensure_ascii=False), flush=True)