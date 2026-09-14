"""Raw temporal HAR representations and one frozen 1D-CNN training recipe.

UCI uses the distributed inertial windows (already preprocessed by the authors),
with exact V1 row identities. WISDM uses newly segmented watch accelerometer
observations, so it is a separate raw-representation sensitivity, not a matched
ARFF comparison. No raw timestamps or session identities are manufactured.

Public APIs: load_uci_raw, load_wisdm_raw, fit_cnn, predict_checkpoint.
PyTorch is imported lazily so raw provenance inspection works without it.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

UCI_CHANNELS = tuple(f"{kind}_{axis}" for kind in ("body_acc", "body_gyro", "total_acc") for axis in "xyz")
WISDM_ACTIVITIES = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "sitting",
    "E": "standing", "F": "typing", "G": "brushing_teeth", "H": "eating_soup",
    "I": "eating_chips", "J": "eating_pasta", "K": "drinking_from_cup",
    "L": "eating_sandwich", "M": "kicking", "O": "catching", "P": "dribbling",
    "Q": "writing", "R": "clapping", "S": "folding_clothes",
}
FIXED_CONFIG = {
    "max_epochs": 40, "patience": 6, "batch_size": 128,
    "learning_rate": 0.001, "weight_decay": 0.0001,
}
ARCHITECTURE = {
    "name": "fixed_1d_cnn", "conv_channels": [64, 128], "kernel_size": 5,
    "padding": 2, "max_pool_size": 2, "adaptive_average_pool_output": 1,
    "dense_units": 128, "dropout": 0.3, "activation": "ReLU",
}


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _bundle(X, y, subjects, sample_ids, metadata):
    X = np.asarray(X, dtype=np.float32)
    y, subjects, sample_ids = (np.asarray(values, dtype=str) for values in (y, subjects, sample_ids))
    if X.ndim != 3 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Raw X must be a nonempty finite N,C,T array")
    if any(len(values) != len(X) for values in (y, subjects, sample_ids)):
        raise ValueError("Raw features, labels, subjects and row identities are misaligned")
    if len(np.unique(sample_ids)) != len(X):
        raise ValueError("Raw sample identities are not unique")
    return {"X": np.ascontiguousarray(X), "y": y, "subjects": subjects,
            "sample_ids": sample_ids, "metadata": metadata}


def load_uci_raw(path):
    """Load the official nine-channel, 128-step UCI windows in V1 row order."""
    root = Path(path)
    label_path = root / "activity_labels.txt"
    activity_map = {}
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            code, label = line.split(maxsplit=1)
            activity_map[int(code)] = label.strip()
    if not activity_map:
        raise ValueError("UCI activity map is empty")
    arrays, labels, people, ids, files = [], [], [], [], [label_path]
    original_counts = {}
    for split in ("train", "test"):
        label_file = root / split / f"y_{split}.txt"
        subject_file = root / split / f"subject_{split}.txt"
        y_codes = np.loadtxt(label_file, dtype=np.int64, ndmin=1)
        subject_codes = np.loadtxt(subject_file, dtype=np.int64, ndmin=1)
        if y_codes.ndim != 1 or subject_codes.shape != y_codes.shape or not len(y_codes):
            raise ValueError(f"UCI {split} labels and subjects are misaligned")
        if np.any(subject_codes <= 0) or any(int(code) not in activity_map for code in y_codes):
            raise ValueError(f"UCI {split} has unknown activity or invalid subject labels")
        channels = []
        for channel in UCI_CHANNELS:
            filename = root / split / "Inertial Signals" / f"{channel}_{split}.txt"
            values = np.loadtxt(filename, dtype=np.float32, ndmin=2)
            if values.shape != (len(y_codes), 128) or not np.isfinite(values).all():
                raise ValueError(f"Invalid or misaligned UCI inertial signal: {filename}")
            channels.append(values)
            files.append(filename)
        arrays.append(np.stack(channels, axis=1))
        labels.extend(activity_map[int(code)] for code in y_codes)
        people.extend(f"uci_{int(code):02d}" for code in subject_codes)
        ids.extend(f"uci_{split}_{index:05d}" for index in range(len(y_codes)))
        files.extend([label_file, subject_file])
        original_counts[split] = int(len(y_codes))
    metadata = {
        "uci_id": 240, "doi": "10.24432/C54S4K", "license": "CC BY 4.0",
        "representation": "official preprocessed inertial sensor windows, nine channels, 128 time steps",
        "analysis_scope": "raw temporal representation with exact V1 row alignment",
        "v1_row_alignment": True, "channels": list(UCI_CHANNELS),
        "window_samples": 128, "nominal_sampling_hz": 50, "official_window_overlap": 0.5,
        "source_split_order": ["train", "test"], "source_split_counts": original_counts,
        "session_ids_available": False, "timestamps_available": False,
        "temporal_limitation": "Supplied windows overlap; row order does not establish session identity or absolute timestamps.",
        "counts": {"retained_windows": len(ids), "excluded_windows": 0},
        "sources": [{"path": str(file.relative_to(root)), "sha256": _sha256(file)} for file in files],
    }
    return _bundle(np.concatenate(arrays), labels, people, ids, metadata)


def load_wisdm_raw(path, allowed_subjects):
    """Segment actual watch/accel records into nonoverlapping 200-observation windows.

    A window cannot cross a file, subject, activity, missing/invalid source row,
    non-increasing timestamp, or gap greater than five times the per-file median
    positive step between adjacent same-subject/activity valid observations.
    Timestamps are parsed as integers, never floats. Tail observations are counted
    and excluded; there is no padding, interpolation, resampling, or session claim.
    """
    root = Path(path)
    allowed = set(map(str, allowed_subjects))
    if not allowed or any(not value.startswith("wisdm_") or not value[6:].isdigit() for value in allowed):
        raise ValueError("allowed_subjects must contain explicit V1 wisdm_<person> identifiers")
    files = sorted((root / "raw" / "watch" / "accel").glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No raw/watch/accel WISDM files under {root}")
    counts = Counter({key: 0 for key in (
        "raw_lines_total", "blank_lines", "valid_observations", "malformed_observations",
        "excluded_subject_observations", "eligible_observations", "retained_observations",
        "discarded_incomplete_segment_observations", "retained_windows", "eligible_segments")})
    boundaries = Counter({key: 0 for key in ("activity_change", "subject_change", "nonmonotonic_timestamp", "large_gap", "invalid_line")})
    arrays, labels, people, ids, provenance, source_info = [], [], [], [], [], []
    observed_subjects = set()
    for filename in files:
        relative = str(filename.relative_to(root))
        rows = []
        before = counts.copy()
        with filename.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                counts["raw_lines_total"] += 1
                stripped = line.strip()
                if not stripped:
                    counts["blank_lines"] += 1
                    continue
                try:
                    fields = stripped.rstrip(";").split(",")
                    if len(fields) != 6:
                        raise ValueError("Expected six raw fields")
                    subject_code = int(fields[0].strip())
                    subject = f"wisdm_{subject_code}"
                    activity = fields[1].strip()
                    timestamp = int(fields[2].strip())
                    xyz = tuple(float(value.strip()) for value in fields[3:])
                    if subject_code <= 0 or activity not in WISDM_ACTIVITIES or not all(np.isfinite(value) and abs(value) <= np.finfo(np.float32).max for value in xyz):
                        raise ValueError("Invalid subject, activity or sensor value")
                except (ValueError, OverflowError):
                    counts["malformed_observations"] += 1
                    continue
                counts["valid_observations"] += 1
                observed_subjects.add(subject)
                if subject not in allowed:
                    counts["excluded_subject_observations"] += 1
                else:
                    counts["eligible_observations"] += 1
                rows.append((subject, activity, timestamp, xyz, line_number))
        positive_steps = [b[2] - a[2] for a, b in zip(rows, rows[1:])
                          if b[4] == a[4] + 1 and a[:2] == b[:2] and b[2] > a[2]]
        median_step = float(np.median(positive_steps)) if positive_steps else None
        gap_threshold = 5 * median_step if median_step is not None else None

        def retain_segment(segment):
            if not segment or segment[0][0] not in allowed:
                return
            counts["eligible_segments"] += 1
            full = len(segment) // 200
            counts["discarded_incomplete_segment_observations"] += len(segment) - 200 * full
            for block in range(full):
                window = segment[block * 200:(block + 1) * 200]
                first, last = window[0], window[-1]
                identity = f"wisdm_raw_watch_accel_{filename.stem}_lines_{first[4]:07d}_{last[4]:07d}"
                arrays.append(np.asarray([row[3] for row in window], dtype=np.float32).T)
                labels.append(first[1])
                people.append(first[0])
                ids.append(identity)
                provenance.append({"sample_id": identity, "source_file": relative,
                                   "first_line": first[4], "last_line": last[4],
                                   "first_timestamp": str(first[2]), "last_timestamp": str(last[2]),
                                   "subject": first[0], "activity": first[1], "observations": 200})
                counts["retained_windows"] += 1
                counts["retained_observations"] += 200

        segment = []
        for row in rows:
            if segment:
                previous = segment[-1]
                reasons = []
                if row[0] != previous[0]:
                    reasons.append("subject_change")
                if row[1] != previous[1]:
                    reasons.append("activity_change")
                if row[4] != previous[4] + 1:
                    reasons.append("invalid_line")
                step = row[2] - previous[2]
                if step <= 0:
                    reasons.append("nonmonotonic_timestamp")
                elif gap_threshold is not None and step > gap_threshold:
                    reasons.append("large_gap")
                if reasons:
                    if previous[0] in allowed or row[0] in allowed:
                        boundaries.update(reasons)
                    retain_segment(segment)
                    segment = []
            segment.append(row)
        retain_segment(segment)
        source_info.append({"path": relative, "sha256": _sha256(filename),
                            "median_positive_timestamp_step": median_step,
                            "large_gap_threshold": gap_threshold,
                            "counts": {key: counts[key] - before[key] for key in counts}})
    missing = sorted(allowed - set(people))
    if missing:
        raise ValueError(f"Allowed subjects missing or with no retained full raw window: {missing}")
    if counts["raw_lines_total"] != counts["blank_lines"] + counts["malformed_observations"] + counts["valid_observations"]:
        raise RuntimeError("Raw line accounting did not reconcile")
    if counts["eligible_observations"] != counts["retained_observations"] + counts["discarded_incomplete_segment_observations"]:
        raise RuntimeError("Raw segmentation accounting did not reconcile")
    metadata = {
        "uci_id": 507, "doi": "10.24432/C5HK59", "license": "CC BY 4.0",
        "device": "watch", "sensor": "accel", "channels": ["accel_x", "accel_y", "accel_z"],
        "representation": "200 contiguous actual raw watch accelerometer observations; nonoverlapping windows",
        "analysis_scope": "separate raw-representation sensitivity", "v1_row_alignment": False,
        "arff_alignment_limitation": "Raw source lines cannot be asserted to match the distributed ARFF examples.",
        "nominal_sampling_hz": 20, "window_samples": 200, "stride_samples": 200,
        "timestamps_available": True, "timestamp_storage": "exact source integers serialized as strings",
        "session_ids_available": False, "activity_map": WISDM_ACTIVITIES.copy(),
        "segmentation_rule": "break at file, subject, activity, skipped source line, nonmonotonic timestamp, or >5 per-file median positive timestamp steps",
        "gap_median_rule": "positive steps only between source-adjacent valid observations with the same subject and activity",
        "boundary_count_scope": "transitions involving an allowed subject; reasons may overlap",
        "counts": dict(counts), "boundary_counts": dict(boundaries),
        "allowed_subjects": sorted(allowed), "observed_subjects": sorted(observed_subjects),
        "retained_subjects": sorted(set(people)), "window_provenance": provenance, "sources": source_info,
    }
    return _bundle(np.stack(arrays), labels, people, ids, metadata)


def _validated_roles(roles, n_rows):
    names = ("train", "validation", "calibration", "test")
    if set(roles) != set(names):
        raise ValueError("roles must have exactly train, validation, calibration and test")
    result = {}
    used = set()
    for name in names:
        values = np.asarray(roles[name])
        if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain nonempty integer row indices")
        if np.any(values < 0) or np.any(values >= n_rows) or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} has duplicate or out-of-range rows")
        if used.intersection(map(int, values)):
            raise ValueError(f"{name} overlaps another role")
        used.update(map(int, values))
        result[name] = values.astype(np.int64, copy=True)
    return result


def _configuration(config):
    config = dict(config or {})
    canonical = {"channels": [64, 128], "kernel_size": 5, "hidden": 128,
                 "dropout": 0.3, "validation_fraction": 0.2}
    for key, fixed_value in canonical.items():
        if key in config and config.pop(key) != fixed_value:
            raise ValueError(f"The scientific CNN architecture and split recipe are fixed ({key})")
    if "lr" in config:
        if "learning_rate" in config and config["learning_rate"] != config["lr"]:
            raise ValueError("Conflicting lr and learning_rate aliases")
        config["learning_rate"] = config.pop("lr")
    unknown = set(config) - set(FIXED_CONFIG) - {"test_mode", "device", "label_names", "metadata"}
    if unknown:
        raise ValueError(f"Unknown CNN config keys: {sorted(unknown)}")
    result = dict(FIXED_CONFIG)
    test_mode = config.get("test_mode", False)
    if not isinstance(test_mode, bool):
        raise ValueError("test_mode must be an explicit boolean")
    for key, default in FIXED_CONFIG.items():
        if key in config and config[key] != default:
            if not test_mode or key != "max_epochs" or not isinstance(config[key], int) or isinstance(config[key], bool) or not 1 <= config[key] <= 2:
                raise ValueError(f"The scientific CNN recipe is fixed; only max_epochs<=2 is allowed in explicit test_mode ({key})")
            result[key] = config[key]
    requested_device = config.get("device", "cpu" if test_mode else "auto")
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if not test_mode and requested_device != "auto":
        raise ValueError("Scientific fits must use automatic CUDA availability; explicit device is for test_mode")
    result.update(test_mode=test_mode, device=requested_device, validation_fraction=0.2, validation_split_owner="caller")
    return result


def _torch_setup(seed, requested_device):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch, torch.device(device)


def build_cnn(input_channels, n_classes):
    """Build the frozen architecture; output unnormalized class logits."""
    import torch
    nn = torch.nn
    return nn.Sequential(
        nn.Conv1d(input_channels, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 128), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(128, n_classes),
    )


def _predict(model, X, mean, scale, device, batch_size, n_classes):
    import torch
    if len(X) == 0:
        raise ValueError("Prediction input must be nonempty")
    model.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray((X[start:start + batch_size] - mean) / scale)).to(device)
            pieces.append(model(batch).detach().cpu())
    logits = torch.cat(pieces).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    if logits.shape != (len(X), n_classes) or not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise FloatingPointError("CNN produced nonfinite or malformed predictions")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise FloatingPointError("CNN softmax rows do not sum to one")
    return {"logits": logits, "probabilities": probabilities}


def fit_cnn(X, y_int, roles, seed, checkpoint_path, config=None):
    """Fit only training rows, select by validation NLL, then predict held-out rows.

    The caller must validate subject/protocol roles and supply label_names in the
    exact integer-class order. Without label_names, the mapping comes only from
    contiguous training class codes. Calibration/test labels never enter fitting,
    normalization, epoch selection, or architecture selection. Metadata is stored
    with the mandatory .pt checkpoint. test_mode outputs are explicitly marked.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y_int)
    if X.ndim != 3 or not len(X) or X.shape[1] < 1 or X.shape[2] < 4 or not np.isfinite(X).all():
        raise ValueError("CNN X must be nonempty finite N,C,T with at least four time steps")
    if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y_int must contain one integer label per row")
    roles = _validated_roles(roles, len(X))
    recipe = _configuration(config)
    config = dict(config or {})
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".pt":
        raise ValueError("A .pt checkpoint_path is mandatory")
    training_classes = np.unique(y[roles["train"]])
    supplied_labels = config.get("label_names")
    label_names = list(map(str, supplied_labels)) if supplied_labels is not None else list(map(str, range(int(training_classes.max()) + 1)))
    n_classes = len(label_names)
    if n_classes < 2 or len(set(label_names)) != n_classes or not np.array_equal(training_classes, np.arange(n_classes)):
        raise ValueError("Training rows must contain all contiguous class codes for the supplied label mapping")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("A label is outside the declared training class mapping")
    extra_metadata = json.loads(json.dumps(config.get("metadata", {}), allow_nan=False))
    torch, device = _torch_setup(int(seed), recipe["device"])
    training_X = X[roles["train"]]
    mean = training_X.mean(axis=(0, 2), dtype=np.float64, keepdims=True).astype(np.float32)
    scale = training_X.std(axis=(0, 2), dtype=np.float64, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    train_features = torch.from_numpy(np.ascontiguousarray((training_X - mean) / scale))
    train_labels = torch.from_numpy(y[roles["train"]].astype(np.int64))
    validation_features = torch.from_numpy(np.ascontiguousarray((X[roles["validation"]] - mean) / scale))
    validation_labels = torch.from_numpy(y[roles["validation"]].astype(np.int64))
    model = build_cnn(X.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    loss_function = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, train_labels),
                                             batch_size=recipe["batch_size"], shuffle=True,
                                             generator=generator, num_workers=0, drop_last=False)
    validation_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(validation_features, validation_labels),
                                                  batch_size=recipe["batch_size"], shuffle=False, num_workers=0)
    history, best_state = [], None
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    for epoch in range(1, recipe["max_epochs"] + 1):
        model.train()
        training_sum = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features.to(device)), targets.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            training_sum += float(loss.detach().cpu()) * len(targets)
        model.eval()
        validation_sum = 0.0
        with torch.inference_mode():
            for features, targets in validation_loader:
                loss = loss_function(model(features.to(device)), targets.to(device))
                validation_sum += float(loss.detach().cpu()) * len(targets)
        validation_nll = validation_sum / len(validation_labels)
        if not np.isfinite(validation_nll):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch, "training_nll": training_sum / len(train_labels), "validation_nll": validation_nll})
        if validation_nll < best_loss:
            best_loss, best_epoch, stale_epochs = validation_nll, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= recipe["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No finite validation-selected checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "schema_version": 1, "seed": int(seed), "architecture": ARCHITECTURE.copy(),
        "input_channels": int(X.shape[1]), "time_steps": int(X.shape[2]), "n_classes": n_classes,
        "label_names": label_names, "label_mapping": {str(index): label for index, label in enumerate(label_names)},
        "recipe": recipe, "test_mode": recipe["test_mode"], "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False,
        "normalization_role": "train", "normalization_ddof": 0, "near_constant_channel_threshold": 1e-8,
        "channel_mean": mean.reshape(-1).tolist(), "channel_scale": scale.reshape(-1).tolist(),
        "selection_role": "validation", "selection_metric": "NLL", "best_epoch": best_epoch,
        "best_validation_nll": best_loss, "epochs_completed": len(history),
        "stopped_early": len(history) < recipe["max_epochs"],
        "role_counts": {name: len(indices) for name, indices in roles.items()},
        "role_indices_sha256": {name: hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest() for name, indices in roles.items()},
        "subject_role_validation": "Caller responsibility; row roles checked disjoint and in range here",
        "reproducibility_scope": "Same device, platform, library versions and inputs; cross-device/version bitwise equivalence is not asserted",
        "caller_metadata": extra_metadata,
    }
    checkpoint = {"model_state_dict": best_state, "channel_mean": torch.from_numpy(mean.copy()),
                  "channel_scale": torch.from_numpy(scale.copy()), "metadata": metadata, "history": history,
                  "role_indices": {name: torch.from_numpy(indices.copy()) for name, indices in roles.items()}}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    result = {"history": history, "metadata": dict(metadata, checkpoint_sha256=_sha256(checkpoint_path), checkpoint_path=str(checkpoint_path))}
    for role in ("calibration", "test"):
        prediction = _predict(model, X[roles[role]], mean, scale, device, recipe["batch_size"], n_classes)
        result[f"{role}_logits"] = prediction["logits"]
        result[f"{role}_probabilities"] = prediction["probabilities"]
    return result


def predict_checkpoint(checkpoint_path, X, device="cpu", batch_size=128):
    """Reload a module-generated checkpoint and apply its saved normalization."""
    if device not in {"auto", "cpu", "cuda"} or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Invalid prediction device or batch size")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    saved = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    torch, target = _torch_setup(metadata["seed"], device)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (metadata["input_channels"], metadata["time_steps"]) or not np.isfinite(X).all():
        raise ValueError("Prediction X is nonfinite or differs from the checkpoint input schema")
    model = build_cnn(metadata["input_channels"], metadata["n_classes"]).to(target)
    model.load_state_dict(saved["model_state_dict"])
    result = _predict(model, X, saved["channel_mean"].numpy(), saved["channel_scale"].numpy(), target, batch_size, metadata["n_classes"])
    result["metadata"] = metadata
    return result