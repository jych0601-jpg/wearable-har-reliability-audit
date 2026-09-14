"""Frozen-artifact postprocessing with participant-level, seed-aware inference.

The four public stage functions take no arguments and never fit base models.
Only synthetic fixtures are used by this module's tests. Final analysis requires
all configured fits; incremental calibration stages never claim a final grid.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import ROOT, atomic_json, now, sha256
import reliability
from reliability import APS, OVRIsotonic, validate_probabilities
from src.calibration import TemperatureScaler, confidence_threshold
import src.calibration as v1_calibration
from src.experiment import evaluate_probabilities
from src.analysis import cluster_bootstrap_equal_dataset, cluster_sign_flip_p, holm_adjust

METHODS = ("raw", "temp", "iso")
CLASSICAL = ("logistic", "random_forest", "extra_trees", "hist_gradient_boosting")
PRIMARY_SEED = 20260911
IDENTITY = ["dataset", "model", "protocol", "seed"]
PERSON = ["dataset", "model", "seed", "subject"]
SCHEMA = 1


def _config():
    return json.loads((ROOT / "configs" / "expanded.json").read_text(encoding="utf-8"))


def successful_fits():
    """Discover the same completed attempt directories as experiments.successful_fits."""
    attempts = []
    for section in ("03_deep_har", "06_robustness"):
        for marker in sorted((ROOT / section / "fits").glob("*/completed.json")):
            attempts.append(ROOT / json.loads(marker.read_text(encoding="utf-8"))["attempt"])
    return attempts


def _records():
    records = []
    for path in successful_fits():
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            raise RuntimeError(f"Completed marker points at unsuccessful attempt: {path}")
        records.append((path, meta))
    return records


def _binding(path):
    names = ("metadata.json", "calibration_raw.npz", "test_raw.npz")
    inputs = {name: sha256(path / name) for name in names}
    marker_path = path.parent / "completed.json"
    if marker_path.exists():
        inputs["base_completion_sha256"] = sha256(marker_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        for item in marker.get("artifacts", []):
            source = ROOT / item["path"]
            if source.name in names and inputs[source.name] != item["sha256"]:
                raise RuntimeError(f"Frozen base input changed: {source}")
    inputs.update(config_sha256=sha256(ROOT / "configs" / "expanded.json"),
                  postprocess_stage_source_sha256=hashlib.sha256(Path(__file__).read_bytes().split(b"\ndef _require_grid(")[0]).hexdigest(),
                  reliability_source_sha256=sha256(reliability.__file__),
                  v1_calibration_source_sha256=sha256(v1_calibration.__file__))
    return inputs


def _resume(out, binding):
    marker = out / "completed.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("schema") != SCHEMA or saved.get("input_hashes") != binding:
        raise RuntimeError(f"Stale postprocessing result: inputs or implementation changed: {out}")
    for artifact in saved["artifacts"]:
        path = out / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Completed postprocessing artifact changed: {path}")
    return True


def _complete(out, binding, names):
    atomic_json(out / "completed.json", {
        "schema": SCHEMA, "status": "complete", "finished_utc": now(),
        "input_hashes": binding,
        "artifacts": [{"path": name, "sha256": sha256(out / name)} for name in names],
    })


def _load_pair(path, meta):
    parts = []
    required = {"probabilities", "logits", "y", "subjects", "sample_ids", "row_indices", "classes"}
    for name in ("calibration", "test"):
        with np.load(path / f"{name}_raw.npz", allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise ValueError(f"Missing prediction fields in {path / name}")
            part = {key: archive[key] for key in required}
        p = validate_probabilities(part["probabilities"])
        n, k = p.shape
        y = part["y"]
        if y.ndim != 1 or len(y) != n or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= k):
            raise ValueError("Labels must be integer indices matching the global class columns")
        if part["logits"].shape != p.shape or not np.isfinite(part["logits"]).all():
            raise ValueError("Invalid saved logits")
        if part["classes"].ndim != 1 or len(part["classes"]) != k or len(np.unique(part["classes"])) != k:
            raise ValueError("Invalid global class schema")
        for key in ("subjects", "sample_ids", "row_indices"):
            if part[key].ndim != 1 or len(part[key]) != n:
                raise ValueError(f"Invalid {key} shape")
        if part["row_indices"].dtype.kind not in "iu" or np.any(part["row_indices"] < 0):
            raise ValueError("Global row indices must be nonnegative integers")
        for key in ("sample_ids", "row_indices"):
            if len(np.unique(part[key])) != n:
                raise ValueError(f"Duplicate {key} within {name} constitutes row leakage")
        if any(value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
               for value in part["subjects"].tolist()):
            raise ValueError("Missing participant identifier")
        part["probabilities"] = p
        parts.append(part)
    cal, test = parts
    if not np.array_equal(cal["classes"], test["classes"]):
        raise ValueError("Calibration/test global class columns differ")
    if "classes" in meta and not np.array_equal(cal["classes"].astype(str), np.asarray(meta["classes"]).astype(str)):
        raise ValueError("Artifact class schema differs from frozen metadata")
    for key in ("sample_ids", "row_indices"):
        if set(cal[key].tolist()).intersection(test[key].tolist()):
            raise ValueError(f"Calibration/test {key} overlap constitutes leakage")
    if meta["protocol"] == "subject_disjoint" and set(cal["subjects"].tolist()).intersection(test["subjects"].tolist()):
        raise ValueError("Calibration/test participant overlap in subject-disjoint protocol")
    return cal, test


def _stage(kind):
    config = _config()
    records = _records()
    resumed = 0
    section = "04_calibration" if kind == "calibration" else "05_conformal"
    for path, meta in records:
        out = ROOT / section / path.parent.name
        binding = _binding(path)
        if _resume(out, binding):
            resumed += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            cal, test = _load_pair(path, meta)
            common = {"identity": {key: meta[key] for key in IDENTITY + ["fold", "representation"]},
                      "base_attempt": str(path.relative_to(ROOT)), "input_hashes": binding,
                      "global_classes": cal["classes"].tolist(), "created_utc": now()}
            if kind == "calibration":
                # Both learned maps see the dedicated calibration labels only.
                temp = TemperatureScaler().fit(cal["probabilities"], cal["y"])
                iso = OVRIsotonic().fit(cal["probabilities"], cal["y"])
                cal_probs = {"raw": cal["probabilities"], "temp": temp.transform(cal["probabilities"]), "iso": iso.transform(cal["probabilities"])}
                test_probs = {"raw": test["probabilities"], "temp": temp.transform(test["probabilities"]), "iso": iso.transform(test["probabilities"])}
                for collection in (cal_probs, test_probs):
                    for p in collection.values():
                        validate_probabilities(p)
                thresholds = {method: {f"{float(c):g}": confidence_threshold(p, float(c))
                                       for c in config["target_coverages"]} for method, p in cal_probs.items()}
                np.savez_compressed(out / "calibration_probabilities.npz", **cal_probs)
                np.savez_compressed(out / "test_probabilities.npz", **test_probs)
                joblib.dump({"temp": temp, "iso": iso}, out / "calibrators.joblib", compress=3)
                atomic_json(out / "methods.json", {**common, "methods": list(METHODS),
                    "temperature": temp.temperature_, "thresholds": thresholds,
                    "temperature_input": "V1 clipped log probabilities, including CNN; native logits retained in base artifacts",
                    "threshold_rule": "V1 calibration confidence quantile, method=higher; target 1 is the minimum calibration confidence, not a guaranteed full-test-retention threshold",
                    "test_labels_used_for_fit": False,
                    "isotonic_rule": "OVR increasing maps, normalized; raw fallback for all-zero mappings"})
                names = ["calibration_probabilities.npz", "test_probabilities.npz", "calibrators.joblib", "methods.json"]
            else:
                sets, thresholds = {}, {}
                for alpha in config["alpha"]:
                    estimator = APS(float(alpha)).fit(cal["probabilities"], cal["y"])
                    sets[f"alpha_{float(alpha):g}"] = estimator.predict(test["probabilities"])
                    q = estimator.threshold_
                    thresholds[f"{float(alpha):g}"] = {"threshold": q if math.isfinite(q) else None,
                        "threshold_kind": "finite" if math.isfinite(q) else "positive_infinity",
                        "n_calibration": len(cal["y"])}
                np.savez_compressed(out / "prediction_sets.npz", **sets)
                atomic_json(out / "methods.json", {**common, "probability_method": "raw", "thresholds": thresholds,
                    "score": "nonrandomized cumulative-inclusive APS; stable ascending class-index ties",
                    "set_rule": "score <= finite-sample threshold; empty sets preserved",
                    "probability_calibrator_used": False, "test_labels_used_for_fit": False,
                    "reference": "Romano, Sesia, Candes (2020), https://arxiv.org/abs/2006.02544"})
                names = ["prediction_sets.npz", "methods.json"]
            _complete(out, binding, names)
        except Exception as exc:
            atomic_json(out / ("failed_" + now().replace(":", "").replace(".", "") + ".json"),
                        {"status": "failed", "input_hashes": binding, "error_type": type(exc).__name__, "error": str(exc)})
            raise
    expected = len(config["datasets"]) * len(config["models"]) * len(config["protocols"]) * len(config["seeds"]) * config["n_folds"]
    return {"stage": kind, "status": "incremental", "completed_fits": len(records), "resumed_fits": resumed, "intended_base_fits": expected}


def calibrate_all():
    return _stage("calibration")


def conformal_all():
    return _stage("conformal")


def _require_grid(records, config):
    expected = set(itertools.product(config["datasets"], config["models"], config["protocols"], config["seeds"], range(config["n_folds"])))
    actual = [(m["dataset"], m["model"], m["protocol"], m["seed"], m["fold"]) for _, m in records]
    if len(actual) != len(set(actual)):
        raise RuntimeError("Duplicate completed fit identities")
    missing = expected.difference(actual)
    if missing:
        raise RuntimeError(f"Incomplete intended grid: {len(missing)} of {len(expected)} fits missing; no final analysis written")
    extras = set(actual).difference(expected)
    for dataset, model, protocol, seed, fold in extras:
        if dataset != "uci_har" or protocol != "leakage_reduced_mixed" or seed != PRIMARY_SEED or model not in config["models"] or fold not in range(config["n_folds"]):
            raise RuntimeError(f"Unrecognized fit outside the intended grid: {(dataset, model, protocol, seed, fold)}")
    if extras and not set(CLASSICAL).intersection(config["models"]).issubset({item[1] for item in extras}):
        raise RuntimeError("Incomplete optional classical-model grid")
    for model in {item[1] for item in extras}:
        if sum(item[1] == model for item in extras) != config["n_folds"]:
            raise RuntimeError(f"Incomplete optional leakage-reduced fold group for {model}")
    return {"expected_base_fits": len(expected), "completed_base_fits": len(expected), "optional_fits": len(extras)}


def _participant_tables(records, config):
    groups = {}
    classes_by_dataset = {}
    for path, meta in records:
        binding = _binding(path)
        caldir, apsdir = (ROOT / section / path.parent.name for section in ("04_calibration", "05_conformal"))
        if not _resume(caldir, binding) or not _resume(apsdir, binding):
            raise RuntimeError("Required calibration or conformal stage is incomplete")
        _, test = _load_pair(path, meta)
        schema = tuple(test["classes"].astype(str))
        if meta["dataset"] in classes_by_dataset and classes_by_dataset[meta["dataset"]] != schema:
            raise ValueError("Global class columns change across fits in a dataset")
        classes_by_dataset[meta["dataset"]] = schema
        with np.load(caldir / "test_probabilities.npz", allow_pickle=False) as z:
            probs = {method: validate_probabilities(z[method]) for method in METHODS}
        for p in probs.values():
            if p.shape != test["probabilities"].shape:
                raise ValueError("Derived probabilities do not align with frozen test rows")
        if not np.array_equal(probs["raw"], test["probabilities"]):
            raise ValueError("Derived raw probabilities differ from frozen test probabilities")
        with np.load(apsdir / "prediction_sets.npz", allow_pickle=False) as z:
            masks = {float(alpha): z[f"alpha_{float(alpha):g}"] for alpha in config["alpha"]}
        for mask in masks.values():
            if mask.dtype != bool or mask.shape != test["probabilities"].shape:
                raise ValueError("Conformal mask shape or dtype is invalid")
        thresholds = json.loads((caldir / "methods.json").read_text(encoding="utf-8"))["thresholds"]
        key = tuple(meta[key] for key in IDENTITY)
        groups.setdefault(key, []).append((meta, test, probs, masks, thresholds))
    metrics, selective, conformal = [], [], []
    for key, pieces in groups.items():
        identity = dict(zip(IDENTITY, key))
        identity["representation"] = pieces[0][0]["representation"]
        if any(piece[0]["representation"] != identity["representation"] for piece in pieces):
            raise ValueError("Representation changes across cross-fitting folds")
        y = np.concatenate([x[1]["y"] for x in pieces])
        subjects = np.concatenate([x[1]["subjects"] for x in pieces])
        for field in ("sample_ids", "row_indices"):
            ids = np.concatenate([x[1][field] for x in pieces])
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate cross-fitted test {field}")
        probs = {m: np.concatenate([x[2][m] for x in pieces]) for m in METHODS}
        masks = {float(a): np.concatenate([x[3][float(a)] for x in pieces]) for a in config["alpha"]}
        for subject in np.unique(subjects):
            ix = subjects == subject
            labels = y[ix]
            who = {**identity, "subject": subject, "n_windows": int(ix.sum()), "n_folds_contributed": sum(bool(np.any(x[1]["subjects"] == subject)) for x in pieces)}
            for method, p in probs.items():
                local = p[ix]
                values = evaluate_probabilities(labels, local, p.shape[1])
                gap = values["confidence_gap"]
                metrics.append({**who, "method": method, **values, "error": 1 - values["accuracy"],
                    "mean_confidence": float(local.max(axis=1).mean()),
                    "overconfidence_positive_gap": max(gap, 0.), "overconfidence_negative_gap": min(gap, 0.)})
                for target in config["target_coverages"]:
                    row_thresholds = np.concatenate([np.full(len(x[1]["y"]), x[4][method][f"{float(target):g}"]) for x in pieces])[ix]
                    retained = local.max(axis=1) >= row_thresholds
                    count = int(retained.sum())
                    selective.append({**who, "method": method, "target_coverage": float(target), "n_retained": count,
                        "coverage": float(retained.mean()), "selective_error": float((local[retained].argmax(axis=1) != labels[retained]).mean()) if count else np.nan,
                        "selective_error_reason": "defined" if count else "no_retained_windows",
                        "full_error": 1 - values["accuracy"]})
            for alpha, mask in masks.items():
                local = mask[ix]
                sizes = local.sum(axis=1)
                singleton = sizes == 1
                count = int(singleton.sum())
                conformal.append({**who, "method": "raw", "alpha": alpha, "nominal_coverage": 1 - alpha,
                    "coverage": float(local[np.arange(len(labels)), labels].mean()),
                    "mean_size": float(sizes.mean()), "median_size": float(np.median(sizes)),
                    "singleton_fraction": float(singleton.mean()), "empty_fraction": float((sizes == 0).mean()),
                    "n_singleton": count, "n_empty": int((sizes == 0).sum()),
                    "full_error": float((probs["raw"][ix].argmax(axis=1) != labels).mean()),
                    "singleton_error": float((local[singleton].argmax(axis=1) != labels[singleton]).mean()) if count else np.nan,
                    "singleton_error_reason": "defined" if count else "no_singletons"})
    return pd.DataFrame(metrics), pd.DataFrame(selective), pd.DataFrame(conformal)



def _pool_participants(frame, models, seeds, require_all_models=True):
    """Average repeated seeds inside each model before equal-model person means.

    Undefined risks retain their participant and availability counts. Missing
    seed records are errors; undefined values in existing records are distinct.
    Expanded pooling requires a defined mean from every model. The original
    V1 primary family passes require_all_models=False to preserve its mean of
    available model differences; every availability count remains explicit.
    """
    selected = frame[frame.model.isin(models) & frame.seed.isin(seeds)].copy()
    if selected.duplicated(["dataset", "model", "seed", "subject"]).any():
        raise ValueError("Duplicate participant/model/seed effect records")
    model_rows = []
    for (dataset, subject, model), group in selected.groupby(["dataset", "subject", "model"], sort=True, observed=True):
        if set(group.seed) != set(seeds):
            raise ValueError(f"Incomplete seed records for {(dataset, subject, model)}")
        values = group.value.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite derived effect")
        available = values[np.isfinite(values)]
        model_rows.append({"dataset": dataset, "subject": subject, "model": model,
                           "value": float(available.mean()) if len(available) else np.nan,
                           "n_seeds_available": len(available)})
    rows = []
    model_frame = pd.DataFrame(model_rows)
    if model_frame.empty:
        return pd.DataFrame(columns=["dataset", "subject", "value", "n_models_expected", "n_models_available", "n_seeds_expected", "n_seeds_available_min", "n_seeds_available_max", "undefined_reason"])
    for (dataset, subject), group in model_frame.groupby(["dataset", "subject"], sort=True, observed=True):
        available = int(group.value.notna().sum())
        complete = set(group.model) == set(models) and available == len(models)
        keep = complete if require_all_models else available > 0
        rows.append({"dataset": dataset, "subject": subject,
                     "value": float(group.value.mean()) if keep else np.nan,
                     "n_models_expected": len(models), "n_models_available": available,
                     "n_seeds_expected": len(seeds),
                     "n_seeds_available_min": int(group.n_seeds_available.min()),
                     "n_seeds_available_max": int(group.n_seeds_available.max()),
                     "undefined_reason": "defined" if keep else "one_or_more_models_unavailable",
                     "model_availability_policy": "all_required_models" if require_all_models else "v1_available_models"})
    return pd.DataFrame(rows)


def _effects(metrics, selective, conformal):
    frames = []

    def add(frame, name, kind="contrast"):
        value = frame[PERSON + ["value"]].copy()
        value["effect"] = name
        value["kind"] = kind
        value["undefined_reason"] = np.where(value.value.isna(), "one_or_more_component_estimands_undefined", "defined")
        frames.append(value)

    raw = metrics[metrics.method.eq("raw")]
    for metric in ("macro_f1", "brier", "nll", "ece_15"):
        wide = raw.pivot(index=PERSON, columns="protocol", values=metric).reset_index()
        wide["value"] = wide["subject_disjoint"] - wide["sample_mixed"]
        add(wide, "protocol_" + metric)
        if "leakage_reduced_mixed" in wide:
            extra = wide[wide.dataset.eq("uci_har") & wide.seed.eq(PRIMARY_SEED) & wide.leakage_reduced_mixed.notna()].copy()
            extra["value"] = extra["leakage_reduced_mixed"] - extra["sample_mixed"]
            add(extra, "leakage_reduced_protocol_" + metric)
    disjoint = metrics[metrics.protocol.eq("subject_disjoint")]
    for metric in ("brier", "ece_15", "macro_f1"):
        wide = disjoint.pivot(index=PERSON, columns="method", values=metric).reset_index()
        for method in ("temp", "iso"):
            wide["value"] = wide[method] - wide["raw"]
            add(wide, f"calibration_{method}_{metric}")
    for method in METHODS:
        selected = selective[selective.protocol.eq("subject_disjoint") & selective.method.eq(method) & np.isclose(selective.target_coverage, .8)].copy()
        selected["value"] = selected.selective_error - selected.full_error
        add(selected, "selective80_" + method)
    for alpha, group in conformal[conformal.protocol.eq("subject_disjoint")].groupby("alpha", sort=True):
        group = group.copy()
        for metric in ("coverage_gap", "mean_size", "singleton_error"):
            group["value"] = group.coverage - (1 - alpha) if metric == "coverage_gap" else group[metric]
            add(group, f"aps_a{alpha:g}_{metric}", "contrast" if metric == "coverage_gap" else "descriptive")
    return pd.concat(frames, ignore_index=True)


def _infer(pooled, config, test=True):
    available = pooled.dropna(subset=["value"])
    total_datasets = int(pooled.dataset.nunique())
    row = {"n_subjects_total": len(pooled), "n_subjects_available": len(available),
           "n_subjects_unavailable": len(pooled) - len(available),
           "n_datasets_total": total_datasets, "n_datasets_available": int(available.dataset.nunique()),
           "n_seeds_expected": int(pooled.n_seeds_expected.iloc[0]) if len(pooled) else 0,
           "n_models_expected": int(pooled.n_models_expected.iloc[0]) if len(pooled) else 0,
           "n_models_available_min": int(pooled.n_models_available.min()) if len(pooled) else 0,
           "n_models_available_max": int(pooled.n_models_available.max()) if len(pooled) else 0,
           "model_availability_policy": pooled.model_availability_policy.iloc[0] if len(pooled) else "undefined",
           "n_seeds_available_min": int(pooled.n_seeds_available_min.min()) if len(pooled) else 0,
           "n_seeds_available_max": int(pooled.n_seeds_available_max.max()) if len(pooled) else 0,
           "draws": int(config["bootstrap_draws"])}
    if available.empty or available.dataset.nunique() != total_datasets:
        return {**row, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "sign_flip_two_sided_p": np.nan, "undefined_reason": "one_or_more_datasets_have_no_available_participants"}
    payload = available[["dataset", "subject", "value"]].sort_values(["dataset", "subject"]).to_csv(index=False, float_format="%.17g")
    key = hashlib.sha256((payload + str((config["bootstrap_draws"], config["statistical_seed"], test, SCHEMA))).encode()).hexdigest()
    cache = ROOT / "07_analysis" / "inference_cache" / (key + ".json")
    if cache.is_file():
        estimate = json.loads(cache.read_text(encoding="utf-8"))
    else:
        estimate = cluster_bootstrap_equal_dataset(available, draws=config["bootstrap_draws"], seed=config["statistical_seed"])
        estimate["sign_flip_two_sided_p"] = cluster_sign_flip_p(available, draws=config["bootstrap_draws"], seed=config["statistical_seed"] + 1000) if test else None
        atomic_json(cache, estimate)
    return {**row, **estimate, "undefined_reason": "defined",
            "estimand_population": "available participants within each dataset; model_availability_policy explicitly recorded; available repeated-seed values averaged within model"}


def _adjust_families(table):
    """Keep untestable planned contrasts in Holm's family as p=1 placeholders.

    The displayed p and adjusted p remain undefined for an untestable estimand.
    Descriptive summaries never enter a hypothesis-testing multiplicity family.
    """
    table = table.copy()
    table["holm_adjusted_p"] = np.nan
    table["n_hypotheses_planned"] = 0
    table["n_hypotheses_testable"] = 0
    for family, group in table.groupby("multiplicity_family", sort=True):
        planned = group[group.kind.eq("contrast")]
        testable = planned.sign_flip_two_sided_p.notna()
        table.loc[group.index, "n_hypotheses_planned"] = len(planned)
        table.loc[group.index, "n_hypotheses_testable"] = int(testable.sum())
        adjusted = holm_adjust(planned.sign_flip_two_sided_p.fillna(1.0))
        for index, value in zip(planned.index, adjusted):
            if pd.notna(table.loc[index, "sign_flip_two_sided_p"]):
                table.loc[index, "holm_adjusted_p"] = value
    return table


def _inference_tables(effects, config):
    primary_map = {
        "protocol_macro_f1": "H1_disjoint_minus_mixed_macro_f1",
        "protocol_brier": "H2_disjoint_minus_mixed_brier",
        "calibration_temp_brier": "H3_calibrated_minus_raw_disjoint_brier",
        "selective80_temp": "H4_selective80_minus_full_error_disjoint",
    }
    rows, pooled_frames = [], []
    if set(CLASSICAL).issubset(config["models"]) and PRIMARY_SEED in config["seeds"]:
        for primary_index, (effect, name) in enumerate(primary_map.items()):
            selected = effects[effects.effect.eq(effect)]
            pooled = _pool_participants(selected, list(CLASSICAL), [PRIMARY_SEED], require_all_models=False)
            ident = {"family": "primary_v1_classical_four_models", "scope": "pooled", "dataset_scope": "all", "model_scope": "all_classical", "effect": effect, "estimand": name, "kind": "contrast"}
            rows.append({**ident, **_infer(pooled, {**config, "statistical_seed": config["statistical_seed"] + primary_index})})
            pooled_frames.append(pooled.assign(**ident))
    for effect, group in effects.groupby("effect", sort=True):
        optional = effect.startswith("leakage_reduced_")
        seeds = [PRIMARY_SEED] if optional else config["seeds"]
        models = sorted(group.model.unique()) if optional else config["models"]
        datasets = sorted(group.dataset.unique())
        scopes = [("pooled", "all", "all", models)]
        scopes += [("model", "all", model, [model]) for model in models]
        scopes += [("dataset", dataset, "all", models) for dataset in datasets]
        scopes += [("model_dataset", dataset, model, [model]) for dataset in datasets for model in models]
        for scope, dataset, model, scope_models in scopes:
            selected = group if dataset == "all" else group[group.dataset.eq(dataset)]
            pooled = _pool_participants(selected, scope_models, seeds)
            ident = {"family": "temporal_overlap_supplement" if optional else "expanded_secondary", "scope": scope,
                     "dataset_scope": dataset, "model_scope": model, "effect": effect, "estimand": effect, "kind": group.kind.iloc[0]}
            rows.append({**ident, **_infer(pooled, config, test=group.kind.iloc[0] == "contrast")})
            pooled_frames.append(pooled.assign(**ident))
    table = pd.DataFrame(rows)
    table["holm_adjusted_p"] = np.nan
    table["multiplicity_family"] = table.family.map({
        "primary_v1_classical_four_models": "four_original_primary_tests",
        "expanded_secondary": "all_reported_expanded_secondary_contrasts_including_supplementary_scopes",
        "temporal_overlap_supplement": "all_temporal_overlap_supplement_contrasts"})
    table = _adjust_families(table)
    return table, pd.concat(pooled_frames, ignore_index=True)


def _seed_tables(effects, config):
    rows, model_rows = [], []
    for effect, group in effects.groupby("effect", sort=True):
        optional = effect.startswith("leakage_reduced_")
        models = sorted(group.model.unique()) if optional else config["models"]
        seeds = [PRIMARY_SEED] if optional else config["seeds"]
        for seed in seeds:
            pooled = _pool_participants(group, models, [seed])
            available = pooled.dropna(subset=["value"])
            valid = not available.empty and available.dataset.nunique() == pooled.dataset.nunique()
            point = float(available.groupby("dataset").value.mean().mean()) if valid else np.nan
            rows.append({"effect": effect, "seed": seed, "estimate": point, "n_subjects_total": len(pooled),
                         "n_subjects_available": len(available), "n_models_expected": len(models)})
        for (dataset, model, seed), part in group.groupby(["dataset", "model", "seed"], sort=True):
            model_rows.append({"effect": effect, "dataset": dataset, "model": model, "seed": seed,
                               "estimate": part.value.mean(), "n_subjects_total": len(part), "n_subjects_available": int(part.value.notna().sum())})
    seeds = pd.DataFrame(rows)
    summaries = []
    for effect, group in seeds.groupby("effect", sort=True):
        values = group.estimate.dropna().to_numpy()
        mean = float(values.mean()) if len(values) else np.nan
        summaries.append({"effect": effect, "n_seeds_planned": len(group), "n_seeds_available": len(values),
            "mean": mean, "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "median": float(np.median(values)) if len(values) else np.nan,
            "q25": float(np.quantile(values, .25)) if len(values) else np.nan,
            "q75": float(np.quantile(values, .75)) if len(values) else np.nan,
            "iqr": float(np.quantile(values, .75) - np.quantile(values, .25)) if len(values) else np.nan,
            "q025": float(np.quantile(values, .025)) if len(values) else np.nan,
            "q975": float(np.quantile(values, .975)) if len(values) else np.nan,
            "positive_fraction": float((values > 0).mean()) if len(values) else np.nan,
            "negative_fraction": float((values < 0).mean()) if len(values) else np.nan,
            "zero_fraction": float((values == 0).mean()) if len(values) else np.nan,
            "direction_consistency": float((np.sign(values) == np.sign(mean)).mean()) if len(values) else np.nan,
            "interpretation": "descriptive split-seed sensitivity; seeds are not independent participants"})
    return seeds, pd.DataFrame(summaries), pd.DataFrame(model_rows)


def _equal_person_summary(frame, grouping, measures):
    # Repeated seeds are averaged first, then participants receive equal weight.
    person = frame.groupby(grouping + ["subject"], observed=True, dropna=False, as_index=False)[measures].mean()
    summary = person.groupby(grouping, observed=True, dropna=False)[measures].mean().reset_index()
    counts = person.groupby(grouping, observed=True, dropna=False).size().reset_index(name="n_subjects_total")
    summary = summary.merge(counts, on=grouping, validate="one_to_one")
    for measure in measures:
        available = person.groupby(grouping, observed=True, dropna=False)[measure].count().reset_index(name="n_subjects_available_" + measure)
        summary = summary.merge(available, on=grouping, validate="one_to_one")
    return summary


def _write_table(frame, directory, name, note=""):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / (name + ".csv"), index=False, na_rep="")
    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ([note, ""] if note else []) + ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    (directory / (name + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_all():
    config, records = _config(), _records()
    grid = _require_grid(records, config)
    metrics, selective, conformal = _participant_tables(records, config)
    processed, tables = ROOT / "07_analysis", ROOT / "09_tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(processed / "participant_metrics.csv", index=False)
    selective.to_csv(processed / "participant_selective.csv", index=False, na_rep="")
    conformal.to_csv(processed / "participant_conformal.csv", index=False, na_rep="")
    effects = _effects(metrics, selective, conformal)
    effects.to_csv(processed / "participant_effects_by_seed.csv", index=False, na_rep="")
    inference, pooled = _inference_tables(effects, config)
    pooled.to_csv(processed / "participant_effects_seed_averaged.csv", index=False, na_rep="")
    seeds, seed_summary, model_seeds = _seed_tables(effects, config)
    seeds.to_csv(processed / "seed_effects.csv", index=False, na_rep="")
    model_seeds.to_csv(processed / "seed_effects_modelwise.csv", index=False, na_rep="")
    note = ("Participant is the inferential unit. Primary: four original hypotheses, four classical models, seed 20260911. "
            "Expanded: repeated seeds averaged within model/participant, models equally weighted within participant, datasets equally weighted. "
            "NA denotes an undefined estimand; availability counts are retained. CIs: participant cluster bootstrap; p values: paired sign flips; "
            "Holm covers each declared family, including all supplementary expanded contrasts. Non-significant and adverse effects are retained.")
    _write_table(inference[inference.family.eq("primary_v1_classical_four_models")], tables, "table_primary_estimands", note)
    _write_table(inference[inference.family.eq("expanded_secondary") & inference.scope.eq("pooled")], tables, "table_expanded_estimands", note)
    _write_table(inference[inference.scope.isin(["model", "model_dataset"])], tables, "table_modelwise_effects", note)
    _write_table(inference[inference.scope.eq("dataset")], tables, "table_dataset_effects", note)
    _write_table(seed_summary, tables, "table_seed_summary", "Descriptive sensitivity across fixed split seeds; these quantiles are not participant confidence intervals.")
    _write_table(inference[inference.family.eq("temporal_overlap_supplement")], tables, "table_temporal_overlap_supplement", "UCI exact shared-window overlap components reduce one leakage mechanism; this is not reconstructed-session splitting or a guarantee of absent residual temporal autocorrelation.")
    model_summary = _equal_person_summary(metrics, ["dataset", "model", "protocol", "method", "representation"],
        ["accuracy", "macro_f1", "brier", "nll", "ece_15", "confidence_gap", "mean_confidence", "overconfidence_positive_gap", "overconfidence_negative_gap", "aurc", "error"])
    selective_summary = _equal_person_summary(selective, ["dataset", "model", "protocol", "method", "target_coverage"], ["coverage", "selective_error", "full_error"])
    conformal_summary = _equal_person_summary(conformal, ["dataset", "model", "protocol", "method", "alpha", "nominal_coverage"],
        ["coverage", "mean_size", "median_size", "singleton_fraction", "empty_fraction", "full_error", "singleton_error"])
    _write_table(model_summary, tables, "table_model_summary", note)
    _write_table(selective_summary, tables, "table_selective_summary", "Six calibration-derived thresholds; undefined selective risk is preserved with availability counts. Target 1 follows the V1 minimum-calibration-confidence rule.")
    _write_table(conformal_summary, tables, "table_conformal_summary", "Raw nonrandomized APS; empty sets included; singleton error is conditional on an available singleton. Coverage under shift is observed, not guaranteed by disjointness.")
    source_files = [processed / name for name in ("participant_metrics.csv", "participant_selective.csv", "participant_conformal.csv", "participant_effects_by_seed.csv", "participant_effects_seed_averaged.csv", "seed_effects.csv", "seed_effects_modelwise.csv")]
    source_files += sorted(tables.glob("table_*.csv"))
    summary = {"status": "complete", **grid, "n_participant_metric_rows": len(metrics), "n_selective_rows": len(selective),
               "n_conformal_rows": len(conformal), "n_seeds": len(config["seeds"]), "bootstrap_draws": config["bootstrap_draws"],
               "source_csvs": [str(path.relative_to(ROOT)) for path in source_files],
               "config_sha256": sha256(ROOT / "configs" / "expanded.json"), "postprocess_source_sha256": sha256(__file__),
               "artifact_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_files}, "finished_utc": now()}
    atomic_json(processed / "completed.json", summary)
    return summary


def figures_all():
    """Render five publication figures, plus the optional overlap supplement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker_path = ROOT / "07_analysis" / "completed.json"
    if not marker_path.is_file():
        raise RuntimeError("Final analysis is incomplete; publication figures are unavailable")
    analysis = json.loads(marker_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "complete" or analysis.get("config_sha256") != sha256(ROOT / "configs" / "expanded.json"):
        raise RuntimeError("Final analysis is incomplete or stale")
    for name, digest in analysis["artifact_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Analysis source CSV changed: {name}")
    config = _config()
    out = ROOT / "08_figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15,
                         "axes.labelsize": 14, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    colors = {"raw": "#444444", "temp": "#0072B2", "iso": "#D55E00"}
    model_names = {"logistic": "Logistic", "random_forest": "Random forest", "extra_trees": "Extra trees",
                   "hist_gradient_boosting": "Hist. boosting", "cnn1d": "1D CNN"}
    dataset_names = {"uci_har": "UCI HAR", "wisdm_watch_accel": "WISDM"}
    models = config["models"]
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    model_colors = dict(zip(models, palette))
    figures = []

    def save(fig, name, sources, caption):
        fig.savefig(out / (name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight", facecolor="white", metadata={"Title": caption, "Creator": "Frozen HAR postprocessing"})
        plt.close(fig)
        figures.append({"name": name, "png": str((out / (name + ".png")).relative_to(ROOT)),
                        "pdf": str((out / (name + ".pdf")).relative_to(ROOT)), "dpi": 300,
                        "source_csvs": sources, "caption": caption,
                        "png_sha256": sha256(out / (name + ".png")), "pdf_sha256": sha256(out / (name + ".pdf"))})

    effect_path = "09_tables/table_modelwise_effects.csv"
    effects = pd.read_csv(ROOT / effect_path)
    selected = effects[effects.family.eq("expanded_secondary") & effects.scope.eq("model_dataset")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("macro_f1", "brier", "nll", "ece_15")):
        group = selected[selected.effect.eq("protocol_" + metric)].sort_values(["dataset_scope", "model_scope"])
        y = np.arange(len(group))
        ax.errorbar(group.estimate, y,
                    xerr=np.maximum(0, np.vstack([group.estimate - group.ci_low, group.ci_high - group.estimate])),
                    fmt="o", color="#0072B2", capsize=2, markersize=4)
        ax.set_yticks(y, [dataset_names.get(d, d) + " | " + model_names.get(m, m) for d, m in zip(group.dataset_scope, group.model_scope)])
        ax.axvline(0, color="0.6", linewidth=.8, linestyle="--")
        ax.set_title({"macro_f1": "Macro-F1", "brier": "Brier score", "nll": "Negative log likelihood", "ece_15": "ECE (15 bins)"}[metric])
        ax.set_xlabel("Subject-disjoint minus sample-mixed")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    save(fig, "figure_1_protocol_effects", [effect_path], "Protocol effects by dataset and model; repeated seeds averaged within participant. Participant bootstrap 95% intervals.")

    model_path = "09_tables/table_model_summary.csv"
    summary = pd.read_csv(ROOT / model_path)
    disjoint = summary[summary.protocol.eq("subject_disjoint")]
    keys = sorted(set(zip(disjoint.dataset, disjoint.model)))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .43 * len(keys) + 1.8)), constrained_layout=True)
    for ax, metric in zip(axes, ("brier", "ece_15")):
        for offset, method in zip((-.18, 0, .18), METHODS):
            lookup = disjoint[disjoint.method.eq(method)].set_index(["dataset", "model"])[metric]
            ax.plot([lookup.loc[key] for key in keys], np.arange(len(keys)) + offset, "o", markersize=5, color=colors[method], label=method.title())
        ax.set_yticks(np.arange(len(keys)), [dataset_names.get(d, d) + " | " + model_names.get(m, m) for d, m in keys])
        ax.set_xlabel("Brier score" if metric == "brier" else "ECE (15 bins)")
        ax.set_title("Subject-disjoint probability quality")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False)
    save(fig, "figure_2_calibration", [model_path], "Raw, temperature-scaled, and isotonic probability quality. Equal participant weights after within-participant seed averaging; adverse calibration results retained.")

    selective_path = "09_tables/table_selective_summary.csv"
    selective = pd.read_csv(ROOT / selective_path)
    selective = selective[selective.protocol.eq("subject_disjoint")]
    datasets = sorted(selective.dataset.unique())
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 3.6 * len(datasets)), squeeze=False, constrained_layout=True)
    for row, dataset in enumerate(datasets):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            for model in models:
                group = selective[selective.dataset.eq(dataset) & selective.model.eq(model) & selective.method.eq(method)].sort_values("coverage")
                if group.empty:
                    continue
                # Include the full-test-risk anchor separately from calibration thresholds.
                x = np.r_[group.coverage.to_numpy(), 1.0]
                y = np.r_[group.selective_error.to_numpy(), group.full_error.iloc[0]]
                order = np.argsort(x, kind="stable")
                ax.plot(x[order], y[order], "o-", markersize=3, color=model_colors[model], label=model_names.get(model, model))
            ax.set_title(dataset_names.get(dataset, dataset) + " | " + method.title())
            ax.set_xlabel("Achieved participant-mean\nretention")
            ax.set_ylabel("Participant-mean retained error")
            ax.set_xlim(0, 1.03)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=11)
    save(fig, "figure_3_selective_risk_coverage", [selective_path], "Risk versus achieved retention at six frozen calibration thresholds, with the actual full-risk endpoint. Risks are available-participant means; availability counts remain in the source table.")

    aps_path = "09_tables/table_conformal_summary.csv"
    aps = pd.read_csv(ROOT / aps_path)
    aps = aps[aps.protocol.eq("subject_disjoint")]
    keys = sorted(set(zip(aps.dataset, aps.model)))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .43 * len(keys) + 1.8)), constrained_layout=True)
    for index, alpha in enumerate(sorted(aps.alpha.unique())):
        color = ("#0072B2", "#D55E00")[index % 2]
        group = aps[np.isclose(aps.alpha, alpha)].set_index(["dataset", "model"])
        y = np.arange(len(keys)) + (index - .5) * .22
        for ax, metric in zip(axes, ("coverage", "mean_size")):
            ax.plot([group.loc[key, metric] for key in keys], y, "o", color=color, markersize=5, label=f"Nominal {1-alpha:.0%}")
        axes[0].axvline(1 - alpha, color=color, linestyle="--", linewidth=.8, alpha=.7)
    for ax, title in zip(axes, ("Observed APS coverage", "Mean prediction-set size")):
        ax.set_yticks(np.arange(len(keys)), [dataset_names.get(d, d) + " | " + model_names.get(m, m) for d, m in keys])
        ax.set_xlabel(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False, fontsize=11, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    axes[0].set_xlim(max(0, min(float(aps.coverage.min()), float(aps.nominal_coverage.min())) - .05), 1.02)
    axes[1].set_xlim(left=0)
    save(fig, "figure_4_aps_coverage_size", [aps_path], "Raw nonrandomized APS coverage and size under subject-disjoint evaluation. Dashed lines denote nominal coverage, not a guarantee under participant shift. Empty sets are retained.")

    seed_path = "07_analysis/seed_effects.csv"
    seeds = pd.read_csv(ROOT / seed_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("macro_f1", "brier", "nll", "ece_15")):
        values = seeds.loc[seeds.effect.eq("protocol_" + metric), "estimate"].dropna().to_numpy()
        if len(values):
            ax.boxplot(values, vert=False, widths=.35, showfliers=False,
                       boxprops={"color": "#0072B2"}, medianprops={"color": "#D55E00"})
            ax.scatter(values, np.linspace(.92, 1.08, len(values)), s=20, color="#0072B2", zorder=3)
        ax.axvline(0, color="0.6", linewidth=.8, linestyle="--")
        ax.set_yticks([])
        ax.set_title({"macro_f1":"Macro-F1","brier":"Brier score","nll":"NLL","ece_15":"ECE (15 bins)"}[metric] + f" | {len(values)} fixed seeds")
        ax.set_xlabel("Subject-disjoint minus sample-mixed")
        ax.grid(axis="x", alpha=.2)
    save(fig, "figure_5_seed_sensitivity", [seed_path], "Split-seed effect distributions. Each point is an equal-dataset participant estimate for a fixed seed; seeds are sensitivity repeats, not independent participants.")

    participant_path = "07_analysis/participant_metrics.csv"
    if summary.protocol.eq("leakage_reduced_mixed").any():
        participants = pd.read_csv(ROOT / participant_path)
        data = participants[participants.dataset.eq("uci_har") & participants.seed.eq(PRIMARY_SEED) & participants.method.eq("raw")]
        data = data.groupby(["model", "protocol"], observed=True)[["macro_f1", "brier", "nll", "ece_15"]].mean()
        optional_models = sorted(set(data.xs("leakage_reduced_mixed", level="protocol").index))
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        protocols = ("sample_mixed", "leakage_reduced_mixed", "subject_disjoint")
        for ax, metric in zip(axes.flat, ("macro_f1", "brier", "nll", "ece_15")):
            for offset, protocol, color in zip((-.24, 0, .24), protocols, ("#777777", "#009E73", "#0072B2")):
                ax.bar(np.arange(len(optional_models)) + offset, [data.loc[(model, protocol), metric] for model in optional_models], width=.22, color=color, label=protocol.replace("_", " "))
            ax.set_xticks(np.arange(len(optional_models)), [model_names.get(m, m) for m in optional_models], rotation=30)
            ax.set_title("UCI HAR | " + {"macro_f1":"Macro-F1","brier":"Brier score","nll":"NLL","ece_15":"ECE (15 bins)"}[metric])
            ax.grid(axis="y", alpha=.2)
        axes[0, 0].legend(frameon=False, fontsize=11, loc="lower left", bbox_to_anchor=(0, 1.14), ncol=3)
        save(fig, "figure_6_temporal_overlap_supplement", [participant_path], "Primary-seed UCI comparison using exact shared-window overlap components. This reduces verified overlap leakage; it does not reconstruct sessions or guarantee removal of temporal autocorrelation.")
    manifest = {"status": "complete", "n_figures": len(figures), "figures": figures,
                "analysis_sha256": sha256(marker_path), "postprocess_source_sha256": sha256(__file__), "finished_utc": now()}
    atomic_json(out / "figure_manifest.json", manifest)
    return manifest

