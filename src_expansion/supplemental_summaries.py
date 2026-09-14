
"""Descriptive supplements from completed frozen artifacts; no new inference."""
from pathlib import Path
from collections import defaultdict
import itertools
import json
import uuid
import numpy as np
import pandas as pd
import common
import reliability
from common import ROOT, sha256, atomic_json, now
from reliability import validate_probabilities

METHODS = ("raw", "temp", "iso")
BIN_EDGES = np.arange(21, dtype=float) / 20
GROUP = ["dataset", "model", "protocol"]
COVERAGE_KEYS = GROUP + ["representation", "method", "alpha"]
METRIC_DIRECTIONS = {"accuracy": True, "macro_f1": True, "brier": False, "nll": False, "ece_15": False}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _expected_seeds(protocol, config):
    if protocol in config["protocols"]:
        return set(map(int, config["seeds"]))
    if protocol == "leakage_reduced_mixed":
        return {int(config.get("statistical_seed", config["seeds"][0]))}
    raise ValueError("Unknown protocol in descriptive input: " + str(protocol))


def coverage_variability(frame, config):
    """Describe participant seed means; windows and seeds are not independent persons."""
    required = COVERAGE_KEYS + ["seed", "subject", "coverage", "nominal_coverage"]
    if not set(required).issubset(frame):
        raise ValueError("Coverage input is missing required columns")
    if frame[required].isna().any().any():
        raise ValueError("Coverage input contains missing values")
    if frame.duplicated(COVERAGE_KEYS + ["subject", "seed"]).any():
        raise ValueError("Duplicate participant-seed coverage records")
    coverage = frame.coverage.to_numpy(float)
    if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("Coverage must be finite and within [0,1]")
    if not set(frame.alpha).issubset(set(config["alpha"])):
        raise ValueError("Unexpected alpha")
    if not np.allclose(frame.nominal_coverage, 1 - frame.alpha, atol=1e-12, rtol=0):
        raise ValueError("Nominal coverage does not match alpha")
    if not frame.method.eq("raw").all():
        raise ValueError("This supplement expects the raw-only conformal protocol")
    rows = []
    for identity, group in frame.groupby(COVERAGE_KEYS, sort=True, observed=True):
        record = dict(zip(COVERAGE_KEYS, identity))
        expected = _expected_seeds(record["protocol"], config)
        participant = []
        for subject, repeated in group.groupby("subject", sort=True, observed=True):
            if set(map(int, repeated.seed)) != expected:
                raise ValueError("Missing or unexpected seed for participant coverage")
            participant.append(float(repeated.coverage.mean()))
        values = np.asarray(participant)
        nominal = 1 - float(record["alpha"])
        record.update(n_people=len(values), n_seeds_expected=len(expected),
            n_seeds_available_min=len(expected), n_seeds_available_max=len(expected),
            nominal_coverage=nominal, sd_reason="defined" if len(values) > 1 else "fewer_than_two_participants",
            summary_unit="participant after arithmetic averaging across repeated seeds",
            inferential=False)
        stats = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            min=float(values.min()), q25=float(np.quantile(values, .25)),
            median=float(np.median(values)), q75=float(np.quantile(values, .75)), max=float(values.max()))
        for name, value in stats.items():
            record["coverage_" + name] = value
            record["coverage_gap_" + name] = value if name == "sd" else value - nominal
        rows.append(record)
    if not rows:
        raise ValueError("Coverage input is empty")
    return pd.DataFrame(rows)


def confidence_histogram(probabilities):
    """Fixed 20 bins: left closed/right open, final bin includes exactly 1."""
    p = validate_probabilities(probabilities)
    confidence = p.max(axis=1)
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence outside [0,1]")
    counts, _ = np.histogram(confidence, bins=BIN_EDGES)
    if int(counts.sum()) != len(p):
        raise ValueError("Histogram failed to retain every test window")
    return counts


def model_ranks(summary, config):
    """Within-dataset/protocol raw ranks with average ranks for exact ties."""
    required = GROUP + ["method", "representation", "n_subjects_total"] + list(METRIC_DIRECTIONS)
    if not set(required).issubset(summary):
        raise ValueError("Model summary is missing required columns")
    raw = summary[summary.method.eq("raw")].copy()
    if raw.empty or raw.duplicated(GROUP).any():
        raise ValueError("Missing or duplicate raw model summaries")
    if not np.isfinite(raw[list(METRIC_DIRECTIONS)].to_numpy(float)).all():
        raise ValueError("Model rank metric is undefined")
    rows = []
    for (dataset, protocol), group in raw.groupby(["dataset", "protocol"], sort=True, observed=True):
        if set(group.model) != set(config["models"]):
            raise ValueError("Incomplete model family for descriptive rankings")
        seeds = _expected_seeds(protocol, config)
        for metric, higher in METRIC_DIRECTIONS.items():
            ranks = group[metric].rank(ascending=not higher, method="average")
            for index, item in group.iterrows():
                rows.append(dict(dataset=dataset, protocol=protocol, model=item.model,
                    representation=item.representation, method="raw", metric=metric,
                    value=float(item[metric]), rank=float(ranks.loc[index]), higher_is_better=higher,
                    n_models=len(group), n_seeds_expected=len(seeds), n_subjects_total=int(item.n_subjects_total),
                    n_subjects_available=int(item.get("n_subjects_available_" + metric, item.n_subjects_total)),
                    tie_rule="average rank for exact metric ties",
                    aggregation="seed mean within participant, followed by equal-participant mean",
                    representation_caution="WISDM CNN raw windows and classical official feature windows differ" if dataset == "wisdm_watch_accel" else "Representations reported explicitly",
                    inferential=False, source_csv="09_tables/table_model_summary.csv"))
    return pd.DataFrame(rows)


def _verified_inputs():
    config_path = ROOT / "configs/expanded.json"
    analysis_path = ROOT / "07_analysis/completed.json"
    if not analysis_path.is_file():
        raise RuntimeError("Completed analysis marker is required")
    config = _json(config_path)
    analysis = _json(analysis_path)
    if analysis.get("status") != "complete":
        raise RuntimeError("Analysis grid is incomplete")
    if analysis.get("config_sha256") != sha256(config_path):
        raise RuntimeError("Frozen config hash changed")
    sources = {str(config_path.relative_to(ROOT)): sha256(config_path),
               str(analysis_path.relative_to(ROOT)): sha256(analysis_path)}
    required_csvs = ["07_analysis/participant_conformal.csv", "09_tables/table_model_summary.csv"]
    declared = {str(Path(name)): digest for name, digest in analysis["artifact_hashes"].items()}
    for name in required_csvs:
        p = ROOT/name
        digest = sha256(p)
        if declared.get(str(Path(name))) != digest:
            raise RuntimeError("Analysis input hash changed: " + name)
        sources[str(Path(name))] = digest
    expected = set(itertools.product(config["datasets"], config["models"], config["protocols"],
        map(int, config["seeds"]), range(int(config["n_folds"]))))
    if analysis.get("expected_base_fits") != len(expected) or analysis.get("completed_base_fits") != len(expected):
        raise RuntimeError("Analysis base fit grid is incomplete")
    records = []
    observed = set()
    for marker in sorted((ROOT/"04_calibration").glob("*/completed.json")):
        completion = _json(marker)
        if completion.get("status") != "complete":
            raise RuntimeError("Calibration fit is incomplete")
        out = marker.parent
        artifacts = {entry["path"]: entry["sha256"] for entry in completion["artifacts"]}
        sources[str(marker.relative_to(ROOT))] = sha256(marker)
        for name in ["methods.json", "test_probabilities.npz"]:
            p = out/name
            digest = sha256(p)
            if artifacts.get(name) != digest:
                raise RuntimeError("Calibration input hash changed: " + str(p))
            sources[str(p.relative_to(ROOT))] = digest
        identity = _json(out/"methods.json")["identity"]
        key = tuple(identity[k] for k in GROUP + ["seed", "fold"])
        if key in observed:
            raise RuntimeError("Duplicate calibration fit in grid")
        observed.add(key)
        records.append((out, identity))
    extras = observed - expected
    if not expected.issubset(observed):
        raise RuntimeError("Missing intended calibration fit grid")
    for dataset, model, protocol, seed, fold in extras:
        if (dataset != "uci_har" or model not in config["models"] or protocol != "leakage_reduced_mixed"
            or seed != int(config.get("statistical_seed", config["seeds"][0]))
            or fold not in range(int(config["n_folds"]))):
            raise RuntimeError("Unexpected optional calibration fit")
    if len(extras) != analysis.get("optional_fits", 0):
        raise RuntimeError("Optional calibration fit count differs from completed analysis")
    for model in {key[1] for key in extras}:
        if len([key for key in extras if key[1] == model]) != int(config["n_folds"]):
            raise RuntimeError("Incomplete optional calibration fit folds")
    code = {"source/" + Path(path).name: sha256(path) for path in
        [__file__, common.__file__, reliability.__file__]}
    return config, analysis, records, {"inputs": sources, "source_code": code}


def _histogram_table(records):
    counts = {}
    representations = {}
    for out, identity in records:
        with np.load(out/"test_probabilities.npz", allow_pickle=False) as data:
            sizes = []
            for method in METHODS:
                key = tuple(identity[k] for k in GROUP + ["seed"]) + (method,)
                p = data[method]
                sizes.append(len(p))
                current = confidence_histogram(p)
                counts[key] = counts.get(key, np.zeros(20, dtype=np.int64)) + current
                rep = identity["representation"]
                if key in representations and representations[key] != rep:
                    raise ValueError("Representation changed across folds")
                representations[key] = rep
            if len(set(sizes)) != 1:
                raise ValueError("Probability methods have different test row counts")
    rows = []
    for key, count in sorted(counts.items()):
        n = int(count.sum())
        for i in range(20):
            rows.append(dict(zip(GROUP + ["seed", "method"], key),
                representation=representations[key], bin_index=i, bin_lower=float(BIN_EDGES[i]),
                bin_upper=float(BIN_EDGES[i+1]), upper_inclusive=i == 19,
                n_windows=n, count=int(count[i]), proportion=float(count[i]/n),
                summary_unit="test window; folds concatenated within seed", inferential=False,
                source_pattern="04_calibration/<fit>/test_probabilities.npz"))
    return pd.DataFrame(rows)


def _write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run():
    """Write or verify three source-bound descriptive outputs after final analysis."""
    config, analysis, records, binding = _verified_inputs()
    marker = ROOT/"07_analysis/supplemental_summary_manifest.json"
    if marker.exists():
        previous = _json(marker)
        if previous["binding"] != binding:
            raise RuntimeError("Supplemental inputs or source code changed; preserved outputs are stale")
        for entry in previous["artifacts"]:
            if not (ROOT/entry["path"]).is_file() or sha256(ROOT/entry["path"]) != entry["sha256"]:
                raise RuntimeError("Supplemental output hash changed: " + entry["path"])
        return {**previous["summary"], "resumed": True}
    coverage = coverage_variability(pd.read_csv(ROOT/"07_analysis/participant_conformal.csv"), config)
    ranks = model_ranks(pd.read_csv(ROOT/"09_tables/table_model_summary.csv"), config)
    histograms = _histogram_table(records)
    outputs = {
        "07_analysis/participant_coverage_variability.csv": coverage,
        "09_tables/table_confidence_distribution.csv": histograms,
        "09_tables/table_expanded_model_ranks.csv": ranks,
    }
    for name, frame in outputs.items():
        _write_csv(ROOT/name, frame)
    notes = ROOT/"09_tables/supplemental_summary_notes.md"
    notes.write_text(
        "# Descriptive supplemental summaries\n\n"
        "All summaries use the completed frozen fit grid. No models were refit, thresholds tuned, "
        "p-values added, or observations excluded.\n\n"
        "- Participant coverage variability: average each participant's coverage across the repeated "
        "seeds first, then compute equal-participant mean, sample SD (ddof=1), minimum, quartiles, "
        "median and maximum. Coverage gaps subtract the nominal level. A single-participant SD "
        "would be undefined and explicitly identified. The optional UCI overlap protocol has one "
        "primary seed; the base protocols have ten seeds in the production configuration.\n"
        "- Confidence distributions: maximum class probability for every saved test window, with "
        "fold counts pooled within dataset/model/protocol/seed/method. The fixed 20 bins are "
        "[i/20,(i+1)/20), except the final bin includes 1. Empty bins are retained. These window "
        "histograms are descriptive and are not participant-level inferential samples.\n"
        "- Model ranks: use the existing raw equal-participant model summary after within-participant "
        "seed averaging. Higher accuracy and macro-F1, and lower Brier/NLL/ECE, rank better. Exact "
        "ties receive average ranks. Ranks are descriptive. WISDM CNN raw-window and classical "
        "official-feature representations differ, so rankings do not isolate architecture alone.\n\n"
        "Inputs: 07_analysis/participant_conformal.csv, 09_tables/table_model_summary.csv and "
        "04_calibration/<fit>/test_probabilities.npz. The manifest in "
        "07_analysis/supplemental_summary_manifest.json binds every consumed artifact and source "
        "module by SHA-256. Existing result CSVs remain unchanged.\n",
        encoding="utf8")
    names = list(outputs) + [str(notes.relative_to(ROOT))]
    summary = dict(status="complete", completed_fits=len(records),
        completed_base_fits=analysis["completed_base_fits"], optional_fits=analysis.get("optional_fits",0),
        coverage_rows=len(coverage), histogram_rows=len(histograms), rank_rows=len(ranks),
        no_added_inference=True)
    atomic_json(marker, dict(schema=1, status="complete", finished_utc=now(), binding=binding,
        summary=summary, artifacts=[dict(path=name, sha256=sha256(ROOT/name)) for name in names]))
    return {**summary, "resumed": False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

