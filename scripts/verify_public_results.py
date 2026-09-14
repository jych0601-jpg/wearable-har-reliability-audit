from pathlib import Path
import csv
import math

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    with path.open(encoding="utf-8", newline="") as f:
        return {row["effect"]: row for row in csv.DictReader(f)}

primary = load(ROOT / "results" / "table_primary_estimands.csv")
expanded = load(ROOT / "results" / "table_expanded_estimands.csv")

expected_primary = {
    "protocol_macro_f1": -0.10916527163266188,
    "protocol_brier": 0.13800600194200185,
    "calibration_temp_brier": 0.0077170523859251515,
    "selective80_temp": -0.05636804427709684,
}
expected_expanded = {
    "protocol_macro_f1": -0.1269923627463203,
    "protocol_brier": 0.15719442508901454,
    "protocol_nll": 0.5459229583976536,
    "protocol_ece_15": 0.0423866168010332,
    "calibration_iso_brier": -0.010265818386272652,
    "calibration_temp_brier": 0.01096589859604181,
    "selective80_raw": -0.05645250666364812,
    "aps_a0.1_mean_size": 5.065283797581537,
    "aps_a0.2_mean_size": 3.556351002953366,
}

for key, expected in expected_primary.items():
    actual = float(primary[key]["estimate"])
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (key, actual, expected)

for key, expected in expected_expanded.items():
    actual = float(expanded[key]["estimate"])
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (key, actual, expected)

print("Public result checks passed.")
