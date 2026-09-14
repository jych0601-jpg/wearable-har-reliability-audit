"""Trace declared report claims back to processed CSV values and rendered text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _matches(series: pd.Series, expected: Any) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and isinstance(expected, (int, float)):
        return np.isclose(series.to_numpy(dtype=float), float(expected), rtol=0.0, atol=1e-12)
    return series.astype(str).eq(str(expected)).to_numpy()


def audit_claims(spec_path: str | Path, project_root: str | Path = ".") -> dict[str, Any]:
    """Audit each registered numerical value and each required textual rendering."""
    root = Path(project_root).resolve()
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for claim in spec["claims"]:
        source = root / claim["source"]
        result: dict[str, Any] = {
            "id": claim["id"],
            "source": claim["source"],
            "column": claim["column"],
            "expected": claim["expected"],
            "observed": None,
            "row_count": 0,
            "source_value_matches": False,
            "evidence": [],
            "passed": False,
        }
        try:
            frame = pd.read_csv(source)
            selected = frame
            for column, value in claim.get("filters", {}).items():
                selected = selected[_matches(selected[column], value)]
            result["row_count"] = int(len(selected))
            aggregate = claim.get("aggregate")
            if aggregate is None and len(selected) == 1:
                observed = float(selected.iloc[0][claim["column"]])
            elif aggregate in {"mean", "min", "max"} and len(selected) > 0:
                values = selected[claim["column"]].to_numpy(dtype=float)
                observed = float(getattr(np, aggregate)(values))
            else:
                observed = None
            if observed is not None:
                result["observed"] = observed
                result["source_value_matches"] = bool(
                    np.isclose(
                        observed,
                        float(claim["expected"]),
                        rtol=0.0,
                        atol=float(claim.get("atol", 1e-12)),
                        equal_nan=False,
                    )
                )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            result["source_error"] = f"{type(exc).__name__}: {exc}"

        evidence_passed = True
        for evidence in claim.get("evidence", []):
            document = root / evidence["file"]
            try:
                content = document.read_text(encoding="utf-8")
                found = evidence["text"] in content
                entry = {**evidence, "found": bool(found)}
            except (FileNotFoundError, UnicodeDecodeError) as exc:
                entry = {
                    **evidence,
                    "found": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result["evidence"].append(entry)
            evidence_passed = evidence_passed and entry["found"]

        result["passed"] = bool(result["source_value_matches"] and evidence_passed)
        results.append(result)

    required_results: list[dict[str, Any]] = []
    for requirement in spec.get("required_files", []):
        path = root / requirement["file"]
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        minimum = int(requirement.get("minimum_bytes", 1))
        required_results.append(
            {
                "file": requirement["file"],
                "exists": exists,
                "size_bytes": int(size),
                "minimum_bytes": minimum,
                "passed": bool(exists and size >= minimum),
            }
        )

    passed = all(item["passed"] for item in results) and all(
        item["passed"] for item in required_results
    )
    return {
        "passed": bool(passed),
        "spec": str(spec_path),
        "n_claims": len(results),
        "n_required_files": len(required_results),
        "claims": results,
        "required_files": required_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = audit_claims(arguments.spec, arguments.project_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
