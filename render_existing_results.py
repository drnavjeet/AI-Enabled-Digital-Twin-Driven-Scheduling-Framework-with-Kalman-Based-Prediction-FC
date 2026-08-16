from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from run_independent_experiments import render_latex


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "independent_experiment"
TEXT_FIELDS = {
    "load",
    "algorithm",
    "metric",
    "comparator",
    "normality_test",
    "test",
    "effect_name",
}


def read_rows(name: str) -> list[dict[str, Any]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for source in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if key in TEXT_FIELDS:
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
        return rows


def main() -> None:
    content = render_latex(
        read_rows("main_summary.csv"),
        read_rows("prediction_summary.csv"),
        read_rows("paired_statistics.csv"),
        read_rows("ablation_summary.csv"),
        read_rows("sensitivity_summary.csv"),
        read_rows("scalability_summary.csv"),
    )
    (RESULTS / "independent_results_tables.tex").write_text(content, encoding="utf-8")
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    (RESULTS / "checkpoint.json").write_text(
        json.dumps(
            {
                "completed_through": "all",
                "main_seeds": config["main_seeds"],
                "secondary_seeds": config["secondary_seeds"],
                "scalability_seeds": config["scalability_seeds"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"rendered {RESULTS / 'independent_results_tables.tex'}")


if __name__ == "__main__":
    main()
