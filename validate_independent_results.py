from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiment import jain_fairness


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "independent_experiment"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def percentile(values: Iterable[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=float), quantile))


def main() -> None:
    failures: list[str] = []
    expected_rows = {
        "main_run_kpis.csv": 360,
        "main_summary.csv": 12,
        "node_counts.csv": 2880,
        "prediction_run_metrics.csv": 360,
        "prediction_summary.csv": 12,
        "paired_statistics.csv": 72,
        "ablation_run_kpis.csv": 420,
        "ablation_summary.csv": 21,
        "sensitivity_run_kpis.csv": 260,
        "sensitivity_summary.csv": 13,
        "scalability_run_kpis.csv": 160,
        "scalability_summary.csv": 16,
    }
    tables = {name: read_csv(RESULTS / name) for name in expected_rows}
    for name, expected in expected_rows.items():
        require(len(tables[name]) == expected, f"{name}: expected {expected} rows", failures)

    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "full_independent_rerun", "manifest is not full", failures)
    require(manifest["config"]["main_seeds"] == 30, "main seed count is not 30", failures)
    require(
        manifest["config"]["secondary_seeds"] == 20,
        "secondary seed count is not 20",
        failures,
    )

    main_rows = tables["main_run_kpis.csv"]
    main_lookup = {
        (row["load"], row["algorithm"], int(row["seed"])): row for row in main_rows
    }
    require(len(main_lookup) == len(main_rows), "duplicate main run key", failures)
    for row in main_rows:
        fairness = float(row["fairness"])
        require(0.0 <= fairness <= 1.0, "fairness outside [0,1]", failures)
        objective = float(row["run_cost_index"])
        require(0.0 <= objective <= 1.0, "run objective outside [0,1]", failures)

    node_groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in tables["node_counts.csv"]:
        key = (row["load"], row["algorithm"], int(row["seed"]))
        node_groups[key].append(int(row["completed_tasks"]))
    for key, counts in node_groups.items():
        require(len(counts) == 8, f"{key}: expected 8 fog-node counts", failures)
        require(
            close(jain_fairness(counts), float(main_lookup[key]["fairness"])),
            f"{key}: fairness does not reconcile",
            failures,
        )

    task_rows = read_gzip_csv(RESULTS / "raw_task_records.csv.gz")
    task_groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in task_rows:
        task_groups[(row["load"], row["algorithm"], int(row["seed"]))].append(row)
    require(len(task_groups) == 360, "raw task records do not cover 360 runs", failures)
    for key, rows in task_groups.items():
        reported = main_lookup[key]
        accepted = [row for row in rows if int(row["rejected"]) == 0]
        latencies = [
            1000.0 * (float(row["completion_s"]) - float(row["arrival_s"]))
            for row in accepted
        ]
        completed = sum(float(row["completion_s"]) <= 110.0 for row in accepted)
        recomputed = {
            "arrived_tasks": len(rows),
            "completed_tasks": completed,
            "mean_latency_ms": statistics_mean(latencies),
            "p95_latency_ms": percentile(latencies, 95.0),
            "dmr_pct": 100.0 * sum(int(row["missed"]) for row in rows) / len(rows),
            "throughput_tasks_s": completed / 90.0,
            "energy_j": sum(float(row["energy_j"]) for row in rows),
            "monetary_cost": sum(float(row["monetary_cost"]) for row in rows),
            "rejection_rate_pct": 100.0 * sum(int(row["rejected"]) for row in rows) / len(rows),
        }
        for metric, value in recomputed.items():
            require(
                close(float(reported[metric]), float(value), tolerance=1e-7),
                f"{key}: task-derived {metric} does not reconcile",
                failures,
            )

    prediction_rows = read_gzip_csv(RESULTS / "raw_prediction_records.csv.gz")
    prediction_groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        prediction_groups[(row["load"], row["algorithm"], int(row["seed"]))].append(row)
    require(len(prediction_groups) == 360, "raw predictions do not cover 360 runs", failures)
    prediction_lookup = {
        (row["load"], row["algorithm"], int(row["seed"])): row
        for row in tables["prediction_run_metrics.csv"]
    }
    for key, rows in prediction_groups.items():
        actual = np.asarray([float(row["y_true"]) for row in rows])
        predicted = np.asarray([float(row["y_pred"]) for row in rows])
        error = actual - predicted
        recomputed = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mape": float(np.mean(np.abs(error) / (np.abs(actual) + 1e-6))),
        }
        for metric, value in recomputed.items():
            require(
                close(float(prediction_lookup[key][metric]), value),
                f"{key}: prediction {metric} does not reconcile",
                failures,
            )

    for row in tables["paired_statistics.csv"]:
        require(int(row["n_pairs"]) == 30, "paired comparison does not use 30 seeds", failures)
        require(0.0 <= float(row["raw_p"]) <= 1.0, "raw p-value outside [0,1]", failures)
        require(0.0 <= float(row["holm_p"]) <= 1.0, "Holm p-value outside [0,1]", failures)

    status: dict[str, Any] = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "validated_files": len(expected_rows) + 4,
        "main_runs": len(main_rows),
        "raw_task_records": len(task_rows),
        "raw_prediction_records": len(prediction_rows),
        "paired_comparisons": len(tables["paired_statistics.csv"]),
    }
    (RESULTS / "validation_result.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


def statistics_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


if __name__ == "__main__":
    main()
