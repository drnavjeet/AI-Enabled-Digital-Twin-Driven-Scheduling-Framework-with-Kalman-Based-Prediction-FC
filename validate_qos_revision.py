from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "qos_revision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the QoS/DRL-OO revision results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    args = parse_args()
    results = args.results_dir.resolve()
    failures: list[str] = []
    seeds = 2 if args.quick else 30
    secondary_seeds = 2 if args.quick else 20
    scalability_seeds = 1 if args.quick else 10
    expected = {
        "main_run_kpis.csv": 3 * 3 * 5 * seeds,
        "main_summary.csv": 45,
        "node_counts.csv": 3 * 3 * 5 * seeds * 8,
        "prediction_run_metrics.csv": 3 * 3 * seeds * (5 + 3 * 2),
        "prediction_summary.csv": 3 * 3 * (5 + 3 * 2),
        "paired_statistics.csv": 3 * 3 * 10 * 4,
        "qos_ablation_run_kpis.csv": 4 * secondary_seeds,
        "qos_ablation_summary.csv": 4,
        "sensitivity_run_kpis.csv": 10 * secondary_seeds,
        "sensitivity_summary.csv": 10,
        "scalability_run_kpis.csv": 4 * 5 * scalability_seeds,
        "scalability_summary.csv": 20,
        "training_log.csv": (1 if args.quick else 5) * (4 if args.quick else 200),
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for name, expected_rows in expected.items():
        path = results / name
        require(path.exists(), f"missing {name}", failures)
        if not path.exists():
            continue
        tables[name] = read_csv(path)
        require(
            len(tables[name]) == expected_rows,
            f"{name}: expected {expected_rows} rows, found {len(tables[name])}",
            failures,
        )

    main_rows = tables.get("main_run_kpis.csv", [])
    main_lookup = {
        (int(row["seed"]), row["qos_regime"], row["load"], row["algorithm"]): row
        for row in main_rows
    }
    task_aggregates: dict[tuple[int, str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    task_count = 0
    task_path = results / "raw_task_records.csv.gz"
    require(task_path.exists(), "missing raw_task_records.csv.gz", failures)
    if task_path.exists():
        with gzip.open(task_path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                task_count += 1
                key = (
                    int(row["seed"]),
                    row["qos_regime"],
                    row["load"],
                    row["algorithm"],
                )
                aggregate = task_aggregates[key]
                aggregate["arrivals"] += 1
                aggregate["misses"] += int(row["missed"])
                aggregate["failed"] += int(row["network_failed"])
                aggregate["energy"] += float(row["energy_j"])
                aggregate["cost"] += float(row["monetary_cost"])
                aggregate["retransmission_bytes"] += float(row["retransmission_bytes"])
                if row["completion_s"] and float(row["completion_s"]) <= 110.0:
                    aggregate["completed"] += 1

    require(
        task_count == sum(int(row["arrived_tasks"]) for row in main_rows),
        "raw task count does not equal summed measured arrivals",
        failures,
    )
    for key, row in main_lookup.items():
        aggregate = task_aggregates[key]
        arrivals = aggregate["arrivals"]
        require(close(arrivals, float(row["arrived_tasks"])), f"arrival mismatch for {key}", failures)
        if arrivals:
            require(
                close(100.0 * aggregate["misses"] / arrivals, float(row["dmr_pct"])),
                f"DMR mismatch for {key}",
                failures,
            )
            require(
                close(100.0 * aggregate["failed"] / arrivals, float(row["network_failure_rate_pct"])),
                f"network-failure mismatch for {key}",
                failures,
            )
        require(close(aggregate["completed"] / 90.0, float(row["throughput_tasks_s"])), f"throughput mismatch for {key}", failures)
        require(close(aggregate["energy"], float(row["energy_j"])), f"energy mismatch for {key}", failures)
        require(close(aggregate["cost"], float(row["monetary_cost"])), f"cost mismatch for {key}", failures)
        require(
            close(aggregate["retransmission_bytes"], float(row["retransmission_bytes"])),
            f"retransmission-byte mismatch for {key}",
            failures,
        )

    for row in tables.get("paired_statistics.csv", []):
        require(int(row["n_pairs"]) == seeds, "paired comparison has wrong seed count", failures)
        require(0.0 <= float(row["holm_p"]) <= 1.0, "invalid Holm p-value", failures)

    sensitivity = tables.get("sensitivity_summary.csv", [])
    loss_rows = sorted(
        (row for row in sensitivity if row["sensitivity_type"] == "loss"),
        key=lambda row: float(row["configured_loss_pct"]),
    )
    jitter_rows = sorted(
        (row for row in sensitivity if row["sensitivity_type"] == "jitter"),
        key=lambda row: float(row["configured_jitter_ms"]),
    )
    if loss_rows:
        require(
            float(loss_rows[-1]["retransmission_overhead_pct_mean"])
            > float(loss_rows[0]["retransmission_overhead_pct_mean"]),
            "loss sensitivity does not increase retransmission overhead",
            failures,
        )
    if jitter_rows:
        require(
            float(jitter_rows[-1]["mean_latency_ms_mean"])
            > float(jitter_rows[0]["mean_latency_ms_mean"]),
            "jitter sensitivity does not increase latency",
            failures,
        )

    manifest_path = results / "experiment_manifest.json"
    require(manifest_path.exists(), "missing experiment manifest", failures)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_status = "quick_development_run" if args.quick else "full_reviewer_revision_run"
        require(manifest.get("status") == expected_status, "manifest status mismatch", failures)
        require(
            "synthetic" in str(manifest.get("provenance", "")).lower(),
            "manifest does not disclose synthetic QoS regimes",
            failures,
        )
    model_files = list((results / "models").glob("*.npz"))
    require(len(model_files) == (1 if args.quick else 5), "wrong number of frozen policies", failures)

    validation = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checked_main_runs": len(main_rows),
        "checked_task_records": task_count,
        "checked_paired_comparisons": len(tables.get("paired_statistics.csv", [])),
        "checked_policy_files": len(model_files),
    }
    (results / "validation_result.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
