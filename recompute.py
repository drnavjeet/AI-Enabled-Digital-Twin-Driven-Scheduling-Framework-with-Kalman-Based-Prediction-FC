from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dt_kf import (
    LinkState,
    ScalarKalman,
    SchedulerConfig,
    Task,
    VenueState,
    ema_predictions,
    error_metrics,
    last_value_predictions,
    select_venue,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "work" / "data_sources"
DEFAULT_IFOGSIM_ROOT = Path(r"C:\Users\Dell\Downloads\CSV\ifogsim-task-scheduler-agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the paper results and run an independent trace-backed predictor benchmark."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ifogsim-root", type=Path, default=DEFAULT_IFOGSIM_ROOT)
    parser.add_argument("--config", type=Path, default=HERE / "example_config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def count_lines(path: Path) -> int:
    count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return count + int(bool(last_byte) and last_byte != b"\n")


def dataset_inventory(data_root: Path, ifogsim_root: Path) -> list[dict[str, Any]]:
    edge_root = data_root / "edge-computing-dataset"
    eua_root = data_root / "eua-dataset"
    dc_root = data_root / "datacenter-traces-datasets"
    edge_files = sorted((edge_root / "Data").rglob("*.csv"))
    edge_rows = sum(count_lines(path) for path in edge_files)

    users_path = eua_root / "users" / "users-melbcbd-generated.csv"
    servers_path = eua_root / "edge-servers" / "site-optus-melbCBD.csv"
    dc_path = dc_root / "alibaba2018" / "machine_usage_days_1_to_8_grouped_10_seconds.csv"
    java_files = list((ifogsim_root / "src").rglob("*.java")) if ifogsim_root.exists() else []
    dc_records = max(count_lines(dc_path) - 1, 0) if dc_path.exists() else 0
    expected_dc_records = 8 * 24 * 60 * 60 // 10

    return [
        {
            "input": "Dataset A",
            "source": "BuptMecMigration/Edge-Computing-Dataset",
            "availability": "downloaded_public_repository" if edge_files else "missing",
            "version": git_commit(edge_root),
            "records": edge_rows,
            "detail": f"{len(edge_files)} CSV files; no header; 22 documented fields",
        },
        {
            "input": "Dataset B",
            "source": "PhuLai/eua-dataset",
            "availability": "downloaded_public_repository" if users_path.exists() else "missing",
            "version": git_commit(eua_root),
            "records": max(count_lines(users_path) - 1, 0) if users_path.exists() else 0,
            "detail": (
                f"Melbourne CBD users; {max(count_lines(servers_path) - 1, 0)} Optus server sites"
                if servers_path.exists()
                else "server CSV missing"
            ),
        },
        {
            "input": "Dataset C",
            "source": "DataCenter-Traces-Datasets / Alibaba 2018",
            "availability": "downloaded_public_repository_and_pypi_package_available"
            if dc_path.exists()
            else "missing",
            "version": git_commit(dc_root),
            "records": dc_records,
            "detail": (
                "10-second aggregate CPU, memory, network-in/out, and disk utilization; "
                f"{expected_dc_records - dc_records} fewer rows than a complete eight-day series"
            ),
        },
        {
            "input": "Dataset D",
            "source": "Held-out final day of Dataset C (this independent benchmark)",
            "availability": "derived_locally" if dc_path.exists() else "missing",
            "version": git_commit(dc_root),
            "records": 8640 if dc_path.exists() else 0,
            "detail": "Explicit choice made here; the paper does not identify its Dataset D",
        },
        {
            "input": "Simulator",
            "source": "iFogSim2 local checkout",
            "availability": "available_locally" if java_files else "missing",
            "version": git_commit(ifogsim_root) if java_files else "unknown",
            "records": len(java_files),
            "detail": "Java source files in the existing local project",
        },
    ]


def load_trace(path: Path, signals: list[str]) -> np.ndarray:
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    columns = []
    for signal in signals:
        if signal not in (raw.dtype.names or ()):
            raise ValueError(f"Signal {signal!r} is not present in {path}")
        values = np.asarray(raw[signal], dtype=float)
        values[(values < 0) | (values > 100)] = np.nan
        valid = np.isfinite(values)
        if np.sum(valid) < 2:
            raise ValueError(f"Signal {signal!r} has too few valid observations")
        indices = np.arange(len(values))
        values = np.interp(indices, indices[valid], values[valid]) / 100.0
        columns.append(values)
    return np.column_stack(columns)


def tune_predictors(
    validation: np.ndarray,
    signals: list[str],
    q_grid: Iterable[float],
    r_grid: Iterable[float],
    alpha_grid: Iterable[float],
    epsilon: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    kalman: dict[str, tuple[float, float]] = {}
    ema: dict[str, float] = {}
    for column, signal in enumerate(signals):
        values = validation[:, column]
        best_kalman = min(
            (
                (
                    error_metrics(
                        values,
                        ScalarKalman(float(q), float(r)).one_step_predictions(values),
                        epsilon,
                    )["mae"],
                    float(q),
                    float(r),
                )
                for q in q_grid
                for r in r_grid
            ),
            key=lambda item: item[0],
        )
        best_ema = min(
            (
                (
                    error_metrics(values, ema_predictions(values, float(alpha)), epsilon)["mae"],
                    float(alpha),
                )
                for alpha in alpha_grid
            ),
            key=lambda item: item[0],
        )
        kalman[signal] = (best_kalman[1], best_kalman[2])
        ema[signal] = best_ema[1]
    return kalman, ema


def t_critical_975(degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        return math.nan
    z = 1.959963984540054
    df = float(degrees_freedom)
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def summarize(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan, math.nan
    standard_deviation = statistics.stdev(values)
    half_width = t_critical_975(len(values) - 1) * standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, half_width


def predictor_benchmark(
    trace: np.ndarray, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    signals = list(config["signals"])
    segment_length = int(config["segment_length"])
    epsilon = float(config["epsilon"])
    samples_per_day = 24 * 60 * 60 // 10
    if len(trace) < 2 * samples_per_day:
        raise ValueError("The benchmark needs at least two days of 10-second trace data")
    validation = trace[-2 * samples_per_day : -samples_per_day]
    held_out = trace[-samples_per_day:]
    kalman_params, ema_params = tune_predictors(
        validation,
        signals,
        config["kalman_q_grid"],
        config["kalman_r_grid"],
        config["ema_alpha_grid"],
        epsilon,
    )

    segment_count = len(held_out) // segment_length
    held_out = held_out[: segment_count * segment_length]
    segments = held_out.reshape(segment_count, segment_length, len(signals))
    segment_load = np.mean(segments, axis=(1, 2))
    low_boundary, high_boundary = np.quantile(segment_load, [1 / 3, 2 / 3])

    detailed: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        load = (
            "low"
            if segment_load[segment_index] <= low_boundary
            else "medium"
            if segment_load[segment_index] <= high_boundary
            else "high"
        )
        for signal_index, signal in enumerate(signals):
            values = segment[:, signal_index]
            q, r = kalman_params[signal]
            predictions = {
                "kalman": ScalarKalman(q, r).one_step_predictions(values),
                "ema": ema_predictions(values, ema_params[signal]),
                "last_value": last_value_predictions(values),
            }
            for model, predicted in predictions.items():
                metrics = error_metrics(values, predicted, epsilon)
                detailed.append(
                    {
                        "segment": segment_index,
                        "load": load,
                        "signal": signal,
                        "model": model,
                        **metrics,
                    }
                )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[(row["load"], row["model"], row["signal"])].append(row)

    by_segment: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        by_segment[(row["load"], row["model"], row["segment"])].append(row)
    for (load, model, segment), rows in by_segment.items():
        grouped[(load, model, "aggregate")].append(
            {
                "segment": segment,
                "mae": statistics.fmean(row["mae"] for row in rows),
                "rmse": statistics.fmean(row["rmse"] for row in rows),
                "mape": statistics.fmean(row["mape"] for row in rows),
            }
        )

    summary: list[dict[str, Any]] = []
    load_order = {"low": 0, "medium": 1, "high": 2}
    model_order = {"kalman": 0, "ema": 1, "last_value": 2}
    for (load, model, signal), rows in sorted(
        grouped.items(),
        key=lambda item: (
            load_order[item[0][0]],
            model_order[item[0][1]],
            0 if item[0][2] == "aggregate" else 1,
            item[0][2],
        ),
    ):
        output: dict[str, Any] = {
            "load": load,
            "model": model,
            "signal": signal,
            "segments": len(rows),
            "observations_per_segment": segment_length - 1,
        }
        for metric in ("mae", "rmse", "mape"):
            mean, sd, ci = summarize([float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_sd"] = sd
            output[f"{metric}_ci95"] = ci
        summary.append(output)

    tuning = {
        "held_out_definition": "final day of Alibaba 2018 10-second aggregate trace",
        "validation_definition": "penultimate day of the same trace",
        "segment_length_observations": segment_length,
        "segment_length_seconds": segment_length * 10,
        "load_boundaries": {"low_upper": float(low_boundary), "medium_upper": float(high_boundary)},
        "kalman": {signal: {"q": q, "r": r} for signal, (q, r) in kalman_params.items()},
        "ema": {signal: {"alpha": alpha} for signal, alpha in ema_params.items()},
    }
    return detailed, summary, tuning


TABLE_IV = {
    "low": {
        "DT-KF-CostAware": [103.19, 189.02, 6.03, 52.62, 4713.0, 0.83],
        "DT-OPT": [112.80, 204.38, 6.55, 51.49, 4903.0, 0.92],
        "SemiGreedy": [113.02, 214.93, 7.32, 50.16, 4906.0, 0.98],
        "Fog-only": [122.03, 220.72, 7.88, 50.39, 4855.0, 1.00],
    },
    "medium": {
        "DT-KF-CostAware": [151.97, 283.55, 7.87, 104.36, 6094.0, 0.96],
        "DT-OPT": [160.93, 299.58, 10.66, 102.55, 6748.0, 1.08],
        "SemiGreedy": [176.93, 324.77, 15.17, 99.96, 6894.0, 1.17],
        "Fog-only": [176.87, 340.05, 18.34, 99.30, 7009.0, 1.22],
    },
    "high": {
        "DT-KF-CostAware": [196.63, 389.28, 7.17, 157.95, 8158.0, 1.22],
        "DT-OPT": [220.54, 441.60, 12.73, 153.40, 8398.0, 1.35],
        "SemiGreedy": [249.08, 492.58, 29.26, 147.59, 8766.0, 1.44],
        "Fog-only": [258.63, 530.04, 35.06, 148.72, 8995.0, 1.51],
    },
}


def result_audit() -> list[dict[str, Any]]:
    metrics = ["mean_latency", "p95_latency", "dmr", "throughput", "energy", "cost_index"]
    claimed_vs_fog = {
        "mean_latency": [15.4, 14.1, 24.0],
        "p95_latency": [14.4, 16.6, 26.6],
        "dmr": [23.5, 57.1, 79.5],
        "throughput": [4.4, 5.1, 6.2],
        "energy": [2.9, 13.1, 9.3],
        "cost_index": [17.0, 21.3, 19.2],
    }
    rows: list[dict[str, Any]] = []
    for load_index, load in enumerate(("low", "medium", "high")):
        proposed = TABLE_IV[load]["DT-KF-CostAware"]
        fog_only = TABLE_IV[load]["Fog-only"]
        for metric_index, metric in enumerate(metrics):
            if metric == "throughput":
                recomputed = 100.0 * (proposed[metric_index] - fog_only[metric_index]) / fog_only[metric_index]
            else:
                recomputed = 100.0 * (fog_only[metric_index] - proposed[metric_index]) / fog_only[metric_index]
            claimed = claimed_vs_fog[metric][load_index]
            rows.append(
                {
                    "section": "Table IV point-estimate claim",
                    "load": load,
                    "metric": metric,
                    "reported_or_claimed": claimed,
                    "recomputed": round(recomputed, 4),
                    "status": "verified_arithmetic" if abs(recomputed - claimed) <= 0.15 else "discrepancy",
                    "severity": "low",
                    "evidence_needed": "Run-level records are still required to validate the underlying point estimates and uncertainty.",
                }
            )

    rows.extend(
        [
            {
                "section": "Table IV/XI",
                "load": "all",
                "metric": "95% confidence intervals",
                "reported_or_claimed": "mean +/- 95% CI",
                "recomputed": "not possible",
                "status": "missing_from_table",
                "severity": "high",
                "evidence_needed": "Per-seed KPI records for every scheduler and load.",
            },
            {
                "section": "Table IV/XI",
                "load": "medium/high",
                "metric": "Cost J (normalized)",
                "reported_or_claimed": "> 1",
                "recomputed": "not possible",
                "status": "definition_conflicts_with_equation_27",
                "severity": "high",
                "evidence_needed": "Aggregation window, denominator, objective weights, and separate monetary cost.",
            },
            {
                "section": "Table IV",
                "load": "all",
                "metric": "Jain fairness baselines",
                "reported_or_claimed": "recompute",
                "recomputed": "not possible",
                "status": "raw_counts_missing",
                "severity": "medium",
                "evidence_needed": "Completed-task count per fog node and per seed.",
            },
            {
                "section": "Equation 14",
                "load": "all",
                "metric": "communication time units",
                "reported_or_claimed": "bytes / bits-per-second",
                "recomputed": "requires 8 * bytes / bits-per-second",
                "status": "unit_error_in_equation",
                "severity": "high",
                "evidence_needed": "Confirm whether simulator bandwidth is bits/s or bytes/s; use one convention everywhere.",
            },
            {
                "section": "Table V/XII",
                "load": "high",
                "metric": "duplicated table conflict",
                "reported_or_claimed": "Table V says recompute; Table XII prints invalid values",
                "recomputed": "not possible",
                "status": "internally_contradictory_results",
                "severity": "high",
                "evidence_needed": "Delete or regenerate Table XII from the same verified run-level records used for Table V.",
            },
            {
                "section": "Table V/XII",
                "load": "high",
                "metric": "DT-KF RMSE",
                "reported_or_claimed": 220,
                "recomputed": "not possible",
                "status": "corrupted_value",
                "severity": "high",
                "evidence_needed": "High-load ground truth and one-step prediction records.",
            },
            {
                "section": "Table V/XII",
                "load": "high",
                "metric": "DT-OPT MAE",
                "reported_or_claimed": 0.0083,
                "recomputed": "not possible",
                "status": "anomalous_value",
                "severity": "high",
                "evidence_needed": "High-load ground truth and DT-OPT prediction records.",
            },
            {
                "section": "Table V/XII",
                "load": "high",
                "metric": "DT-OPT RMSE/MAPE",
                "reported_or_claimed": "duplicates SemiGreedy exactly",
                "recomputed": "not possible",
                "status": "probable_copy_error",
                "severity": "high",
                "evidence_needed": "Raw predictions for both algorithms.",
            },
            {
                "section": "Table V/XII",
                "load": "all",
                "metric": "predictor identity",
                "reported_or_claimed": "scheduler names used as prediction baselines",
                "recomputed": "not applicable",
                "status": "methodology_ambiguous",
                "severity": "high",
                "evidence_needed": "Define the predictor attached to DT-OPT, SemiGreedy, and Fog-only, or relabel rows as Kalman/EMA/last-value.",
            },
        ]
    )

    valid_prediction_means = {
        "low": {"proposed": [0.0624, 0.0954, 0.1173], "fog": [0.1197, 0.2105, 0.2142]},
        "medium": {"proposed": [0.0769, 0.1217, 0.1276], "fog": [0.1485, 0.2460, 0.2501]},
    }
    for load, values in valid_prediction_means.items():
        for index, metric in enumerate(("mae", "rmse", "mape")):
            reduction = 100.0 * (values["fog"][index] - values["proposed"][index]) / values["fog"][index]
            rows.append(
                {
                    "section": "Table V low/medium arithmetic",
                    "load": load,
                    "metric": f"{metric}_reduction_vs_fog_only",
                    "reported_or_claimed": "47.9-48.2% MAE; 50.5-54.7% RMSE; 45.2-49.0% MAPE",
                    "recomputed": round(reduction, 4),
                    "status": "verified_arithmetic_only",
                    "severity": "low",
                    "evidence_needed": "Raw records are required to validate the displayed means and +/- terms.",
                }
            )
    return rows


def scheduler_smoke_test(config_data: dict[str, Any]) -> dict[str, Any]:
    config = SchedulerConfig(**config_data)
    task = Task(
        task_id="trace-mapped-example",
        arrival_s=0.0,
        deadline_s=0.75,
        compute_mi=800.0,
        uplink_bytes=120_000,
        downlink_bytes=20_000,
        min_accuracy=0.85,
        min_trust=0.70,
        energy_budget_j=2.0,
        inference=True,
    )
    venues = [
        VenueState("device", "local", 500, 1e-9, 0.0, 0.90, 0.82, battery_fraction=0.70),
        VenueState(
            "fog-1",
            "fog",
            6000,
            1e-12,
            2e-5,
            0.90,
            0.91,
            queue_work_mi=((0.50, 300.0),),
            link=LinkState(16e6, 30e6, 0.008, 0.009),
        ),
        VenueState(
            "cloud",
            "cloud",
            20000,
            1e-13,
            4e-5,
            0.97,
            0.96,
            queue_work_mi=((0.40, 1000.0),),
            link=LinkState(25e6, 50e6, 0.055, 0.055),
        ),
    ]
    return asdict(select_venue(task, venues, config))


def render_report(
    inventory: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    tuning: dict[str, Any],
) -> str:
    aggregate = [row for row in benchmark if row["signal"] == "aggregate"]
    lines = [
        "# Reproduction and Validation Report",
        "",
        "## Overall assessment: Needs revision",
        "",
        "The public input datasets and iFogSim2 are available, and the scheduler equations have been implemented. "
        "The paper's exact end-to-end results cannot be reproduced from the PDF because the implementation commit, "
        "exact parameters, trace windows, preprocessing, run seeds, and per-run outputs are absent. The corrupted high-load "
        "prediction cells cannot be repaired without the original ground-truth and prediction records.",
        "",
        "## Dataset availability",
        "",
        "| Input | Availability | Records | Detail |",
        "|---|---:|---:|---|",
    ]
    for row in inventory:
        lines.append(
            f"| {row['input']} | {row['availability']} | {row['records']} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Independent prediction benchmark",
            "",
            "This benchmark is a new, reproducible result, not a reconstruction of Table XII. It uses the final day of "
            "the Alibaba 2018 10-second aggregate trace as held-out data, 20-minute non-overlapping segments, and load "
            "terciles based on mean normalized CPU/network utilization. Kalman Q/R and EMA alpha are selected on the "
            "penultimate day. Metrics are segment means with Student-t 95% confidence-interval half-widths.",
            "",
            "| Load | Model | MAE | RMSE | Fractional MAPE | Segments |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            "| {load} | {model} | {mae_mean:.4f} +/- {mae_ci95:.4f} | "
            "{rmse_mean:.4f} +/- {rmse_ci95:.4f} | {mape_mean:.4f} +/- {mape_ci95:.4f} | {segments} |".format(
                **row
            )
        )

    high_issues = [row for row in audit if row["severity"] == "high"]
    lines.extend(
        [
            "",
            "## Blocking result issues",
            "",
        ]
    )
    for row in high_issues:
        lines.append(
            f"- **{row['section']} / {row['metric']}**: {row['status']}. Needed: {row['evidence_needed']}"
        )
    lines.extend(
        [
            "",
            "## What is needed for an exact paper rerun",
            "",
            "1. The authors' iFogSim2 repository/commit and modified Java classes, including all three baseline implementations.",
            "2. Exact Dataset A-C file selections, trace windows, retained record counts, outlier rule, load-scaling factors, and Dataset D definition.",
            "3. Numerical Kalman F/H/Q/R/x0/P0 settings, observation cadence, missing-data behavior, and one-step prediction alignment.",
            "4. Every currently blank Table III parameter: task conversion constants, deadlines, trust/accuracy/battery generation, link model, energy coefficients, pricing, gates, and objective weights.",
            "5. Paired per-seed task, prediction, node, and KPI records. At minimum, provide the five claimed runs; 20-30 paired seeds are needed for the requested inferential analysis.",
            "6. A definition of the run-cost index and the raw per-node completion counts used for Jain fairness.",
            "",
            "## Reproducibility notes",
            "",
            f"- Held-out setup: `{tuning['held_out_definition']}`.",
            f"- Segment length: {tuning['segment_length_seconds']} seconds.",
            "- The implementation applies `8 * bytes / bits-per-second` for communication time. Equation 14 in the PDF omits this factor even though task sizes are declared in bytes and bandwidth in bits/s.",
            "- The paper's low/medium relative error reductions are arithmetically consistent with displayed means; that does not validate the means themselves.",
            "- Table IV percentage changes versus Fog-only are arithmetically consistent with the displayed point estimates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = dataset_inventory(args.data_root.resolve(), args.ifogsim_root.resolve())
    audit = result_audit()
    trace_path = (
        args.data_root.resolve()
        / "datacenter-traces-datasets"
        / "alibaba2018"
        / "machine_usage_days_1_to_8_grouped_10_seconds.csv"
    )
    trace = load_trace(trace_path, list(config["benchmark"]["signals"]))
    detailed, benchmark, tuning = predictor_benchmark(trace, config["benchmark"])
    smoke_test = scheduler_smoke_test(config["scheduler"])

    write_csv(output_dir / "dataset_inventory.csv", inventory)
    write_csv(output_dir / "paper_result_audit.csv", audit)
    write_csv(output_dir / "prediction_segment_metrics.csv", detailed)
    write_csv(output_dir / "prediction_metrics.csv", benchmark)
    (output_dir / "tuned_parameters.json").write_text(
        json.dumps(tuning, indent=2), encoding="utf-8"
    )
    (output_dir / "scheduler_smoke_test.json").write_text(
        json.dumps(smoke_test, indent=2), encoding="utf-8"
    )
    (output_dir / "validation_report.md").write_text(
        render_report(inventory, audit, benchmark, tuning), encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
