from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from experiment import (
    AlgorithmVariant,
    ExperimentConfig,
    ablation_algorithms,
    generate_scenario,
    load_experiment_data,
    main_algorithms,
    prediction_metrics,
    sensitivity_algorithms,
    simulate_run,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "work" / "data_sources"
DEFAULT_OUTPUT_DIR = HERE / "results" / "independent_experiment"
LOADS = ("low", "medium", "high")
PRIMARY_METRICS = (
    "mean_latency_ms",
    "p95_latency_ms",
    "dmr_pct",
    "throughput_tasks_s",
    "energy_j",
    "monetary_cost",
    "run_cost_index",
    "fairness",
    "rejection_rate_pct",
    "local_pct",
    "fog_pct",
    "cloud_pct",
)
LOWER_IS_BETTER = {
    "mean_latency_ms",
    "p95_latency_ms",
    "dmr_pct",
    "energy_j",
    "monetary_cost",
    "run_cost_index",
    "rejection_rate_pct",
    "network_failure_rate_pct",
    "mean_packet_loss_pct",
    "mean_jitter_ms",
    "p95_jitter_ms",
    "retransmission_overhead_pct",
    "retransmission_energy_j",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent paired DT-KF scheduling experiments."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two or three seeds for development checks; not publication results.",
    )
    return parser.parse_args()


class GzipCsvSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(path, "wt", newline="", encoding="utf-8")
        self.writer: csv.DictWriter[str] | None = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(rows[0]))
            self.writer.writeheader()
        self.writer.writerows(rows)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "GzipCsvSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def summarize(values: Iterable[float]) -> tuple[int, float, float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0, math.nan, math.nan, math.nan
    mean = statistics.fmean(finite)
    if len(finite) < 2:
        return len(finite), mean, math.nan, math.nan
    sd = statistics.stdev(finite)
    ci = t_critical_975(len(finite) - 1) * sd / math.sqrt(len(finite))
    return len(finite), mean, sd, ci


def summarize_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        group = groups[key]
        result = dict(zip(group_fields, key))
        for metric in metrics:
            n, mean, sd, ci = summarize(float(row[metric]) for row in group)
            result[f"{metric}_n"] = n
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
            result[f"{metric}_ci95"] = ci
        output.append(result)
    return output


def _continued_beta(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_beta(a, b, x) / a
    return 1.0 - front * _continued_beta(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        return math.nan
    if value == 0:
        return 0.5
    x = degrees_freedom / (degrees_freedom + value * value)
    tail = 0.5 * regularized_beta(x, degrees_freedom / 2.0, 0.5)
    return 1.0 - tail if value > 0 else tail


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def jarque_bera_p(values: np.ndarray) -> float:
    if len(values) < 8:
        return 1.0
    centered = values - float(np.mean(values))
    variance = float(np.mean(np.square(centered)))
    if variance <= 1e-18:
        return 1.0
    skew = float(np.mean(centered**3) / variance**1.5)
    kurtosis = float(np.mean(centered**4) / variance**2)
    statistic = len(values) / 6.0 * (skew**2 + (kurtosis - 3.0) ** 2 / 4.0)
    return math.exp(-statistic / 2.0)


def paired_t_test(differences: np.ndarray) -> tuple[float, float]:
    sd = float(np.std(differences, ddof=1))
    if sd <= 1e-18:
        mean = float(np.mean(differences))
        return (math.copysign(math.inf, mean), 0.0) if mean != 0.0 else (0.0, 1.0)
    statistic = float(np.mean(differences)) / (sd / math.sqrt(len(differences)))
    p_value = 2.0 * (1.0 - student_t_cdf(abs(statistic), len(differences) - 1))
    return statistic, min(max(p_value, 0.0), 1.0)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def wilcoxon_signed_rank(differences: np.ndarray) -> tuple[float, float, float]:
    nonzero = differences[np.abs(differences) > 1e-15]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0.0
    ranks = _average_ranks(np.abs(nonzero))
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    n = len(nonzero)
    mean = n * (n + 1) / 4.0
    _, tie_counts = np.unique(np.abs(nonzero), return_counts=True)
    tie_correction = float(np.sum(tie_counts**3 - tie_counts)) / 48.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    z = (positive - mean) / math.sqrt(max(variance, 1e-18))
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    effect = (positive - negative) / (positive + negative)
    return min(positive, negative), min(max(p_value, 0.0), 1.0), effect


def _cohens_dz(values: np.ndarray) -> float:
    sd = float(np.std(values, ddof=1))
    if sd <= 1e-18:
        return math.copysign(math.inf, float(np.mean(values))) if np.mean(values) else 0.0
    return float(np.mean(values)) / sd


def _rank_biserial(values: np.ndarray) -> float:
    return wilcoxon_signed_rank(values)[2]


def bootstrap_effect_ci(
    differences: np.ndarray,
    effect_function: Callable[[np.ndarray], float],
    seed: int,
    samples: int = 1000,
) -> tuple[float, float]:
    if np.all(np.abs(differences) <= 1e-18):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    if effect_function is _cohens_dz:
        sampled = differences[
            rng.integers(0, len(differences), size=(samples, len(differences)))
        ]
        means = np.mean(sampled, axis=1)
        standard_deviations = np.std(sampled, axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimates_array = means / standard_deviations
        estimates_array = estimates_array[np.isfinite(estimates_array)]
        if len(estimates_array) == 0:
            return math.nan, math.nan
        return tuple(
            float(value) for value in np.percentile(estimates_array, [2.5, 97.5])
        )
    estimates: list[float] = []
    for _ in range(samples):
        sample = rng.choice(differences, size=len(differences), replace=True)
        estimate = effect_function(sample)
        if math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return math.nan, math.nan
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_statistics(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    proposed_name: str = "DT-KF-CostAware",
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["load"]), str(row["algorithm"]), int(row["seed"])): row for row in rows
    }
    algorithms = sorted({str(row["algorithm"]) for row in rows if row["algorithm"] != proposed_name})
    output: list[dict[str, Any]] = []
    for load in LOADS:
        family_start = len(output)
        for metric in metrics:
            for comparator in algorithms:
                seeds = sorted(
                    seed
                    for row_load, algorithm, seed in lookup
                    if row_load == load
                    and algorithm == proposed_name
                    and (load, comparator, seed) in lookup
                )
                if len(seeds) < 2:
                    continue
                proposed = np.asarray(
                    [float(lookup[(load, proposed_name, seed)][metric]) for seed in seeds]
                )
                baseline = np.asarray(
                    [float(lookup[(load, comparator, seed)][metric]) for seed in seeds]
                )
                differences = baseline - proposed if metric in LOWER_IS_BETTER else proposed - baseline
                n, mean, sd, ci = summarize(differences)
                normality_p = jarque_bera_p(differences)
                if normality_p >= 0.05:
                    statistic, raw_p = paired_t_test(differences)
                    method = "paired_t"
                    effect = _cohens_dz(differences)
                    effect_name = "cohens_dz"
                    effect_function = _cohens_dz
                else:
                    statistic, raw_p, effect = wilcoxon_signed_rank(differences)
                    method = "wilcoxon_signed_rank"
                    effect_name = "rank_biserial"
                    effect_function = _rank_biserial
                effect_low, effect_high = bootstrap_effect_ci(
                    differences,
                    effect_function,
                    seed=10_000 + len(output),
                )
                output.append(
                    {
                        "load": load,
                        "metric": metric,
                        "comparator": comparator,
                        "n_pairs": n,
                        "mean_improvement": mean,
                        "improvement_sd": sd,
                        "improvement_ci95": ci,
                        "normality_test": "Jarque-Bera",
                        "normality_p": normality_p,
                        "test": method,
                        "statistic": statistic,
                        "raw_p": raw_p,
                        "holm_p": math.nan,
                        "effect_name": effect_name,
                        "effect_size": effect,
                        "effect_ci95_low": effect_low,
                        "effect_ci95_high": effect_high,
                    }
                )
        family = output[family_start:]
        adjusted = holm_adjust([float(row["raw_p"]) for row in family])
        for row, corrected in zip(family, adjusted):
            row["holm_p"] = corrected
    return output


def _run_main(
    data: Any,
    config: ExperimentConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    prediction_runs: list[dict[str, Any]] = []
    task_path = output_dir / "raw_task_records.csv.gz"
    prediction_path = output_dir / "raw_prediction_records.csv.gz"
    with GzipCsvSink(task_path) as task_sink, GzipCsvSink(prediction_path) as prediction_sink:
        for load in LOADS:
            for seed_index in range(config.main_seeds):
                seed = 1000 + seed_index
                scenario = generate_scenario(data, config, load, seed)
                for algorithm in main_algorithms():
                    run, node_rows, task_rows, prediction_rows = simulate_run(
                        scenario,
                        algorithm,
                        config,
                        record_tasks=True,
                        record_predictions=True,
                    )
                    runs.append(run)
                    nodes.extend(node_rows)
                    task_sink.write(task_rows)
                    prediction_sink.write(prediction_rows)
                    prediction_runs.append(
                        {
                            "seed": seed,
                            "load": load,
                            "algorithm": algorithm.name,
                            **prediction_metrics(prediction_rows, config.mape_epsilon),
                        }
                    )
            print(f"main complete: {load}")
    return runs, nodes, prediction_runs


def _run_secondary(
    data: Any,
    config: ExperimentConfig,
    variants: tuple[AlgorithmVariant, ...],
    loads: tuple[str, ...],
    seed_offset: int,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for load in loads:
        for seed_index in range(config.secondary_seeds):
            scenario = generate_scenario(data, config, load, seed_offset + seed_index)
            for variant in variants:
                run, _, _, _ = simulate_run(scenario, variant, config)
                runs.append(run)
        print(f"secondary complete: {load} ({len(variants)} variants)")
    return runs


def _run_scalability(data: Any, config: ExperimentConfig) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    scales = ((50, 5), (200, 10), (500, 25), (1000, 50))
    for task_count, fog_count in scales:
        for seed_index in range(config.scalability_seeds):
            scenario = generate_scenario(
                data,
                config,
                "high",
                4000 + seed_index,
                fog_count=fog_count,
                fixed_task_count=task_count,
            )
            for algorithm in main_algorithms():
                run, _, _, _ = simulate_run(
                    scenario,
                    algorithm,
                    config,
                    measure_memory=True,
                )
                run["task_count"] = task_count
                run["fog_nodes"] = fog_count
                runs.append(run)
        print(f"scalability complete: N={task_count}, M={fog_count}")
    return runs


def _format_pm(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {ci:.{digits}f}"


def render_report(
    config: ExperimentConfig,
    main_summary: list[dict[str, Any]],
    prediction_summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    sensitivity_summary: list[dict[str, Any]],
    scalability_summary: list[dict[str, Any]],
) -> str:
    main_lookup = {(row["load"], row["algorithm"]): row for row in main_summary}
    pred_lookup = {(row["load"], row["algorithm"]): row for row in prediction_summary}
    lines = [
        "# Independent Recomputed Results",
        "",
        "> These values come from the reproducible experiment in this repository. They are not a reconstruction of the manuscript's undocumented runs.",
        "",
        f"Main evaluation: {config.main_seeds} paired seeds per load and algorithm. Secondary studies: {config.secondary_seeds} paired seeds.",
        "Rejected tasks count as deadline misses. Throughput uses the 90 s post-warm-up measurement window. Jain fairness is computed from completed-task counts across fog nodes for every seed.",
        "",
        "## Main scheduling results (mean +/- Student-t 95% CI)",
        "",
        "| Load | Algorithm | Mean latency (ms) | p95 (ms) | DMR (%) | Throughput | Energy (J) | Cost ($) | Objective | Fairness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for load in LOADS:
        for algorithm in ("DT-KF-CostAware", "DT-OPT", "SemiGreedy", "Fog-only"):
            row = main_lookup[(load, algorithm)]
            lines.append(
                "| {load} | {algorithm} | {lat} | {p95} | {dmr} | {throughput} | {energy} | {cost} | {objective} | {fairness} |".format(
                    load=load,
                    algorithm=algorithm,
                    lat=_format_pm(row["mean_latency_ms_mean"], row["mean_latency_ms_ci95"]),
                    p95=_format_pm(row["p95_latency_ms_mean"], row["p95_latency_ms_ci95"]),
                    dmr=_format_pm(row["dmr_pct_mean"], row["dmr_pct_ci95"]),
                    throughput=_format_pm(row["throughput_tasks_s_mean"], row["throughput_tasks_s_ci95"]),
                    energy=_format_pm(row["energy_j_mean"], row["energy_j_ci95"], 2),
                    cost=_format_pm(row["monetary_cost_mean"], row["monetary_cost_ci95"], 3),
                    objective=_format_pm(row["run_cost_index_mean"], row["run_cost_index_ci95"], 4),
                    fairness=_format_pm(row["fairness_mean"], row["fairness_ci95"], 4),
                )
            )
    lines.extend(
        [
            "",
            "## Prediction results, including all high-load cells",
            "",
            "| Load | Algorithm | MAE | RMSE | Fractional MAPE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for load in LOADS:
        for algorithm in ("DT-KF-CostAware", "DT-OPT", "SemiGreedy", "Fog-only"):
            row = pred_lookup[(load, algorithm)]
            lines.append(
                f"| {load} | {algorithm} | {_format_pm(row['mae_mean'], row['mae_ci95'], 5)} | "
                f"{_format_pm(row['rmse_mean'], row['rmse_ci95'], 5)} | "
                f"{_format_pm(row['mape_mean'], row['mape_ci95'], 5)} |"
            )
    significant = [row for row in paired if float(row["holm_p"]) < 0.05]
    lines.extend(
        [
            "",
            "## Paired statistical analysis",
            "",
            f"{len(significant)} of {len(paired)} declared paired comparisons have Holm-corrected p < 0.05. The complete table reports the test selected from a Jarque-Bera check, corrected p-value, effect size, and bootstrap 95% CI.",
            "",
            "## Ablation summary",
            "",
            "| Load | Variant | Latency (ms) | DMR (%) | Throughput | Objective | Fairness |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_summary:
        lines.append(
            f"| {row['load']} | {row['algorithm']} | {row['mean_latency_ms_mean']:.3f} | "
            f"{row['dmr_pct_mean']:.3f} | {row['throughput_tasks_s_mean']:.3f} | "
            f"{row['run_cost_index_mean']:.4f} | {row['fairness_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## High-load sensitivity summary",
            "",
            "| Configuration | Latency (ms) | DMR (%) | Energy (J) | Cost ($) | Objective | Fog/Cloud/Local (%) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity_summary:
        lines.append(
            f"| {row['algorithm']} | {row['mean_latency_ms_mean']:.3f} | {row['dmr_pct_mean']:.3f} | "
            f"{row['energy_j_mean']:.2f} | {row['monetary_cost_mean']:.3f} | "
            f"{row['run_cost_index_mean']:.4f} | {row['fog_pct_mean']:.1f}/{row['cloud_pct_mean']:.1f}/{row['local_pct_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Scalability summary",
            "",
            "| N | M | Algorithm | Decision median/p95/p99 (us) | Peak memory (KiB) | CPU (%) | Simulator tasks/s |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in scalability_summary:
        lines.append(
            f"| {row['task_count']} | {row['fog_nodes']} | {row['algorithm']} | "
            f"{row['decision_median_us_mean']:.2f}/{row['decision_p95_us_mean']:.2f}/{row['decision_p99_us_mean']:.2f} | "
            f"{row['peak_memory_kib_mean']:.1f} | {row['cpu_utilization_pct_mean']:.1f} | "
            f"{row['simulator_throughput_tasks_s_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Audit boundary",
            "",
            "The raw task and prediction records, run KPIs, node counts, manifest, and all secondary results are stored alongside this report. Because the paper does not disclose its original configuration or run logs, these values should replace placeholders only when the manuscript labels them as an independent rerun with the configuration reported here.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def render_latex(
    main_summary: list[dict[str, Any]],
    prediction_summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    sensitivity_summary: list[dict[str, Any]],
    scalability_summary: list[dict[str, Any]],
) -> str:
    main_lookup = {(row["load"], row["algorithm"]): row for row in main_summary}
    pred_lookup = {(row["load"], row["algorithm"]): row for row in prediction_summary}
    lines = [
        "% Independently recomputed results. Do not present as a reconstruction of the undocumented original runs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Independent paired rerun (mean $\\pm$ Student-$t$ 95\\% CI over 30 seeds).}",
        "\\label{tab:independent_main_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.25pt}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Load & Algorithm & Mean lat. & p95 & DMR & Throughput & Energy & Cost & Objective & Jain \\\\",
        "\\midrule",
    ]
    for load in LOADS:
        for algorithm in ("DT-KF-CostAware", "DT-OPT", "SemiGreedy", "Fog-only"):
            row = main_lookup[(load, algorithm)]
            lines.append(
                f"{load.title()} & {algorithm} & "
                f"{row['mean_latency_ms_mean']:.2f} $\\pm$ {row['mean_latency_ms_ci95']:.2f} & "
                f"{row['p95_latency_ms_mean']:.2f} $\\pm$ {row['p95_latency_ms_ci95']:.2f} & "
                f"{row['dmr_pct_mean']:.2f} $\\pm$ {row['dmr_pct_ci95']:.2f} & "
                f"{row['throughput_tasks_s_mean']:.2f} $\\pm$ {row['throughput_tasks_s_ci95']:.2f} & "
                f"{row['energy_j_mean']:.1f} $\\pm$ {row['energy_j_ci95']:.1f} & "
                f"{row['monetary_cost_mean']:.3f} $\\pm$ {row['monetary_cost_ci95']:.3f} & "
                f"{row['run_cost_index_mean']:.4f} $\\pm$ {row['run_cost_index_ci95']:.4f} & "
                f"{row['fairness_mean']:.4f} $\\pm$ {row['fairness_ci95']:.4f} \\\\"
            )
        lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Independent high-load prediction errors (mean $\\pm$ Student-$t$ 95\\% CI over 30 paired seeds).}",
            "\\label{tab:independent_high_prediction}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Algorithm & MAE & RMSE & MAPE \\\\",
            "\\midrule",
        ]
    )
    for algorithm in ("DT-KF-CostAware", "DT-OPT", "SemiGreedy", "Fog-only"):
        row = pred_lookup[("high", algorithm)]
        lines.append(
            f"{algorithm} & {row['mae_mean']:.5f} $\\pm$ {row['mae_ci95']:.5f} & "
            f"{row['rmse_mean']:.5f} $\\pm$ {row['rmse_ci95']:.5f} & "
            f"{row['mape_mean']:.5f} $\\pm$ {row['mape_ci95']:.5f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    for load in LOADS:
        lines.extend(
            [
                "",
                "\\begin{table*}[t]",
                "\\centering",
                f"\\caption{{Paired statistical comparisons for {load} load; positive differences favor DT-KF-CostAware.}}",
                f"\\label{{tab:independent_paired_{load}}}",
                "\\scriptsize",
                "\\begin{tabular}{llllrr}",
                "\\toprule",
                'Metric & Comparator & Improvement $\\pm$ 95\\% CI & Test & Holm $p$ & Effect (bootstrap 95\\% CI) \\\\',
                "\\midrule",
            ]
        )
        for row in paired:
            if row["load"] != load:
                continue
            lines.append(
                f"{latex_escape(row['metric'])} & {latex_escape(row['comparator'])} & "
                f"{row['mean_improvement']:.4g} $\\pm$ {row['improvement_ci95']:.3g} & "
                f"{latex_escape(row['test'])} & {row['holm_p']:.3g} & "
                f"{latex_escape(row['effect_name'])}={row['effect_size']:.3g} "
                f"[{row['effect_ci95_low']:.3g}, {row['effect_ci95_high']:.3g}] \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    lines.extend(
        [
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Matched component ablation over 20 paired seeds; entries are run means.}",
            "\\label{tab:independent_ablation}",
            "\\scriptsize",
            "\\begin{tabular}{llrrrrr}",
            "\\toprule",
            'Load & Variant & Mean latency & DMR & Throughput & Objective & Jain \\\\',
            "\\midrule",
        ]
    )
    for row in ablation_summary:
        lines.append(
            f"{str(row['load']).title()} & {latex_escape(row['algorithm'])} & "
            f"{row['mean_latency_ms_mean']:.2f} & {row['dmr_pct_mean']:.2f} & "
            f"{row['throughput_tasks_s_mean']:.2f} & {row['run_cost_index_mean']:.4f} & "
            f"{row['fairness_mean']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    lines.extend(
        [
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{High-load sensitivity analysis over 20 paired seeds; entries are run means.}",
            "\\label{tab:independent_sensitivity}",
            "\\scriptsize",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            'Configuration & Latency & DMR & Energy & Cost & Objective & Fog & Cloud & Local \\\\',
            "\\midrule",
        ]
    )
    for row in sensitivity_summary:
        lines.append(
            f"{latex_escape(row['algorithm'])} & {row['mean_latency_ms_mean']:.2f} & "
            f"{row['dmr_pct_mean']:.2f} & {row['energy_j_mean']:.2f} & "
            f"{row['monetary_cost_mean']:.3f} & {row['run_cost_index_mean']:.4f} & "
            f"{row['fog_pct_mean']:.1f} & {row['cloud_pct_mean']:.1f} & "
            f"{row['local_pct_mean']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    lines.extend(
        [
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Scalability results over 10 paired seeds per scale; entries are run means.}",
            "\\label{tab:independent_scalability}",
            "\\scriptsize",
            "\\begin{tabular}{rrlrrrrrr}",
            "\\toprule",
            '$N$ & $M$ & Algorithm & Median & p95 & p99 & Peak KiB & CPU & Sim. tasks/s \\\\',
            "\\midrule",
        ]
    )
    for row in scalability_summary:
        lines.append(
            f"{int(row['task_count'])} & {int(row['fog_nodes'])} & {latex_escape(row['algorithm'])} & "
            f"{row['decision_median_us_mean']:.2f} & {row['decision_p95_us_mean']:.2f} & "
            f"{row['decision_p99_us_mean']:.2f} & {row['peak_memory_kib_mean']:.1f} & "
            f"{row['cpu_utilization_pct_mean']:.1f} & {row['simulator_throughput_tasks_s_mean']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    if args.quick:
        config = replace(config, main_seeds=3, secondary_seeds=2, scalability_seeds=2)

    print("loading datasets")
    data = load_experiment_data(args.data_root.resolve())
    print("running main paired experiment")
    main_runs, node_rows, prediction_runs = _run_main(data, config, output_dir)
    main_summary = summarize_rows(main_runs, ("load", "algorithm"), PRIMARY_METRICS)
    prediction_summary = summarize_rows(
        prediction_runs, ("load", "algorithm"), ("mae", "rmse", "mape")
    )
    paired = paired_statistics(
        main_runs,
        (
            "mean_latency_ms",
            "p95_latency_ms",
            "dmr_pct",
            "throughput_tasks_s",
            "energy_j",
            "monetary_cost",
            "run_cost_index",
            "fairness",
        ),
    )
    write_csv(output_dir / "main_run_kpis.csv", main_runs)
    write_csv(output_dir / "main_summary.csv", main_summary)
    write_csv(output_dir / "node_counts.csv", node_rows)
    write_csv(output_dir / "prediction_run_metrics.csv", prediction_runs)
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "paired_statistics.csv", paired)
    (output_dir / "checkpoint.json").write_text(
        json.dumps({"completed_through": "main", "main_seeds": config.main_seeds}, indent=2),
        encoding="utf-8",
    )

    print("running paired ablations")
    ablation_runs = _run_secondary(
        data, config, ablation_algorithms(), LOADS, seed_offset=2000
    )
    ablation_summary = summarize_rows(
        ablation_runs, ("load", "algorithm"), PRIMARY_METRICS
    )
    write_csv(output_dir / "ablation_run_kpis.csv", ablation_runs)
    write_csv(output_dir / "ablation_summary.csv", ablation_summary)
    (output_dir / "checkpoint.json").write_text(
        json.dumps(
            {"completed_through": "ablation", "secondary_seeds": config.secondary_seeds},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("running high-load sensitivity")
    sensitivity_runs = _run_secondary(
        data,
        config,
        sensitivity_algorithms(),
        ("high",),
        seed_offset=3000,
    )
    sensitivity_summary = summarize_rows(
        sensitivity_runs, ("load", "algorithm"), PRIMARY_METRICS
    )
    write_csv(output_dir / "sensitivity_run_kpis.csv", sensitivity_runs)
    write_csv(output_dir / "sensitivity_summary.csv", sensitivity_summary)
    (output_dir / "checkpoint.json").write_text(
        json.dumps(
            {"completed_through": "sensitivity", "secondary_seeds": config.secondary_seeds},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("running scalability study")
    scalability_runs = _run_scalability(data, config)
    scalability_summary = summarize_rows(
        scalability_runs,
        ("task_count", "fog_nodes", "algorithm"),
        (
            "decision_median_us",
            "decision_p95_us",
            "decision_p99_us",
            "peak_memory_kib",
            "cpu_utilization_pct",
            "simulator_throughput_tasks_s",
            "fairness",
            "rejection_rate_pct",
        ),
    )
    write_csv(output_dir / "scalability_run_kpis.csv", scalability_runs)
    write_csv(output_dir / "scalability_summary.csv", scalability_summary)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "quick_development_run" if args.quick else "full_independent_rerun",
        "provenance": (
            "Independent trace-backed experiment. Not a reconstruction of the manuscript's "
            "undocumented original simulator runs."
        ),
        "data_root": str(args.data_root.resolve()),
        "config": asdict(config),
        "main_algorithms": [asdict(item) for item in main_algorithms()],
        "ablation_algorithms": [asdict(item) for item in ablation_algorithms()],
        "sensitivity_algorithms": [asdict(item) for item in sensitivity_algorithms()],
        "metric_definitions": {
            "dmr_pct": "100 * (rejected + completed after deadline) / arrived after warm-up",
            "throughput_tasks_s": "tasks completed within the 90 s measurement window / 90",
            "fairness": "Jain index over per-fog-node completed-task counts, computed per seed",
            "mape": "fractional absolute percentage error with epsilon=1e-6",
            "run_cost_index": "mean bounded realized four-component objective; range [0,1]",
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "independent_results_report.md").write_text(
        render_report(
            config,
            main_summary,
            prediction_summary,
            paired,
            ablation_summary,
            sensitivity_summary,
            scalability_summary,
        ),
        encoding="utf-8",
    )
    (output_dir / "independent_results_tables.tex").write_text(
        render_latex(
            main_summary,
            prediction_summary,
            paired,
            ablation_summary,
            sensitivity_summary,
            scalability_summary,
        ),
        encoding="utf-8",
    )
    (output_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "completed_through": "all",
                "main_seeds": config.main_seeds,
                "secondary_seeds": config.secondary_seeds,
                "scalability_seeds": config.scalability_seeds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote independent results to {output_dir}")


if __name__ == "__main__":
    main()
