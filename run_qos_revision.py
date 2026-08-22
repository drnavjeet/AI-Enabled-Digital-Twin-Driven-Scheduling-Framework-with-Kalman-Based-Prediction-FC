from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from drl_oo import DRLOOConfig, DRLOOPolicy, train_policy
from experiment import (
    AlgorithmVariant,
    ExperimentConfig,
    generate_scenario,
    load_experiment_data,
    prediction_metrics,
    qos_algorithms,
    simulate_run,
)
from run_independent_experiments import (
    GzipCsvSink,
    paired_statistics,
    summarize_rows,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "work" / "data_sources"
DEFAULT_OUTPUT_DIR = HERE / "results" / "qos_revision"
LOADS = ("low", "medium", "high")
QOS_REGIMES = ("clean", "moderate", "impaired")
POLICY_SEEDS = (501, 502, 503, 504, 505)
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
    "network_failure_rate_pct",
    "mean_packet_loss_pct",
    "mean_jitter_ms",
    "p95_jitter_ms",
    "retransmission_overhead_pct",
    "retransmission_energy_j",
    "sla_success_pct",
    "decision_median_us",
    "decision_p95_us",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the packet-loss/jitter and DRL-OO reviewer-revision experiments."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def train_or_load_policies(
    data: Any,
    experiment_config: ExperimentConfig,
    output_dir: Path,
    *,
    quick: bool,
) -> tuple[list[DRLOOPolicy], list[dict[str, float]]]:
    policy_seeds = POLICY_SEEDS[:1] if quick else POLICY_SEEDS
    training_config = DRLOOConfig(
        episodes=4 if quick else 200,
        tasks_per_episode=30 if quick else 64,
        updates_per_episode=1 if quick else 2,
    )
    policies: list[DRLOOPolicy] = []
    logs: list[dict[str, float]] = []
    model_dir = output_dir / "models"
    for policy_seed in policy_seeds:
        path = model_dir / f"drl_oo_seed_{policy_seed}.npz"
        if path.exists() and path.with_suffix(".json").exists():
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if metadata.get("config") == asdict(training_config):
                policies.append(DRLOOPolicy.load(path))
                stored_log = output_dir / f"training_log_seed_{policy_seed}.csv"
                if stored_log.exists():
                    import csv

                    with stored_log.open(newline="", encoding="utf-8") as handle:
                        logs.extend(
                            {key: float(value) for key, value in row.items()}
                            for row in csv.DictReader(handle)
                        )
                print(f"loaded trained policy {policy_seed}")
                continue
        print(f"training DRL-OO policy {policy_seed}")
        policy, policy_log = train_policy(
            data,
            experiment_config,
            policy_seed=policy_seed,
            config=training_config,
        )
        policy.save(
            path,
            {
                "source_article": "https://link.springer.com/article/10.1186/s13638-025-02534-0",
                "adaptation_note": (
                    "The paper's server-plus-VM action is adapted to the common venue-selection "
                    "action space used by every scheduler in this simulator."
                ),
            },
        )
        write_csv(output_dir / f"training_log_seed_{policy_seed}.csv", policy_log)
        policies.append(policy)
        logs.extend(policy_log)
        print(f"training complete: policy {policy_seed}")
    return policies, logs


def target_metrics(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in ("load", "packet_loss", "jitter_s"):
        selected = [row for row in rows if row.get("target", "load") == target]
        if not selected:
            continue
        output.append({"target": target, **prediction_metrics(selected, epsilon)})
    return output


def run_main(
    data: Any,
    config: ExperimentConfig,
    policies: list[DRLOOPolicy],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    prediction_runs: list[dict[str, Any]] = []
    with GzipCsvSink(output_dir / "raw_task_records.csv.gz") as task_sink, GzipCsvSink(
        output_dir / "raw_prediction_records.csv.gz"
    ) as prediction_sink:
        for qos_regime in QOS_REGIMES:
            for load in LOADS:
                for seed_index in range(config.main_seeds):
                    seed = 10_000 + seed_index
                    scenario = generate_scenario(
                        data, config, load, seed, qos_regime=qos_regime
                    )
                    assigned_policy = policies[seed_index % len(policies)]
                    for algorithm in qos_algorithms():
                        policy = assigned_policy if algorithm.policy == "drl_oo" else None
                        run, node_rows, task_rows, prediction_rows = simulate_run(
                            scenario,
                            algorithm,
                            config,
                            record_tasks=True,
                            record_predictions=True,
                            record_qos_predictions=algorithm.qos_aware,
                            policy_model=policy,
                        )
                        policy_seed: int | str = policy.seed if policy is not None else ""
                        run["policy_seed"] = policy_seed
                        for row in task_rows:
                            row["policy_seed"] = policy_seed
                        for row in prediction_rows:
                            row["policy_seed"] = policy_seed
                        runs.append(run)
                        nodes.extend(node_rows)
                        task_sink.write(task_rows)
                        prediction_sink.write(prediction_rows)
                        for metrics in target_metrics(prediction_rows, config.mape_epsilon):
                            prediction_runs.append(
                                {
                                    "seed": seed,
                                    "load": load,
                                    "qos_regime": qos_regime,
                                    "algorithm": algorithm.name,
                                    "policy_seed": policy_seed,
                                    **metrics,
                                }
                            )
                print(f"main complete: {qos_regime}/{load}")
    return runs, nodes, prediction_runs


def qos_ablation_variants() -> tuple[AlgorithmVariant, ...]:
    full = AlgorithmVariant(
        "Full-QoS", "kalman", "cost", qos_aware=True, minimum_lhs=0.30
    )
    return (
        full,
        replace(full, name="No-QoS-awareness", qos_aware=False),
        replace(full, name="Loss-only", jitter_aware=False),
        replace(full, name="Jitter-only", loss_aware=False),
    )


def run_ablation(data: Any, config: ExperimentConfig) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed_index in range(config.secondary_seeds):
        scenario = generate_scenario(
            data, config, "high", 20_000 + seed_index, qos_regime="impaired"
        )
        for variant in qos_ablation_variants():
            run, _, _, _ = simulate_run(scenario, variant, config)
            runs.append(run)
    print("QoS ablation complete")
    return runs


def run_sensitivity(data: Any, config: ExperimentConfig) -> list[dict[str, Any]]:
    settings = [
        ("loss", loss, 0.008) for loss in (0.0, 0.01, 0.03, 0.05, 0.10)
    ] + [
        ("jitter", 0.015, jitter) for jitter in (0.0, 0.005, 0.015, 0.030, 0.050)
    ]
    variant = AlgorithmVariant("DT-KF-CostAware", "kalman", "cost", qos_aware=True)
    runs: list[dict[str, Any]] = []
    for sensitivity_type, loss_rate, jitter_s in settings:
        for seed_index in range(config.secondary_seeds):
            scenario = generate_scenario(
                data,
                config,
                "high",
                30_000 + seed_index,
                qos_regime="moderate",
                qos_loss_rate=loss_rate,
                qos_jitter_s=jitter_s,
            )
            run, _, _, _ = simulate_run(scenario, variant, config)
            run.update(
                {
                    "sensitivity_type": sensitivity_type,
                    "configured_loss_pct": 100.0 * loss_rate,
                    "configured_jitter_ms": 1000.0 * jitter_s,
                }
            )
            runs.append(run)
        print(
            f"sensitivity complete: {sensitivity_type}, loss={100 * loss_rate:g}%, jitter={1000 * jitter_s:g} ms"
        )
    return runs


def run_scalability(
    data: Any,
    config: ExperimentConfig,
    policies: list[DRLOOPolicy],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    scales = ((50, 5), (200, 10), (500, 25), (1000, 50))
    for task_count, fog_count in scales:
        for seed_index in range(config.scalability_seeds):
            scenario = generate_scenario(
                data,
                config,
                "high",
                40_000 + seed_index,
                fog_count=fog_count,
                fixed_task_count=task_count,
                qos_regime="moderate",
            )
            assigned_policy = policies[seed_index % len(policies)]
            for algorithm in qos_algorithms():
                policy = assigned_policy if algorithm.policy == "drl_oo" else None
                run, _, _, _ = simulate_run(
                    scenario,
                    algorithm,
                    config,
                    measure_memory=seed_index == 0,
                    policy_model=policy,
                )
                run.update(
                    {
                        "task_count": task_count,
                        "fog_nodes": fog_count,
                        "policy_seed": policy.seed if policy is not None else "",
                    }
                )
                runs.append(run)
        print(f"scalability complete: tasks={task_count}, fog={fog_count}")
    return runs


def paired_by_qos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "mean_latency_ms",
        "p95_latency_ms",
        "dmr_pct",
        "throughput_tasks_s",
        "energy_j",
        "run_cost_index",
        "fairness",
        "network_failure_rate_pct",
        "retransmission_overhead_pct",
        "sla_success_pct",
    )
    output: list[dict[str, Any]] = []
    for qos_regime in QOS_REGIMES:
        subset = [row for row in rows if row["qos_regime"] == qos_regime]
        for row in paired_statistics(subset, metrics):
            output.append({"qos_regime": qos_regime, **row})
    return output


def render_report(
    main_summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    sensitivity_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Packet-loss, jitter, and DRL-OO revision results",
        "",
        "These are new paired simulator results, not reconstructed manuscript values. Packet loss and jitter use documented synthetic regimes because the available datasets contain no aligned QoS traces.",
        "",
        "## Main results",
        "",
        "| QoS | Load | Algorithm | Mean latency (ms) | DMR (%) | Throughput | Energy (J) | Network failure (%) | Retransmission (%) | SLA success (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_summary:
        lines.append(
            f"| {row['qos_regime']} | {row['load']} | {row['algorithm']} | "
            f"{row['mean_latency_ms_mean']:.2f} +/- {row['mean_latency_ms_ci95']:.2f} | "
            f"{row['dmr_pct_mean']:.2f} | {row['throughput_tasks_s_mean']:.2f} | "
            f"{row['energy_j_mean']:.2f} | {row['network_failure_rate_pct_mean']:.3f} | "
            f"{row['retransmission_overhead_pct_mean']:.2f} | {row['sla_success_pct_mean']:.2f} |"
        )
    significant = sum(float(row["holm_p"]) < 0.05 for row in paired)
    lines.extend(
        [
            "",
            "## Statistical comparisons",
            "",
            f"{significant} of {len(paired)} predeclared paired comparisons have Holm-adjusted p < 0.05. The CSV table contains test selection, effect size, and bootstrap interval.",
            "",
            "## QoS ablation",
            "",
            "| Variant | Latency (ms) | DMR (%) | Throughput | Energy (J) | SLA success (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_summary:
        lines.append(
            f"| {row['algorithm']} | {row['mean_latency_ms_mean']:.2f} | {row['dmr_pct_mean']:.2f} | "
            f"{row['throughput_tasks_s_mean']:.2f} | {row['energy_j_mean']:.2f} | {row['sla_success_pct_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "The complete one-factor-at-a-time loss and jitter results are in `sensitivity_summary.csv`.",
            "",
            "## Interpretation guardrail",
            "",
            "The DRL-OO result is a paper-aligned discrete-action adaptation evaluated in the common venue-selection action space. It is not claimed as a bit-for-bit reproduction of the source paper, whose code and training data were not published.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_latex(main_summary: list[dict[str, Any]]) -> str:
    rows = [
        "% Generated packet-loss/jitter and DRL-OO revision table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired revised evaluation under the impaired network regime (mean $\\pm$ 95\\% CI over 30 seeds).}",
        "\\label{tab:qos_ai_revised}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Load & Algorithm & Latency (ms) & DMR (\\%) & Throughput & Energy (J) & SLA (\\%) \\\\",
        "\\midrule",
    ]
    for row in main_summary:
        if row["qos_regime"] != "impaired":
            continue
        algorithm = str(row["algorithm"]).replace("_", "\\_")
        rows.append(
            f"{row['load']} & {algorithm} & {row['mean_latency_ms_mean']:.2f} $\\pm$ {row['mean_latency_ms_ci95']:.2f} & "
            f"{row['dmr_pct_mean']:.2f} & {row['throughput_tasks_s_mean']:.2f} & "
            f"{row['energy_j_mean']:.2f} & {row['sla_success_pct_mean']:.2f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(rows) + "\n"


def write_reviewer_response(output_dir: Path) -> None:
    text = """# Reviewer response additions

## Comment 16: packet loss and jitter

We agree that the original network model did not explicitly represent packet loss or jitter. The revised simulator now models time-varying uplink/downlink loss and jitter, retry-capped packet retransmissions, retransmission delay and energy, transport failure, and one-step QoS prediction. All schedulers experience identical paired QoS traces. Because the source datasets do not provide aligned loss/jitter observations, the revision uses three documented synthetic regimes and reports a separate one-factor-at-a-time sensitivity analysis. The manuscript now states this limitation explicitly.

## Comment 17: recent AI scheduler

We agree that the original baselines did not include a sufficiently recent learning scheduler. We added a paper-aligned adaptation of Wang and Sun's 2025 DRL scheduler with ordinal optimization (EURASIP Journal on Wireless Communications and Networking, DOI: 10.1186/s13638-025-02534-0). The implementation uses actor and critic networks, replay learning, target-network soft updates, binary action perturbations, 100 ordinal candidates, and critic top-10 filtering. Five independently seeded policies are trained only on training scenarios, frozen, and assigned evenly across the 30 paired evaluation seeds. Since the source article does not publish code or training data and our common simulator exposes venue selection rather than server-plus-VM allocation, we label this a paper-aligned discrete-action adaptation rather than an exact reproduction.
"""
    (output_dir / "reviewer_response_additions.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    if args.quick:
        config = replace(config, main_seeds=2, secondary_seeds=2, scalability_seeds=1)

    print("loading datasets")
    data = load_experiment_data(args.data_root.resolve())
    policies, training_log = train_or_load_policies(
        data, config, output_dir, quick=args.quick
    )
    if training_log:
        write_csv(output_dir / "training_log.csv", training_log)

    print("running main QoS/AI experiment")
    main_runs, node_rows, prediction_runs = run_main(data, config, policies, output_dir)
    main_summary = summarize_rows(
        main_runs, ("qos_regime", "load", "algorithm"), PRIMARY_METRICS
    )
    prediction_summary = summarize_rows(
        prediction_runs,
        ("qos_regime", "load", "algorithm", "target"),
        ("mae", "rmse", "mape"),
    )
    paired = paired_by_qos(main_runs)
    write_csv(output_dir / "main_run_kpis.csv", main_runs)
    write_csv(output_dir / "main_summary.csv", main_summary)
    write_csv(output_dir / "node_counts.csv", node_rows)
    write_csv(output_dir / "prediction_run_metrics.csv", prediction_runs)
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "paired_statistics.csv", paired)

    print("running QoS ablation")
    ablation_runs = run_ablation(data, config)
    ablation_summary = summarize_rows(
        ablation_runs, ("algorithm",), PRIMARY_METRICS
    )
    write_csv(output_dir / "qos_ablation_run_kpis.csv", ablation_runs)
    write_csv(output_dir / "qos_ablation_summary.csv", ablation_summary)

    print("running loss/jitter sensitivity")
    sensitivity_runs = run_sensitivity(data, config)
    sensitivity_summary = summarize_rows(
        sensitivity_runs,
        ("sensitivity_type", "configured_loss_pct", "configured_jitter_ms"),
        PRIMARY_METRICS,
    )
    write_csv(output_dir / "sensitivity_run_kpis.csv", sensitivity_runs)
    write_csv(output_dir / "sensitivity_summary.csv", sensitivity_summary)

    print("running revised scalability")
    scalability_runs = run_scalability(data, config, policies)
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
            "dmr_pct",
            "sla_success_pct",
        ),
    )
    write_csv(output_dir / "scalability_run_kpis.csv", scalability_runs)
    write_csv(output_dir / "scalability_summary.csv", scalability_summary)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "quick_development_run" if args.quick else "full_reviewer_revision_run",
        "provenance": (
            "New trace-backed simulator experiment with explicitly synthetic packet-loss and "
            "jitter regimes; not a reconstruction of undocumented manuscript values."
        ),
        "data_root": str(args.data_root.resolve()),
        "config": asdict(config),
        "qos_regimes": {
            "clean": {"base_packet_loss": 0.001, "base_jitter_s": 0.0015},
            "moderate": {"base_packet_loss": 0.015, "base_jitter_s": 0.008},
            "impaired": {"base_packet_loss": 0.050, "base_jitter_s": 0.025},
        },
        "algorithms": [asdict(item) for item in qos_algorithms()],
        "drl_oo": {
            "policy_seeds": [policy.seed for policy in policies],
            "source": "https://link.springer.com/article/10.1186/s13638-025-02534-0",
            "claim": "paper-aligned discrete-action adaptation, not an exact reproduction",
        },
        "metric_definitions": {
            "network_failure_rate_pct": "100 * retry-exhausted task flows / measured arrivals",
            "retransmission_overhead_pct": "100 * retransmitted payload bytes / original remote payload bytes",
            "sla_success_pct": "100 * tasks neither rejected, retry-exhausted, nor completed after deadline / arrivals",
            "packet_loss_model": "independent packet loss with 1460-byte payload and at most three retries",
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "qos_revision_report.md").write_text(
        render_report(main_summary, paired, ablation_summary, sensitivity_summary),
        encoding="utf-8",
    )
    (output_dir / "qos_revision_tables.tex").write_text(
        render_latex(main_summary), encoding="utf-8"
    )
    write_reviewer_response(output_dir)
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
    print(f"wrote QoS revision results to {output_dir}")


if __name__ == "__main__":
    main()
