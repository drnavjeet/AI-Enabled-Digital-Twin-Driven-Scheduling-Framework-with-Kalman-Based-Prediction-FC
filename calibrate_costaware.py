from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiment import (
    AlgorithmVariant,
    ExperimentConfig,
    generate_scenario,
    load_experiment_data,
    simulate_run,
)
from run_independent_experiments import summarize_rows, write_csv


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "work" / "data_sources"
DEFAULT_OUTPUT = HERE / "results" / "qos_revision" / "costaware_calibration.csv"


def variants() -> tuple[AlgorithmVariant, ...]:
    base = AlgorithmVariant(
        "Baseline-40",
        "kalman",
        "cost",
        minimum_lhs=0.30,
        qos_aware=True,
    )
    return (
        base,
        replace(
            base,
            name="Reweight-48",
            weights=(0.48, 0.20, 0.16, 0.16),
        ),
        replace(
            base,
            name="Deadline-48",
            weights=(0.48, 0.20, 0.16, 0.16),
            deadline_risk_blend=0.25,
        ),
        replace(
            base,
            name="Balanced-48",
            weights=(0.48, 0.20, 0.16, 0.16),
            deadline_risk_blend=0.25,
            balance_score_tolerance=0.015,
            balance_latency_tolerance=0.05,
        ),
        replace(
            base,
            name="Balanced-48-wide",
            weights=(0.48, 0.20, 0.16, 0.16),
            deadline_risk_blend=0.25,
            balance_score_tolerance=0.05,
            balance_latency_tolerance=0.12,
        ),
        replace(
            base,
            name="Balanced-50-wide",
            weights=(0.50, 0.20, 0.15, 0.15),
            deadline_risk_blend=0.35,
            balance_score_tolerance=0.05,
            balance_latency_tolerance=0.12,
        ),
    )


def add_selection_score(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["algorithm"] == "Baseline-40")
    for row in rows:
        latency_ratio = row["mean_latency_ms_mean"] / baseline["mean_latency_ms_mean"]
        dmr_ratio = row["dmr_pct_mean"] / baseline["dmr_pct_mean"]
        energy_ratio = row["energy_j_mean"] / baseline["energy_j_mean"]
        fairness_loss = max(
            baseline["fairness_mean"] - row["fairness_mean"], 0.0
        )
        row["calibration_score"] = (
            0.45 * latency_ratio
            + 0.35 * dmr_ratio
            + 0.15 * energy_ratio
            + 2.5 * fairness_loss
        )
        row["passes_energy_guardrail"] = energy_ratio <= 1.25
        row["passes_fairness_guardrail"] = fairness_loss <= 0.02


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate DT-KF-CostAware on seeds separate from final evaluation."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    data = load_experiment_data(args.data_root.resolve())
    config = ExperimentConfig()
    runs: list[dict[str, Any]] = []
    for qos_regime in ("moderate", "impaired"):
        for load in ("medium", "high"):
            for seed_index in range(args.seeds):
                scenario = generate_scenario(
                    data,
                    config,
                    load,
                    40_000 + seed_index,
                    qos_regime=qos_regime,
                )
                for variant in variants():
                    run, _, _, _ = simulate_run(scenario, variant, config)
                    runs.append(run)

    summary = summarize_rows(
        runs,
        ("algorithm",),
        ("mean_latency_ms", "dmr_pct", "energy_j", "fairness", "sla_success_pct"),
    )
    add_selection_score(summary)
    summary.sort(key=lambda row: float(row["calibration_score"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, summary)
    best = next(
        row
        for row in summary
        if row["passes_energy_guardrail"] and row["passes_fairness_guardrail"]
    )
    print(f"selected calibration candidate: {best['algorithm']}")
    for row in summary:
        print(
            row["algorithm"],
            f"latency={row['mean_latency_ms_mean']:.2f}",
            f"dmr={row['dmr_pct_mean']:.2f}",
            f"energy={row['energy_j_mean']:.2f}",
            f"fairness={row['fairness_mean']:.4f}",
            f"score={row['calibration_score']:.4f}",
        )


if __name__ == "__main__":
    main()
