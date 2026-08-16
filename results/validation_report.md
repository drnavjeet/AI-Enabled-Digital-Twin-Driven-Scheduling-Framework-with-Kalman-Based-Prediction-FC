# Reproduction and Validation Report

## Overall assessment: Needs revision

The public input datasets and iFogSim2 are available, and the scheduler equations have been implemented. The paper's exact end-to-end results cannot be reproduced from the PDF because the implementation commit, exact parameters, trace windows, preprocessing, run seeds, and per-run outputs are absent. The corrupted high-load prediction cells cannot be repaired without the original ground-truth and prediction records.

## Dataset availability

| Input | Availability | Records | Detail |
|---|---:|---:|---|
| Dataset A | downloaded_public_repository | 482687 | 93 CSV files; no header; 22 documented fields |
| Dataset B | downloaded_public_repository | 816 | Melbourne CBD users; 125 Optus server sites |
| Dataset C | downloaded_public_repository_and_pypi_package_available | 67242 | 10-second aggregate CPU, memory, network-in/out, and disk utilization; 1878 fewer rows than a complete eight-day series |
| Dataset D | derived_locally | 8640 | Explicit choice made here; the paper does not identify its Dataset D |
| Simulator | available_locally | 446 | Java source files in the existing local project |

## Independent prediction benchmark

This benchmark is a new, reproducible result, not a reconstruction of Table XII. It uses the final day of the Alibaba 2018 10-second aggregate trace as held-out data, 20-minute non-overlapping segments, and load terciles based on mean normalized CPU/network utilization. Kalman Q/R and EMA alpha are selected on the penultimate day. Metrics are segment means with Student-t 95% confidence-interval half-widths.

| Load | Model | MAE | RMSE | Fractional MAPE | Segments |
|---|---|---:|---:|---:|---:|
| low | kalman | 0.0049 +/- 0.0003 | 0.0064 +/- 0.0004 | 0.0140 +/- 0.0010 | 24 |
| low | ema | 0.0050 +/- 0.0003 | 0.0065 +/- 0.0004 | 0.0141 +/- 0.0011 | 24 |
| low | last_value | 0.0049 +/- 0.0003 | 0.0065 +/- 0.0004 | 0.0141 +/- 0.0010 | 24 |
| medium | kalman | 0.0048 +/- 0.0004 | 0.0063 +/- 0.0005 | 0.0113 +/- 0.0008 | 24 |
| medium | ema | 0.0049 +/- 0.0004 | 0.0065 +/- 0.0005 | 0.0115 +/- 0.0009 | 24 |
| medium | last_value | 0.0048 +/- 0.0004 | 0.0063 +/- 0.0005 | 0.0114 +/- 0.0008 | 24 |
| high | kalman | 0.0043 +/- 0.0002 | 0.0057 +/- 0.0004 | 0.0087 +/- 0.0006 | 24 |
| high | ema | 0.0044 +/- 0.0003 | 0.0058 +/- 0.0004 | 0.0089 +/- 0.0006 | 24 |
| high | last_value | 0.0043 +/- 0.0002 | 0.0057 +/- 0.0004 | 0.0087 +/- 0.0006 | 24 |

## Blocking result issues

- **Table IV/XI / 95% confidence intervals**: missing_from_table. Needed: Per-seed KPI records for every scheduler and load.
- **Table IV/XI / Cost J (normalized)**: definition_conflicts_with_equation_27. Needed: Aggregation window, denominator, objective weights, and separate monetary cost.
- **Equation 14 / communication time units**: unit_error_in_equation. Needed: Confirm whether simulator bandwidth is bits/s or bytes/s; use one convention everywhere.
- **Table V/XII / duplicated table conflict**: internally_contradictory_results. Needed: Delete or regenerate Table XII from the same verified run-level records used for Table V.
- **Table V/XII / DT-KF RMSE**: corrupted_value. Needed: High-load ground truth and one-step prediction records.
- **Table V/XII / DT-OPT MAE**: anomalous_value. Needed: High-load ground truth and DT-OPT prediction records.
- **Table V/XII / DT-OPT RMSE/MAPE**: probable_copy_error. Needed: Raw predictions for both algorithms.
- **Table V/XII / predictor identity**: methodology_ambiguous. Needed: Define the predictor attached to DT-OPT, SemiGreedy, and Fog-only, or relabel rows as Kalman/EMA/last-value.

## What is needed for an exact paper rerun

1. The authors' iFogSim2 repository/commit and modified Java classes, including all three baseline implementations.
2. Exact Dataset A-C file selections, trace windows, retained record counts, outlier rule, load-scaling factors, and Dataset D definition.
3. Numerical Kalman F/H/Q/R/x0/P0 settings, observation cadence, missing-data behavior, and one-step prediction alignment.
4. Every currently blank Table III parameter: task conversion constants, deadlines, trust/accuracy/battery generation, link model, energy coefficients, pricing, gates, and objective weights.
5. Paired per-seed task, prediction, node, and KPI records. At minimum, provide the five claimed runs; 20-30 paired seeds are needed for the requested inferential analysis.
6. A definition of the run-cost index and the raw per-node completion counts used for Jain fairness.

## Reproducibility notes

- Held-out setup: `final day of Alibaba 2018 10-second aggregate trace`.
- Segment length: 1200 seconds.
- The implementation applies `8 * bytes / bits-per-second` for communication time. Equation 14 in the PDF omits this factor even though task sizes are declared in bytes and bandwidth in bits/s.
- The paper's low/medium relative error reductions are arithmetically consistent with displayed means; that does not validate the means themselves.
- Table IV percentage changes versus Fog-only are arithmetically consistent with the displayed point estimates.
