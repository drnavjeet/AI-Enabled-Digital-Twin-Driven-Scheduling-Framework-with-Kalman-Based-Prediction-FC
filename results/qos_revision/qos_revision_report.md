# Packet-loss, jitter, and DRL-OO revision results

These are new paired simulator results, not reconstructed manuscript values. Packet loss and jitter use documented synthetic regimes because the available datasets contain no aligned QoS traces.

DT-KF-CostAware uses weights 0.48/0.20/0.16/0.16 for latency/energy/cost/accuracy, a 0.25 deadline-risk blend, and near-optimal fog balancing. These settings were selected on independent calibration seeds 40000-40007 before the final seeds 10000-10029 were evaluated.

## Main results

| QoS | Load | Algorithm | Mean latency (ms) | DMR (%) | Throughput | Energy (J) | Network failure (%) | Retransmission (%) | SLA success (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | high | DRL-OO | 383.59 +/- 4.19 | 8.57 | 17.84 | 276.37 | 0.000 | 0.13 | 91.43 |
| clean | high | DT-KF-CostAware | 388.23 +/- 4.72 | 9.36 | 17.83 | 225.06 | 0.000 | 0.12 | 90.64 |
| clean | high | DT-OPT | 440.30 +/- 5.54 | 13.92 | 17.81 | 188.68 | 0.000 | 0.13 | 86.08 |
| clean | high | Fog-only | 632.58 +/- 76.53 | 43.39 | 17.82 | 117.89 | 0.000 | 0.15 | 56.61 |
| clean | high | SemiGreedy | 310.35 +/- 4.84 | 1.25 | 17.92 | 369.11 | 0.000 | 0.12 | 98.75 |
| clean | low | DRL-OO | 232.43 +/- 4.97 | 0.32 | 4.97 | 85.85 | 0.000 | 0.11 | 99.68 |
| clean | low | DT-KF-CostAware | 217.47 +/- 3.61 | 0.13 | 4.97 | 109.44 | 0.000 | 0.09 | 99.87 |
| clean | low | DT-OPT | 224.38 +/- 4.57 | 0.40 | 4.97 | 96.59 | 0.000 | 0.09 | 99.60 |
| clean | low | Fog-only | 213.48 +/- 3.98 | 0.00 | 4.97 | 111.08 | 0.000 | 0.10 | 100.00 |
| clean | low | SemiGreedy | 167.18 +/- 1.58 | 0.00 | 4.98 | 234.68 | 0.000 | 0.07 | 100.00 |
| clean | medium | DRL-OO | 296.60 +/- 5.24 | 2.36 | 9.95 | 124.44 | 0.000 | 0.12 | 97.64 |
| clean | medium | DT-KF-CostAware | 274.59 +/- 4.37 | 1.19 | 9.95 | 157.34 | 0.000 | 0.12 | 98.81 |
| clean | medium | DT-OPT | 298.14 +/- 6.00 | 2.59 | 9.95 | 124.80 | 0.000 | 0.13 | 97.41 |
| clean | medium | Fog-only | 282.38 +/- 5.82 | 0.06 | 9.95 | 131.27 | 0.000 | 0.12 | 99.94 |
| clean | medium | SemiGreedy | 218.22 +/- 2.11 | 0.00 | 9.96 | 319.90 | 0.000 | 0.11 | 100.00 |
| impaired | high | DRL-OO | 482.77 +/- 6.15 | 25.36 | 16.95 | 265.61 | 0.008 | 6.90 | 74.64 |
| impaired | high | DT-KF-CostAware | 475.64 +/- 5.31 | 24.86 | 17.17 | 235.08 | 0.014 | 6.73 | 75.14 |
| impaired | high | DT-OPT | 508.01 +/- 5.95 | 27.54 | 17.06 | 221.13 | 0.016 | 6.91 | 72.46 |
| impaired | high | Fog-only | 697.41 +/- 77.09 | 57.26 | 17.80 | 117.95 | 0.016 | 7.97 | 42.74 |
| impaired | high | SemiGreedy | 395.99 +/- 6.26 | 12.81 | 17.90 | 347.40 | 0.008 | 6.55 | 87.19 |
| impaired | low | DRL-OO | 277.83 +/- 5.32 | 1.15 | 4.97 | 85.80 | 0.000 | 5.01 | 98.85 |
| impaired | low | DT-KF-CostAware | 262.34 +/- 4.14 | 0.43 | 4.97 | 107.39 | 0.000 | 4.69 | 99.57 |
| impaired | low | DT-OPT | 273.41 +/- 5.16 | 1.57 | 4.97 | 93.94 | 0.007 | 4.97 | 98.43 |
| impaired | low | Fog-only | 255.30 +/- 4.22 | 0.03 | 4.97 | 110.52 | 0.000 | 5.01 | 99.97 |
| impaired | low | SemiGreedy | 216.57 +/- 2.56 | 0.03 | 4.97 | 230.69 | 0.000 | 4.00 | 99.97 |
| impaired | medium | DRL-OO | 363.49 +/- 6.03 | 6.77 | 9.94 | 123.95 | 0.007 | 6.14 | 93.23 |
| impaired | medium | DT-KF-CostAware | 340.93 +/- 6.37 | 3.87 | 9.94 | 150.69 | 0.000 | 5.63 | 96.13 |
| impaired | medium | DT-OPT | 367.64 +/- 7.20 | 7.56 | 9.94 | 123.60 | 0.004 | 6.19 | 92.44 |
| impaired | medium | Fog-only | 337.11 +/- 6.14 | 0.81 | 9.95 | 130.09 | 0.004 | 6.39 | 99.19 |
| impaired | medium | SemiGreedy | 284.27 +/- 3.24 | 0.42 | 9.95 | 307.16 | 0.000 | 5.23 | 99.58 |
| moderate | high | DRL-OO | 412.66 +/- 4.81 | 11.97 | 17.75 | 272.80 | 0.000 | 2.04 | 88.03 |
| moderate | high | DT-KF-CostAware | 413.95 +/- 5.07 | 12.30 | 17.77 | 227.23 | 0.000 | 1.93 | 87.70 |
| moderate | high | DT-OPT | 459.02 +/- 5.26 | 16.19 | 17.76 | 199.27 | 0.000 | 2.03 | 83.81 |
| moderate | high | Fog-only | 649.97 +/- 77.07 | 46.36 | 17.81 | 117.98 | 0.000 | 2.28 | 53.64 |
| moderate | high | SemiGreedy | 333.35 +/- 5.11 | 2.39 | 17.92 | 362.70 | 0.000 | 1.91 | 97.61 |
| moderate | low | DRL-OO | 244.94 +/- 5.07 | 0.54 | 4.97 | 85.79 | 0.000 | 1.47 | 99.46 |
| moderate | low | DT-KF-CostAware | 229.73 +/- 3.70 | 0.20 | 4.97 | 108.83 | 0.000 | 1.36 | 99.80 |
| moderate | low | DT-OPT | 238.15 +/- 4.91 | 0.71 | 4.97 | 95.68 | 0.000 | 1.43 | 99.29 |
| moderate | low | Fog-only | 224.51 +/- 4.00 | 0.00 | 4.97 | 110.95 | 0.000 | 1.42 | 100.00 |
| moderate | low | SemiGreedy | 180.76 +/- 1.77 | 0.00 | 4.98 | 233.69 | 0.000 | 1.15 | 100.00 |
| moderate | medium | DRL-OO | 315.78 +/- 5.41 | 3.60 | 9.95 | 124.23 | 0.000 | 1.82 | 96.40 |
| moderate | medium | DT-KF-CostAware | 293.03 +/- 5.00 | 1.78 | 9.95 | 155.66 | 0.000 | 1.67 | 98.22 |
| moderate | medium | DT-OPT | 317.96 +/- 6.31 | 3.95 | 9.95 | 124.48 | 0.000 | 1.83 | 96.05 |
| moderate | medium | Fog-only | 296.64 +/- 5.85 | 0.07 | 9.95 | 130.92 | 0.000 | 1.91 | 99.93 |
| moderate | medium | SemiGreedy | 236.08 +/- 2.37 | 0.00 | 9.96 | 316.67 | 0.000 | 1.52 | 100.00 |

## Statistical comparisons

267 of 360 predeclared paired comparisons have Holm-adjusted p < 0.05. The CSV table contains test selection, effect size, and bootstrap interval.

## QoS ablation

| Variant | Latency (ms) | DMR (%) | Throughput | Energy (J) | SLA success (%) |
|---|---:|---:|---:|---:|---:|
| Full-QoS | 474.64 | 23.55 | 17.36 | 238.98 | 76.45 |
| Jitter-only | 474.17 | 23.52 | 17.36 | 238.75 | 76.48 |
| Loss-only | 477.28 | 27.08 | 17.84 | 236.07 | 72.92 |
| No-QoS-awareness | 477.83 | 27.13 | 17.84 | 235.55 | 72.87 |

## Sensitivity

The complete one-factor-at-a-time loss and jitter results are in `sensitivity_summary.csv`.

## Interpretation guardrail

The DRL-OO result is a paper-aligned discrete-action adaptation evaluated in the common venue-selection action space. It is not claimed as a bit-for-bit reproduction of the source paper, whose code and training data were not published.

The run_cost_index uses each scheduler variant's configured objective weights. Because DT-KF-CostAware was recalibrated to 0.48/0.20/0.16/0.16, that index is retained for within-variant diagnostics but is not interpreted as a cross-algorithm ranking metric.
