# Calculation Status

Exactly 74 relative comparisons are identifiable from the displayed table values.
They verify arithmetic only; they do not validate the underlying experiment or uncertainty.

## DT-KF-CostAware improvement versus Fog-only

| Metric | Low | Medium | High |
|---|---:|---:|---:|
| Mean latency | 15.4388% | 14.0781% | 23.9725% |
| p95 latency | 14.3621% | 16.6152% | 26.5565% |
| Deadline-miss ratio | 23.4772% | 57.0883% | 79.5493% |
| Throughput | 4.4255% | 5.0957% | 6.2063% |
| Energy | 2.9248% | 13.0546% | 9.3052% |
| Run-cost index | 17.0000% | 21.3115% | 19.2053% |

## Prediction-error reduction versus Fog-only

| Load | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Low | 47.8697% | 54.6793% | 45.2381% |
| Medium | 48.2155% | 50.5285% | 48.9804% |
| High | 48.7696% | unavailable | unavailable |

## Not calculable from the supplied summary

- Nine baseline Jain-fairness values: per-fog-node completed-task counts are absent.
- Five high-load prediction cells: the original ground truth and aligned predictions are absent.
- Performance SDs, 95% CIs, paired tests, corrected p-values, and effect sizes: per-seed KPIs are absent.
- Ablation, sensitivity, dispatcher-control, and scalability tables: these require new paired simulator runs.
- The original fairness-improvement percentages imply approximate Fog-only values, but reverse-engineering rounded claims is not a valid calculation and is intentionally excluded.

Populate the CSVs in `input_templates/` with the original simulation records to calculate these items.
