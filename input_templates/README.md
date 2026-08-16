# Raw result inputs

Populate these CSVs with the original paired simulation outputs. Keep one observation per row and use the same seed, workload trace, and topology for every algorithm.

- `run_kpis.csv`: one row per seed/load/algorithm for uncertainty and paired statistical analysis.
- `node_counts.csv`: one row per fog node for Jain fairness.
- `prediction_records.csv`: aligned ground truth and one-step predictions for MAE, RMSE, and fractional MAPE.
- `task_records.csv`: task-level audit data for latency, deadline misses, throughput, energy, cost, rejection, and venue selection.

Do not enter manuscript means as if they were individual runs. Confidence intervals and paired tests require the original per-seed values.
