# AI-Enabled Digital Twin-Driven Scheduling Framework with Kalman-Based Prediction for Fog Computing

This work addresses efficient workload management under dynamic conditions and limited fog resources. Its core combines Digital Twin technology with Kalman-based prediction.

## Reproduction Package

This package implements the paper's deterministic venue-selection kernel and runs an independent, trace-backed benchmark for the prediction component. It also audits every result that can be checked from the PDF's displayed values.

It does **not** claim to reproduce the paper's exact Tables IV/V or XI/XII. The PDF omits the source commit, exact parameters, trace windows, preprocessing, run seeds, raw predictions, and run-level KPI records required for an exact reproduction.

## Included

- `dt_kf.py`: scalar Kalman, EMA and last-value predictors; Link Health Score; EDF queue estimate; constraints C1-C7; positive-maximum normalization; four-term objective; deterministic venue selection.
- `recompute.py`: dataset inventory, predictor tuning/evaluation, paper arithmetic audit, and generated report.
- `results/calculated_point_estimates.csv`: all 74 comparisons that are exactly calculable from the displayed tables.
- `results/calculation_status.md`: concise calculated-versus-missing result summary.
- `results/missing_data_requirements.csv`: the inputs required for cells and analyses that are not identifiable from aggregate values.
- `input_templates/`: CSV schemas for the original run, node, prediction, and task records.
- `example_config.json`: explicit illustrative scheduler values and benchmark grids. These are not represented as the paper's hidden settings.
- `tests/test_dt_kf.py`: unit and unit-convention checks.
- `results/`: generated CSV, JSON, and Markdown outputs.
- `experiment.py`: event-driven paired-seed simulator with EDF/FCFS dispatch, admission control, node accounting, and aligned prediction logging.
- `run_independent_experiments.py`: main, ablation, sensitivity, scalability, confidence-interval, paired-test, and effect-size calculations.
- `validate_independent_results.py`: independent reconciliation of reported KPIs against raw task, node, and prediction records.
- `results/independent_experiment/`: complete rerun outputs, including compressed raw records and publication-ready LaTeX tables.
- `results/excel/`: verified Excel workbooks for the complete results, per-seed records, and aligned raw predictions.

## Run

From this directory:

```powershell
$python = 'C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m unittest discover -s tests -v
& $python recompute.py
& $python run_independent_experiments.py
& $python validate_independent_results.py
```

For a quick pipeline smoke test, use `run_independent_experiments.py --quick`. The
full command runs 1,200 algorithm-seed scenarios and writes checkpoints as each
experiment family completes.

## Independent rerun

The independent experiment uses 30 paired seeds for the three-load main comparison,
20 paired seeds for ablation and sensitivity, and 10 paired seeds per scalability
configuration. Rejected tasks count as deadline misses, Jain fairness is computed
from per-node completed-task counts, and prediction errors use one-step-ahead aligned
ground truth and predictions.

The generated report is
`results/independent_experiment/independent_results_report.md`. It includes all nine
baseline Jain-fairness values, all high-load prediction cells, Student-t 95% CIs,
paired tests with Holm correction, effect sizes with bootstrap CIs, and the complete
ablation, sensitivity, and scalability tables. These values are new reproducible
simulation results; they are not presented as recovered values from undocumented
original runs.

The default data location is:

```text
C:\Users\Dell\Documents\Codex\2026-08-16\im\work\data_sources
```

To use another checkout:

```powershell
& $python recompute.py --data-root 'D:\research\data_sources' --ifogsim-root 'D:\research\iFogSim2'
```

## Dataset mapping

- Dataset A: `BuptMecMigration/Edge-Computing-Dataset` for MEC request times and byte volumes.
- Dataset B: `PhuLai/eua-dataset` for user and edge-server coordinates.
- Dataset C: `DataCenter-Traces-Datasets`, Alibaba 2018 10-second aggregate utilization trace.
- Dataset D in this benchmark: the final day of Dataset C, held out from tuning.

Dataset C is also packaged on PyPI as `datacentertracesdatasets`. The repository CSV is used directly here, so the benchmark requires no package installation beyond NumPy.

## Important unit correction

The paper declares task sizes in bytes and bandwidth in bits/s. Therefore, communication time is implemented as `8 * bytes / bits_per_second`. Equation 14 in the PDF omits the factor eight; using it literally would understate serialization time by 8x.
