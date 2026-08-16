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

## Run

From this directory:

```powershell
$python = 'C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m unittest discover -s tests -v
& $python recompute.py
```

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
