from __future__ import annotations

import csv
import math
import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dt_kf import (
    LinkState,
    ScalarKalman,
    SchedulerConfig,
    Task,
    VenueState,
    ema_predictions,
    last_value_predictions,
    select_venue,
)


LoadLevel = Literal["low", "medium", "high"]
Policy = Literal["cost", "latency", "nearest_fog"]
Predictor = Literal["kalman", "ema", "last", "mean"]


@dataclass(frozen=True)
class ExperimentConfig:
    warmup_s: float = 20.0
    measurement_s: float = 90.0
    telemetry_step_s: float = 1.0
    main_seeds: int = 30
    secondary_seeds: int = 20
    scalability_seeds: int = 10
    fog_nodes: int = 8
    arrival_rates: tuple[tuple[str, float], ...] = (
        ("low", 5.0),
        ("medium", 10.0),
        ("high", 18.0),
    )
    load_centers: tuple[tuple[str, float], ...] = (
        ("low", 0.25),
        ("medium", 0.50),
        ("high", 0.75),
    )
    kalman_q: float = 0.001
    kalman_r: float = 0.01
    ema_alpha: float = 0.12
    mape_epsilon: float = 1e-6

    def rate(self, load: LoadLevel) -> float:
        return dict(self.arrival_rates)[load]

    def center(self, load: LoadLevel) -> float:
        return dict(self.load_centers)[load]


@dataclass(frozen=True)
class AlgorithmVariant:
    name: str
    predictor: Predictor
    policy: Policy
    use_dt_state: bool = True
    use_lhs: bool = True
    admission: bool = True
    queue_mode: Literal["edf", "fcfs", "none"] = "edf"
    allowed_kinds: tuple[str, ...] = ("local", "fog", "cloud")
    refresh_interval: int = 1
    price_multiplier: float = 1.0
    minimum_lhs: float = 0.67
    weights: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)


@dataclass(frozen=True)
class ExperimentData:
    payload_bytes: np.ndarray
    user_coordinates: np.ndarray
    site_coordinates: np.ndarray
    telemetry_signal: np.ndarray


@dataclass(frozen=True)
class SimTask:
    task: Task
    user_index: int


@dataclass(frozen=True)
class FogProfile:
    name: str
    latitude: float
    longitude: float
    base_mips: float
    energy_coefficient: float
    price_per_mi: float
    trust: float
    accuracy: float


@dataclass(frozen=True)
class Scenario:
    seed: int
    load: LoadLevel
    tasks: tuple[SimTask, ...]
    true_load: np.ndarray
    link_noise: np.ndarray
    fog_profiles: tuple[FogProfile, ...]
    user_coordinates: np.ndarray
    total_duration_s: float


@dataclass
class ScheduledJob:
    task: Task
    venue: VenueState
    service_arrival_s: float
    service_s: float
    downlink_s: float
    energy_j: float
    monetary_cost: float
    measured: bool
    order: int
    service_start_s: float = math.inf
    service_end_s: float = math.inf
    completion_s: float = math.inf


@dataclass
class NodeRuntime:
    jobs: list[ScheduledJob] | None = None

    def __post_init__(self) -> None:
        if self.jobs is None:
            self.jobs = []

    def prune(self, arrival_s: float) -> None:
        self.jobs = [job for job in self.jobs or [] if job.service_end_s > arrival_s]

    def schedule(self, job: ScheduledJob, arrival_s: float, mode: str) -> None:
        self.prune(arrival_s)
        if mode == "fcfs":
            previous_end = max(
                (item.service_end_s for item in self.jobs or []),
                default=arrival_s,
            )
            job.service_start_s = max(previous_end, job.service_arrival_s)
            job.service_end_s = job.service_start_s + job.service_s
            job.completion_s = job.service_end_s + job.downlink_s
            self.jobs.append(job)
            return
        self.jobs.append(job)
        running = [
            item
            for item in self.jobs
            if item.service_start_s < arrival_s < item.service_end_s
        ]
        queued = [item for item in self.jobs if item not in running]
        cursor = max((item.service_end_s for item in running), default=arrival_s)
        while queued:
            available = [item for item in queued if item.service_arrival_s <= cursor]
            if not available:
                cursor = min(item.service_arrival_s for item in queued)
                available = [item for item in queued if item.service_arrival_s <= cursor]
            item = min(available, key=lambda value: (value.task.deadline_s, value.order))
            queued.remove(item)
            item.service_start_s = cursor
            item.service_end_s = item.service_start_s + item.service_s
            item.completion_s = item.service_end_s + item.downlink_s
            cursor = item.service_end_s


def load_experiment_data(data_root: Path, max_payloads: int = 80_000) -> ExperimentData:
    edge_root = data_root / "edge-computing-dataset" / "Data"
    payloads: list[tuple[float, float]] = []
    for path in sorted(edge_root.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            for row in csv.reader(handle):
                if len(row) <= 11:
                    continue
                try:
                    uplink = max(float(row[10] or 0.0), 0.0)
                    downlink = max(float(row[11] or 0.0), 0.0)
                except ValueError:
                    continue
                if uplink + downlink > 0:
                    payloads.append((uplink, downlink))
                if len(payloads) >= max_payloads:
                    break
        if len(payloads) >= max_payloads:
            break
    if not payloads:
        raise FileNotFoundError("Dataset A payload records were not found")

    users_path = data_root / "eua-dataset" / "users" / "users-melbcbd-generated.csv"
    sites_path = data_root / "eua-dataset" / "edge-servers" / "site-optus-melbCBD.csv"
    users = _read_coordinates(users_path, "Latitude", "Longitude")
    sites = _read_coordinates(sites_path, "LATITUDE", "LONGITUDE")

    telemetry_path = (
        data_root
        / "datacenter-traces-datasets"
        / "alibaba2018"
        / "machine_usage_days_1_to_8_grouped_10_seconds.csv"
    )
    telemetry = np.genfromtxt(telemetry_path, delimiter=",", names=True, dtype=float)
    cpu = np.asarray(telemetry["cpu_util_percent"], dtype=float) / 100.0
    net_in = _robust_unit_scale(np.asarray(telemetry["net_in"], dtype=float))
    net_out = _robust_unit_scale(np.asarray(telemetry["net_out"], dtype=float))
    signal = np.clip(0.60 * cpu + 0.20 * net_in + 0.20 * net_out, 0.0, 1.0)
    signal = signal[np.isfinite(signal)]
    if len(signal) < 1000:
        raise ValueError("Dataset C does not contain enough valid telemetry rows")

    return ExperimentData(
        payload_bytes=np.asarray(payloads, dtype=float),
        user_coordinates=users,
        site_coordinates=sites,
        telemetry_signal=signal,
    )


def _read_coordinates(path: Path, latitude: str, longitude: str) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((float(row[latitude]), float(row[longitude])))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise FileNotFoundError(f"No valid coordinates in {path}")
    return np.asarray(rows, dtype=float)


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    valid = values[np.isfinite(values)]
    low, high = np.percentile(valid, [5.0, 95.0])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def main_algorithms() -> tuple[AlgorithmVariant, ...]:
    return (
        AlgorithmVariant("DT-KF-CostAware", "kalman", "cost"),
        AlgorithmVariant("DT-OPT", "ema", "cost"),
        AlgorithmVariant(
            "SemiGreedy",
            "last",
            "latency",
            use_lhs=False,
            admission=False,
            queue_mode="fcfs",
        ),
        AlgorithmVariant(
            "Fog-only",
            "mean",
            "nearest_fog",
            use_dt_state=True,
            use_lhs=False,
            admission=False,
            queue_mode="fcfs",
            allowed_kinds=("fog",),
        ),
    )


def ablation_algorithms() -> tuple[AlgorithmVariant, ...]:
    full = AlgorithmVariant("Full", "kalman", "cost")
    return (
        full,
        replace(full, name="No-Kalman", predictor="last"),
        replace(full, name="No-DT-state", predictor="mean", use_dt_state=False, queue_mode="none"),
        replace(full, name="No-LHS", use_lhs=False, minimum_lhs=0.0),
        replace(full, name="No-cost", weights=(1 / 3, 1 / 3, 0.0, 1 / 3)),
        replace(full, name="No-admission", admission=False, use_lhs=False, minimum_lhs=0.0),
        replace(full, name="No-EDF", queue_mode="fcfs"),
    )


def sensitivity_algorithms() -> tuple[AlgorithmVariant, ...]:
    full = AlgorithmVariant("Reference", "kalman", "cost")
    variants = [full]
    for interval in (2, 5, 15, 30):
        variants.append(replace(full, name=f"Refresh-{interval}s", refresh_interval=interval))
    variants.extend(
        [
            replace(full, name="Latency-dominant", weights=(0.55, 0.15, 0.15, 0.15)),
            replace(full, name="Energy-dominant", weights=(0.15, 0.55, 0.15, 0.15)),
            replace(full, name="Cost-dominant", weights=(0.15, 0.15, 0.55, 0.15)),
            replace(full, name="Accuracy-dominant", weights=(0.15, 0.15, 0.15, 0.55)),
        ]
    )
    for multiplier in (0.5, 2.0):
        variants.append(replace(full, name=f"Price-x{multiplier:g}", price_multiplier=multiplier))
    for threshold in (0.30, 0.70):
        variants.append(replace(full, name=f"LHS-{threshold:.2f}", minimum_lhs=threshold))
    return tuple(variants)


def generate_scenario(
    data: ExperimentData,
    config: ExperimentConfig,
    load: LoadLevel,
    seed: int,
    *,
    fog_count: int | None = None,
    fixed_task_count: int | None = None,
) -> Scenario:
    rng = np.random.default_rng(seed)
    total_duration_s = config.warmup_s + config.measurement_s
    telemetry_points = int(math.ceil(total_duration_s / config.telemetry_step_s)) + 2
    start = int(rng.integers(0, len(data.telemetry_signal) - telemetry_points - 1))
    raw_signal = data.telemetry_signal[start : start + telemetry_points]
    centered = raw_signal - float(np.mean(raw_signal))
    variability = 0.45 + 0.55 * config.center(load)
    true_load = np.clip(
        config.center(load)
        + variability * centered
        + rng.normal(0.0, 0.018 + 0.018 * config.center(load), telemetry_points),
        0.05,
        0.95,
    )

    fog_count = fog_count or config.fog_nodes
    if fog_count > len(data.site_coordinates):
        raise ValueError("Requested more fog nodes than Dataset B provides")
    site_indices = rng.choice(len(data.site_coordinates), size=fog_count, replace=False)
    fog_profiles = tuple(
        FogProfile(
            name=f"fog-{index + 1}",
            latitude=float(data.site_coordinates[site_index, 0]),
            longitude=float(data.site_coordinates[site_index, 1]),
            base_mips=float(rng.uniform(3200.0, 5200.0)),
            energy_coefficient=float(rng.uniform(1.5e-11, 2.5e-11)),
            price_per_mi=float(rng.uniform(0.8e-5, 1.4e-5)),
            trust=float(rng.uniform(0.82, 0.98)),
            accuracy=float(rng.uniform(0.88, 0.96)),
        )
        for index, site_index in enumerate(site_indices)
    )
    link_noise = np.clip(rng.normal(0.0, 0.07, (telemetry_points, fog_count)), -0.20, 0.20)

    if fixed_task_count is None:
        arrivals: list[float] = []
        arrival = 0.0
        while arrival < total_duration_s:
            arrival += float(rng.exponential(1.0 / config.rate(load)))
            if arrival < total_duration_s:
                arrivals.append(arrival)
    else:
        arrivals = np.linspace(
            config.warmup_s,
            total_duration_s - 1e-6,
            fixed_task_count,
            dtype=float,
        ).tolist()

    tasks: list[SimTask] = []
    for index, arrival_s in enumerate(arrivals):
        payload_index = int(rng.integers(0, len(data.payload_bytes)))
        uplink, downlink = data.payload_bytes[payload_index]
        uplink = float(np.clip(uplink, 64.0, 2_000_000.0))
        downlink = float(np.clip(downlink, 32.0, 4_000_000.0))
        total_bytes = uplink + downlink
        compute_mi = float(np.clip(300.0 + 10.0 * math.sqrt(total_bytes), 300.0, 4500.0))
        relative_deadline = float(
            np.clip((0.14 + compute_mi / 2200.0) * rng.uniform(0.85, 1.35), 0.22, 2.6)
        )
        inference = bool(rng.random() < 0.55)
        task = Task(
            task_id=f"s{seed}-{load}-{index}",
            arrival_s=float(arrival_s),
            deadline_s=float(arrival_s + relative_deadline),
            compute_mi=compute_mi,
            uplink_bytes=uplink,
            downlink_bytes=downlink,
            min_accuracy=float(rng.uniform(0.84, 0.91)) if inference else 0.0,
            min_trust=float(rng.uniform(0.72, 0.86)),
            energy_budget_j=float(rng.uniform(0.5, 1.6)),
            inference=inference,
        )
        tasks.append(SimTask(task, int(rng.integers(0, len(data.user_coordinates)))))

    return Scenario(
        seed=seed,
        load=load,
        tasks=tuple(tasks),
        true_load=true_load,
        link_noise=link_noise,
        fog_profiles=fog_profiles,
        user_coordinates=data.user_coordinates,
        total_duration_s=total_duration_s,
    )


def predictor_series(
    true_load: np.ndarray,
    variant: AlgorithmVariant,
    config: ExperimentConfig,
) -> np.ndarray:
    if variant.predictor == "kalman":
        base = ScalarKalman(config.kalman_q, config.kalman_r).one_step_predictions(true_load)
    elif variant.predictor == "ema":
        base = ema_predictions(true_load, config.ema_alpha)
    elif variant.predictor == "last":
        base = last_value_predictions(true_load)
    else:
        base = np.full_like(true_load, float(np.mean(true_load[: min(10, len(true_load))])))
    base[0] = float(np.mean(true_load[: min(10, len(true_load))]))
    interval = max(int(variant.refresh_interval), 1)
    if interval > 1:
        held = np.empty_like(base)
        for index in range(len(base)):
            refresh_index = (index // interval) * interval
            held[index] = base[refresh_index]
        base = held
    return np.clip(base, 0.02, 0.98)


def simulate_run(
    scenario: Scenario,
    variant: AlgorithmVariant,
    config: ExperimentConfig,
    *,
    record_tasks: bool = False,
    record_predictions: bool = False,
    measure_memory: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predicted_load = predictor_series(scenario.true_load, variant, config)
    runtimes: dict[str, NodeRuntime] = {"local": NodeRuntime(), "cloud": NodeRuntime()}
    runtimes.update({profile.name: NodeRuntime() for profile in scenario.fog_profiles})
    scheduled_jobs: list[ScheduledJob] = []
    rejected_tasks: list[Task] = []
    decision_times_us: list[float] = []

    if measure_memory:
        tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for order, sim_task in enumerate(scenario.tasks):
        task = sim_task.task
        telemetry_index = min(
            int(task.arrival_s / config.telemetry_step_s), len(scenario.true_load) - 1
        )
        current_load = float(scenario.true_load[telemetry_index])
        estimate = float(predicted_load[telemetry_index])
        user = scenario.user_coordinates[sim_task.user_index]
        for runtime in runtimes.values():
            runtime.prune(task.arrival_s)

        predicted_venues = _build_venues(
            task,
            user,
            scenario,
            runtimes,
            telemetry_index,
            estimate,
            variant,
            actual=False,
        )
        decision_start = time.perf_counter_ns()
        selected = _choose_venue(task, predicted_venues, variant)
        decision_times_us.append((time.perf_counter_ns() - decision_start) / 1000.0)
        measured = task.arrival_s >= config.warmup_s
        if selected is None:
            if measured:
                rejected_tasks.append(task)
            continue

        true_venues = _build_venues(
            task,
            user,
            scenario,
            runtimes,
            telemetry_index,
            current_load,
            variant,
            actual=True,
        )
        actual = next(venue for venue in true_venues if venue.name == selected)
        communication_s, uplink_s, downlink_s = _communication_times(task, actual)
        venue_energy = actual.energy_coefficient * task.compute_mi * actual.mips**2
        device_energy = 0.0
        if actual.kind != "local" and actual.link is not None:
            tx_s = max(uplink_s - actual.link.uplink_latency_s, 0.0)
            rx_s = max(downlink_s - actual.link.downlink_latency_s, 0.0)
            device_energy = 1.3 * tx_s + 0.9 * rx_s
        job = ScheduledJob(
            task=task,
            venue=actual,
            service_arrival_s=task.arrival_s + communication_s - downlink_s,
            service_s=task.compute_mi / actual.mips,
            downlink_s=downlink_s,
            energy_j=venue_energy + device_energy,
            monetary_cost=actual.price_per_mi * task.compute_mi,
            measured=measured,
            order=order,
        )
        runtimes[selected].schedule(
            job,
            task.arrival_s,
            "edf" if variant.queue_mode == "edf" else "fcfs",
        )
        scheduled_jobs.append(job)

    wall_s = max(time.perf_counter() - wall_start, 1e-9)
    cpu_s = max(time.process_time() - cpu_start, 0.0)
    peak_memory_kib = math.nan
    if measure_memory:
        _, peak_memory = tracemalloc.get_traced_memory()
        peak_memory_kib = peak_memory / 1024.0
        tracemalloc.stop()

    measured_jobs = [job for job in scheduled_jobs if job.measured]
    arrivals = len(measured_jobs) + len(rejected_tasks)
    completions = sum(job.completion_s <= scenario.total_duration_s for job in measured_jobs)
    misses = len(rejected_tasks) + sum(
        job.completion_s > job.task.deadline_s for job in measured_jobs
    )
    latency_array = np.asarray(
        [1000.0 * (job.completion_s - job.task.arrival_s) for job in measured_jobs],
        dtype=float,
    )
    total_energy_j = sum(job.energy_j for job in measured_jobs)
    total_monetary_cost = sum(job.monetary_cost for job in measured_jobs)
    venue_counts = {"local": 0, "fog": 0, "cloud": 0, "rejected": len(rejected_tasks)}
    fog_counts = {profile.name: 0 for profile in scenario.fog_profiles}
    realized_objectives = [1.0] * len(rejected_tasks)
    for job in measured_jobs:
        venue_counts[job.venue.kind] += 1
        if job.venue.kind == "fog" and job.completion_s <= scenario.total_duration_s:
            fog_counts[job.venue.name] += 1
        latency_s = job.completion_s - job.task.arrival_s
        accuracy_loss = 1.0 - job.venue.predicted_accuracy if job.task.inference else 0.0
        realized_objectives.append(
            float(
                np.dot(
                    np.asarray(variant.weights),
                    np.asarray(
                        [
                            min(latency_s / job.task.relative_deadline_s, 1.0),
                            min(job.energy_j / 1.5, 1.0),
                            min(job.monetary_cost / 0.08, 1.0),
                            min(max(accuracy_loss, 0.0) / 0.20, 1.0),
                        ]
                    ),
                )
            )
        )

    task_records: list[dict[str, Any]] = []
    if record_tasks:
        task_records.extend(
            _task_record(scenario, variant, task, None, None, True, True, 0.0, 0.0)
            for task in rejected_tasks
        )
        task_records.extend(
            _task_record(
                scenario,
                variant,
                job.task,
                job.venue,
                job.completion_s,
                job.completion_s > job.task.deadline_s,
                False,
                job.energy_j,
                job.monetary_cost,
            )
            for job in measured_jobs
        )

    fairness = jain_fairness(list(fog_counts.values()))
    run = {
        "seed": scenario.seed,
        "load": scenario.load,
        "algorithm": variant.name,
        "arrived_tasks": arrivals,
        "completed_tasks": completions,
        "mean_latency_ms": float(np.mean(latency_array)) if len(latency_array) else math.nan,
        "p95_latency_ms": float(np.percentile(latency_array, 95.0)) if len(latency_array) else math.nan,
        "dmr_pct": 100.0 * misses / arrivals if arrivals else math.nan,
        "throughput_tasks_s": completions / config.measurement_s,
        "energy_j": total_energy_j,
        "monetary_cost": total_monetary_cost,
        "run_cost_index": float(np.mean(realized_objectives)) if realized_objectives else math.nan,
        "fairness": fairness,
        "rejection_rate_pct": 100.0 * len(rejected_tasks) / arrivals if arrivals else math.nan,
        "local_pct": 100.0 * venue_counts["local"] / arrivals if arrivals else math.nan,
        "fog_pct": 100.0 * venue_counts["fog"] / arrivals if arrivals else math.nan,
        "cloud_pct": 100.0 * venue_counts["cloud"] / arrivals if arrivals else math.nan,
        "decision_median_us": float(np.median(decision_times_us)),
        "decision_p95_us": float(np.percentile(decision_times_us, 95.0)),
        "decision_p99_us": float(np.percentile(decision_times_us, 99.0)),
        "cpu_s": cpu_s,
        "cpu_utilization_pct": 100.0 * cpu_s / wall_s,
        "wall_s": wall_s,
        "peak_memory_kib": peak_memory_kib,
        "simulator_throughput_tasks_s": len(scenario.tasks) / wall_s,
    }
    node_rows = [
        {
            "seed": scenario.seed,
            "load": scenario.load,
            "algorithm": variant.name,
            "fog_node_id": name,
            "completed_tasks": count,
        }
        for name, count in fog_counts.items()
    ]
    prediction_rows: list[dict[str, Any]] = []
    if record_predictions:
        for index, (actual, predicted) in enumerate(zip(scenario.true_load, predicted_load)):
            prediction_rows.append(
                {
                    "seed": scenario.seed,
                    "load": scenario.load,
                    "algorithm": variant.name,
                    "segment_id": scenario.seed,
                    "timestamp_s": index * config.telemetry_step_s,
                    "y_true": float(actual),
                    "y_pred": float(predicted),
                }
            )
    return run, node_rows, task_records, prediction_rows


def _choose_venue(
    task: Task,
    venues: list[VenueState],
    variant: AlgorithmVariant,
) -> str | None:
    allowed = [venue for venue in venues if venue.kind in variant.allowed_kinds]
    if not allowed:
        return None
    if variant.policy == "nearest_fog":
        return min(
            allowed,
            key=lambda venue: (
                _communication_times(task, venue)[0]
                + sum(work for _, work in venue.queue_work_mi) / venue.mips
                + task.compute_mi / venue.mips
            ),
        ).name

    effective_task = task
    if not variant.admission:
        effective_task = replace(
            task,
            deadline_s=task.arrival_s + 1e9,
            min_accuracy=0.0,
            min_trust=0.0,
            energy_budget_j=math.inf,
        )
    weights = variant.weights if variant.policy == "cost" else (1.0, 0.0, 0.0, 0.0)
    scheduler_config = SchedulerConfig(
        latency_weight=weights[0],
        energy_weight=weights[1],
        cost_weight=weights[2],
        accuracy_weight=weights[3],
        minimum_bandwidth_bps=800_000.0,
        minimum_lhs=variant.minimum_lhs if variant.use_lhs else 0.0,
        minimum_battery_fraction=0.08,
        reference_uplink_bps=25_000_000.0,
        reference_downlink_bps=50_000_000.0,
        reference_uplink_latency_s=0.060,
        reference_downlink_latency_s=0.060,
    )
    decision = select_venue(effective_task, allowed, scheduler_config)
    return decision.selected_venue


def _build_venues(
    task: Task,
    user: np.ndarray,
    scenario: Scenario,
    runtimes: dict[str, NodeRuntime],
    telemetry_index: int,
    load_value: float,
    variant: AlgorithmVariant,
    *,
    actual: bool,
) -> list[VenueState]:
    venues = [
        VenueState(
            name="local",
            kind="local",
            mips=max(1100.0 * (1.0 - 0.35 * load_value), 500.0),
            energy_coefficient=2.1e-10,
            price_per_mi=0.0,
            trust=0.99,
            predicted_accuracy=0.85,
            queue_work_mi=_queue_work(task, runtimes["local"], 1100.0, variant),
            battery_fraction=0.65,
        )
    ]
    for fog_index, profile in enumerate(scenario.fog_profiles):
        distance_km = haversine_km(
            float(user[0]), float(user[1]), profile.latitude, profile.longitude
        )
        noise = float(scenario.link_noise[telemetry_index, fog_index]) if actual else 0.0
        congestion_factor = max(0.10, 1.0 - 0.72 * load_value + noise)
        uplink_bps = max(32_000_000.0 * congestion_factor / (1.0 + 0.04 * distance_km), 500_000.0)
        downlink_bps = max(64_000_000.0 * congestion_factor / (1.0 + 0.04 * distance_km), 800_000.0)
        latency = (0.003 + 0.0007 * distance_km) * (1.0 + 2.8 * load_value - noise)
        mips = max(profile.base_mips * (1.0 - 0.68 * load_value), profile.base_mips * 0.20)
        venues.append(
            VenueState(
                name=profile.name,
                kind="fog",
                mips=mips,
                energy_coefficient=profile.energy_coefficient,
                price_per_mi=profile.price_per_mi * variant.price_multiplier,
                trust=profile.trust,
                predicted_accuracy=profile.accuracy,
                queue_work_mi=_queue_work(task, runtimes[profile.name], mips, variant),
                link=LinkState(uplink_bps, downlink_bps, latency, latency),
            )
        )
    cloud_factor = max(0.18, 1.0 - 0.52 * load_value)
    cloud_mips = max(15_000.0 * (1.0 - 0.45 * load_value), 7000.0)
    cloud_latency = 0.038 * (1.0 + 1.9 * load_value)
    venues.append(
        VenueState(
            name="cloud",
            kind="cloud",
            mips=cloud_mips,
            energy_coefficient=5.0e-12,
            price_per_mi=3.2e-5 * variant.price_multiplier,
            trust=0.985,
            predicted_accuracy=0.975,
            queue_work_mi=_queue_work(task, runtimes["cloud"], cloud_mips, variant),
            link=LinkState(
                85_000_000.0 * cloud_factor,
                150_000_000.0 * cloud_factor,
                cloud_latency,
                cloud_latency,
            ),
        )
    )
    return venues


def _queue_work(
    task: Task,
    runtime: NodeRuntime,
    predicted_mips: float,
    variant: AlgorithmVariant,
) -> tuple[tuple[float, float], ...]:
    if not variant.use_dt_state or variant.queue_mode == "none":
        return ()
    jobs = runtime.jobs or []
    if variant.queue_mode == "edf":
        running_work = sum(
            max(job.service_end_s - task.arrival_s, 0.0) * predicted_mips
            for job in jobs
            if job.service_start_s < task.arrival_s < job.service_end_s
        )
        queued_work = sum(
            job.task.compute_mi
            for job in jobs
            if job.service_start_s >= task.arrival_s and job.task.deadline_s < task.deadline_s
        )
        work = running_work + queued_work
    else:
        work = max(
            max((job.service_end_s for job in jobs), default=task.arrival_s)
            - task.arrival_s,
            0.0,
        ) * predicted_mips
    return ((task.deadline_s - 1e-9, work),) if work > 0 else ()


def _communication_times(task: Task, venue: VenueState) -> tuple[float, float, float]:
    if venue.kind == "local" or venue.link is None:
        return 0.0, 0.0, 0.0
    uplink = 8.0 * task.uplink_bytes / venue.link.uplink_bps + venue.link.uplink_latency_s
    downlink = 8.0 * task.downlink_bytes / venue.link.downlink_bps + venue.link.downlink_latency_s
    return uplink + downlink, uplink, downlink


def _task_record(
    scenario: Scenario,
    variant: AlgorithmVariant,
    task: Task,
    venue: VenueState | None,
    completion_s: float | None,
    missed: bool,
    rejected: bool,
    energy_j: float,
    monetary_cost: float,
) -> dict[str, Any]:
    return {
        "seed": scenario.seed,
        "load": scenario.load,
        "algorithm": variant.name,
        "task_id": task.task_id,
        "arrival_s": task.arrival_s,
        "completion_s": completion_s if completion_s is not None else "",
        "relative_deadline_s": task.relative_deadline_s,
        "missed": int(missed),
        "rejected": int(rejected),
        "energy_j": energy_j,
        "monetary_cost": monetary_cost,
        "selected_venue": venue.name if venue is not None else "",
        "venue_kind": venue.kind if venue is not None else "rejected",
    }


def prediction_metrics(rows: list[dict[str, Any]], epsilon: float = 1e-6) -> dict[str, float]:
    actual = np.asarray([float(row["y_true"]) for row in rows], dtype=float)
    predicted = np.asarray([float(row["y_pred"]) for row in rows], dtype=float)
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float(np.mean(np.abs(error) / (np.abs(actual) + epsilon))),
    }


def jain_fairness(counts: list[int] | np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    denominator = len(values) * float(np.sum(np.square(values)))
    if denominator <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / denominator)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(value))
