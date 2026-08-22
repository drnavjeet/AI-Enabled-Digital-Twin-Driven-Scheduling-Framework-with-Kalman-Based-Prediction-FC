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
    expected_retransmission_factor,
    last_value_predictions,
    select_venue,
)


LoadLevel = Literal["low", "medium", "high"]
QoSRegime = Literal["none", "clean", "moderate", "impaired"]
Policy = Literal["cost", "latency", "nearest_fog", "drl_oo"]
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
    packet_payload_bytes: int = 1460
    max_network_retries: int = 3

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
    qos_aware: bool = False
    loss_aware: bool = True
    jitter_aware: bool = True


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
    packet_loss: np.ndarray
    jitter_s: np.ndarray
    qos_regime: QoSRegime
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
    packet_loss_rate: float = 0.0
    jitter_s: float = 0.0
    retransmission_bytes: float = 0.0
    network_failed: bool = False
    uplink_serialization_s: float = 0.0
    downlink_serialization_s: float = 0.0


@dataclass(frozen=True)
class NetworkTransfer:
    communication_s: float
    uplink_s: float
    downlink_s: float
    uplink_serialization_s: float
    downlink_serialization_s: float
    jitter_s: float
    retransmission_bytes: float
    failed: bool


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
        AlgorithmVariant("DT-KF-CostAware", "kalman", "cost", qos_aware=True),
        AlgorithmVariant("DT-OPT", "ema", "cost", qos_aware=True),
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


def qos_algorithms() -> tuple[AlgorithmVariant, ...]:
    """Main reviewer-revision algorithms, including the 2025 DRL-OO adaptation."""
    revised = tuple(
        replace(item, minimum_lhs=0.30) if item.qos_aware else item
        for item in main_algorithms()
    )
    return revised + (
        AlgorithmVariant(
            "DRL-OO-2025",
            "last",
            "drl_oo",
            use_dt_state=True,
            use_lhs=False,
            admission=True,
            queue_mode="edf",
            qos_aware=True,
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
    qos_regime: QoSRegime = "none",
    qos_loss_rate: float | None = None,
    qos_jitter_s: float | None = None,
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

    qos_defaults = {
        "none": (0.0, 0.0),
        "clean": (0.001, 0.0015),
        "moderate": (0.015, 0.008),
        "impaired": (0.050, 0.025),
    }
    base_loss, base_jitter = qos_defaults[qos_regime]
    if qos_loss_rate is not None:
        base_loss = qos_loss_rate
    if qos_jitter_s is not None:
        base_jitter = qos_jitter_s
    if not 0.0 <= base_loss < 1.0 or base_jitter < 0.0:
        raise ValueError("QoS loss and jitter parameters must be non-negative and loss < 1")

    qos_nodes = fog_count + 1  # Fog links plus the cloud link.
    temporal_noise = rng.normal(0.0, 0.22, (telemetry_points, qos_nodes))
    for index in range(1, telemetry_points):
        temporal_noise[index] = 0.72 * temporal_noise[index - 1] + 0.28 * temporal_noise[index]
    node_factors = rng.uniform(0.82, 1.18, qos_nodes)
    node_factors[-1] *= 0.80
    load_multiplier = (0.55 + 0.90 * true_load)[:, None]
    packet_loss = np.clip(
        base_loss * load_multiplier * node_factors[None, :] * np.exp(temporal_noise),
        0.0,
        0.20,
    )
    jitter_factors = rng.uniform(0.82, 1.18, qos_nodes)
    jitter_factors[-1] *= 1.20
    jitter_s = np.clip(
        base_jitter
        * load_multiplier
        * jitter_factors[None, :]
        * np.exp(0.75 * temporal_noise),
        0.0,
        0.15,
    )

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
        packet_loss=packet_loss,
        jitter_s=jitter_s,
        qos_regime=qos_regime,
        fog_profiles=fog_profiles,
        user_coordinates=data.user_coordinates,
        total_duration_s=total_duration_s,
    )


def predictor_series(
    true_load: np.ndarray,
    variant: AlgorithmVariant,
    config: ExperimentConfig,
    *,
    lower: float = 0.02,
    upper: float = 0.98,
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
    return np.clip(base, lower, upper)


def predictor_matrix(
    values: np.ndarray,
    variant: AlgorithmVariant,
    config: ExperimentConfig,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Apply the selected strictly one-step-ahead predictor independently per link."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a telemetry-by-link matrix")
    predicted = np.column_stack(
        [
            predictor_series(
                matrix[:, index], variant, config, lower=lower, upper=upper
            )
            for index in range(matrix.shape[1])
        ]
    )
    return np.clip(predicted, lower, upper)


def simulate_run(
    scenario: Scenario,
    variant: AlgorithmVariant,
    config: ExperimentConfig,
    *,
    record_tasks: bool = False,
    record_predictions: bool = False,
    record_qos_predictions: bool = False,
    measure_memory: bool = False,
    policy_model: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predicted_load = predictor_series(scenario.true_load, variant, config)
    if variant.qos_aware:
        predicted_loss = predictor_matrix(
            scenario.packet_loss, variant, config, lower=0.0, upper=0.20
        )
        predicted_jitter = predictor_matrix(
            scenario.jitter_s, variant, config, lower=0.0, upper=0.15
        )
        if not variant.loss_aware:
            predicted_loss = np.zeros_like(scenario.packet_loss)
        if not variant.jitter_aware:
            predicted_jitter = np.zeros_like(scenario.jitter_s)
    else:
        predicted_loss = np.zeros_like(scenario.packet_loss)
        predicted_jitter = np.zeros_like(scenario.jitter_s)
    runtimes: dict[str, NodeRuntime] = {"local": NodeRuntime(), "cloud": NodeRuntime()}
    runtimes.update({profile.name: NodeRuntime() for profile in scenario.fog_profiles})
    scheduled_jobs: list[ScheduledJob] = []
    rejected_tasks: list[Task] = []
    network_failed_tasks: list[tuple[Task, VenueState, NetworkTransfer]] = []
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
            qos_loss=predicted_loss[telemetry_index],
            qos_jitter=predicted_jitter[telemetry_index],
        )
        decision_start = time.perf_counter_ns()
        selected = _choose_venue(
            task,
            predicted_venues,
            variant,
            policy_model=policy_model,
            decision_seed=scenario.seed * 1_000_003 + order,
        )
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
            qos_loss=scenario.packet_loss[telemetry_index],
            qos_jitter=scenario.jitter_s[telemetry_index],
        )
        actual = next(venue for venue in true_venues if venue.name == selected)
        transfer = _realized_network_transfer(
            task,
            actual,
            config,
            scenario_seed=scenario.seed,
            task_order=order,
        )
        if transfer.failed:
            if measured:
                network_failed_tasks.append((task, actual, transfer))
            continue
        venue_energy = actual.energy_coefficient * task.compute_mi * actual.mips**2
        device_energy = 0.0
        if actual.kind != "local" and actual.link is not None:
            device_energy = (
                1.3 * transfer.uplink_serialization_s
                + 0.9 * transfer.downlink_serialization_s
            )
        job = ScheduledJob(
            task=task,
            venue=actual,
            service_arrival_s=task.arrival_s + transfer.uplink_s,
            service_s=task.compute_mi / actual.mips,
            downlink_s=transfer.downlink_s,
            energy_j=venue_energy + device_energy,
            monetary_cost=actual.price_per_mi * task.compute_mi,
            measured=measured,
            order=order,
            packet_loss_rate=actual.link.packet_loss_rate if actual.link is not None else 0.0,
            jitter_s=transfer.jitter_s,
            retransmission_bytes=transfer.retransmission_bytes,
            uplink_serialization_s=transfer.uplink_serialization_s,
            downlink_serialization_s=transfer.downlink_serialization_s,
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
    arrivals = len(measured_jobs) + len(rejected_tasks) + len(network_failed_tasks)
    completions = sum(job.completion_s <= scenario.total_duration_s for job in measured_jobs)
    misses = len(rejected_tasks) + len(network_failed_tasks) + sum(
        job.completion_s > job.task.deadline_s for job in measured_jobs
    )
    latency_array = np.asarray(
        [1000.0 * (job.completion_s - job.task.arrival_s) for job in measured_jobs],
        dtype=float,
    )
    failed_energy_j = sum(
        1.3 * transfer.uplink_serialization_s + 0.9 * transfer.downlink_serialization_s
        for _, _, transfer in network_failed_tasks
    )
    total_energy_j = sum(job.energy_j for job in measured_jobs) + failed_energy_j
    total_monetary_cost = sum(job.monetary_cost for job in measured_jobs)
    venue_counts = {
        "local": 0,
        "fog": 0,
        "cloud": 0,
        "rejected": len(rejected_tasks),
        "network_failed": len(network_failed_tasks),
    }
    fog_counts = {profile.name: 0 for profile in scenario.fog_profiles}
    realized_objectives = [1.0] * (len(rejected_tasks) + len(network_failed_tasks))
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

    remote_jobs = [job for job in measured_jobs if job.venue.link is not None]
    qos_loss_values = [job.packet_loss_rate for job in remote_jobs] + [
        venue.link.packet_loss_rate
        for _, venue, _ in network_failed_tasks
        if venue.link is not None
    ]
    qos_jitter_values = [job.jitter_s for job in remote_jobs] + [
        transfer.jitter_s for _, _, transfer in network_failed_tasks
    ]
    retransmission_bytes = sum(job.retransmission_bytes for job in remote_jobs) + sum(
        transfer.retransmission_bytes for _, _, transfer in network_failed_tasks
    )
    original_network_bytes = sum(
        job.task.uplink_bytes + job.task.downlink_bytes for job in remote_jobs
    ) + sum(
        task.uplink_bytes + task.downlink_bytes for task, _, _ in network_failed_tasks
    )
    retransmission_energy_j = 0.0
    for job in remote_jobs:
        link = job.venue.link
        if link is None:
            continue
        base_up = 8.0 * job.task.uplink_bytes / link.uplink_bps
        base_down = 8.0 * job.task.downlink_bytes / link.downlink_bps
        retransmission_energy_j += (
            1.3 * max(job.uplink_serialization_s - base_up, 0.0)
            + 0.9 * max(job.downlink_serialization_s - base_down, 0.0)
        )
    for task, venue, transfer in network_failed_tasks:
        link = venue.link
        if link is None:
            continue
        base_up = 8.0 * task.uplink_bytes / link.uplink_bps
        base_down = 8.0 * task.downlink_bytes / link.downlink_bps
        retransmission_energy_j += (
            1.3 * max(transfer.uplink_serialization_s - base_up, 0.0)
            + 0.9 * max(transfer.downlink_serialization_s - base_down, 0.0)
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
                packet_loss_rate=job.packet_loss_rate,
                jitter_s=job.jitter_s,
                retransmission_bytes=job.retransmission_bytes,
            )
            for job in measured_jobs
        )
        task_records.extend(
            _task_record(
                scenario,
                variant,
                task,
                venue,
                None,
                True,
                False,
                1.3 * transfer.uplink_serialization_s
                + 0.9 * transfer.downlink_serialization_s,
                0.0,
                network_failed=True,
                transfer=transfer,
            )
            for task, venue, transfer in network_failed_tasks
        )

    fairness = jain_fairness(list(fog_counts.values()))
    run = {
        "seed": scenario.seed,
        "load": scenario.load,
        "qos_regime": scenario.qos_regime,
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
        "network_failure_rate_pct": 100.0 * len(network_failed_tasks) / arrivals if arrivals else math.nan,
        "mean_packet_loss_pct": 100.0 * float(np.mean(qos_loss_values)) if qos_loss_values else 0.0,
        "mean_jitter_ms": 1000.0 * float(np.mean(qos_jitter_values)) if qos_jitter_values else 0.0,
        "p95_jitter_ms": 1000.0 * float(np.percentile(qos_jitter_values, 95.0)) if qos_jitter_values else 0.0,
        "retransmission_bytes": retransmission_bytes,
        "retransmission_overhead_pct": 100.0 * retransmission_bytes / original_network_bytes if original_network_bytes else 0.0,
        "retransmission_energy_j": retransmission_energy_j,
        "sla_success_pct": 100.0 * (arrivals - misses) / arrivals if arrivals else math.nan,
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
            "qos_regime": scenario.qos_regime,
            "algorithm": variant.name,
            "fog_node_id": name,
            "completed_tasks": count,
        }
        for name, count in fog_counts.items()
    ]
    prediction_rows: list[dict[str, Any]] = []
    if record_predictions:
        for index, (actual, predicted) in enumerate(zip(scenario.true_load, predicted_load)):
            row = {
                "seed": scenario.seed,
                "load": scenario.load,
                "algorithm": variant.name,
                "segment_id": scenario.seed,
                "timestamp_s": index * config.telemetry_step_s,
                "y_true": float(actual),
                "y_pred": float(predicted),
            }
            if record_qos_predictions:
                row.update({"qos_regime": scenario.qos_regime, "target": "load", "link_id": "system"})
            prediction_rows.append(row)
        if record_qos_predictions:
            for target, actual_matrix, predicted_matrix in (
                ("packet_loss", scenario.packet_loss, predicted_loss),
                ("jitter_s", scenario.jitter_s, predicted_jitter),
            ):
                for index in range(actual_matrix.shape[0]):
                    for link_index in range(actual_matrix.shape[1]):
                        prediction_rows.append(
                            {
                                "seed": scenario.seed,
                                "load": scenario.load,
                                "algorithm": variant.name,
                                "segment_id": scenario.seed,
                                "timestamp_s": index * config.telemetry_step_s,
                                "y_true": float(actual_matrix[index, link_index]),
                                "y_pred": float(predicted_matrix[index, link_index]),
                                "qos_regime": scenario.qos_regime,
                                "target": target,
                                "link_id": "cloud" if link_index == actual_matrix.shape[1] - 1 else f"fog-{link_index + 1}",
                            }
                        )
    return run, node_rows, task_records, prediction_rows


def _choose_venue(
    task: Task,
    venues: list[VenueState],
    variant: AlgorithmVariant,
    *,
    policy_model: Any | None = None,
    decision_seed: int = 0,
) -> str | None:
    allowed = [venue for venue in venues if venue.kind in variant.allowed_kinds]
    if not allowed:
        return None
    if variant.policy == "drl_oo":
        if policy_model is None:
            raise ValueError("DRL-OO scheduling requires a trained policy model")
        return policy_model.select_venue(task, allowed, decision_seed=decision_seed)
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
    qos_loss: np.ndarray | None = None,
    qos_jitter: np.ndarray | None = None,
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
        loss_rate = float(qos_loss[fog_index]) if qos_loss is not None else 0.0
        jitter_value = float(qos_jitter[fog_index]) if qos_jitter is not None else 0.0
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
                link=LinkState(
                    uplink_bps,
                    downlink_bps,
                    latency,
                    latency,
                    packet_loss_rate=loss_rate,
                    jitter_s=jitter_value,
                ),
            )
        )
    cloud_factor = max(0.18, 1.0 - 0.52 * load_value)
    cloud_mips = max(15_000.0 * (1.0 - 0.45 * load_value), 7000.0)
    cloud_latency = 0.038 * (1.0 + 1.9 * load_value)
    cloud_index = len(scenario.fog_profiles)
    cloud_loss = float(qos_loss[cloud_index]) if qos_loss is not None else 0.0
    cloud_jitter = float(qos_jitter[cloud_index]) if qos_jitter is not None else 0.0
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
                packet_loss_rate=cloud_loss,
                jitter_s=cloud_jitter,
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
    factor = expected_retransmission_factor(
        venue.link.packet_loss_rate, venue.link.max_retries
    )
    uplink = (
        8.0 * task.uplink_bytes / venue.link.uplink_bps * factor
        + venue.link.uplink_latency_s
        + venue.link.jitter_s
    )
    downlink = (
        8.0 * task.downlink_bytes / venue.link.downlink_bps * factor
        + venue.link.downlink_latency_s
        + venue.link.jitter_s
    )
    return uplink + downlink, uplink, downlink


def _realized_network_transfer(
    task: Task,
    venue: VenueState,
    config: ExperimentConfig,
    *,
    scenario_seed: int,
    task_order: int,
) -> NetworkTransfer:
    if venue.kind == "local" or venue.link is None:
        return NetworkTransfer(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    link = venue.link
    venue_code = 0 if venue.kind == "cloud" else int(venue.name.split("-")[-1])

    def direction(
        byte_count: float,
        bps: float,
        latency_s: float,
        direction_code: int,
    ) -> tuple[float, float, float, bool]:
        if byte_count <= 0:
            return latency_s, 0.0, 0.0, False
        sequence = np.random.SeedSequence(
            [scenario_seed, task_order, venue_code, direction_code, 91_337]
        )
        rng = np.random.default_rng(sequence)
        packet_count = max(int(math.ceil(byte_count / config.packet_payload_bytes)), 1)
        outstanding = packet_count
        transmitted_packets = packet_count
        retry_rounds = 0
        failed = False
        for attempt in range(link.max_retries + 1):
            lost = int(rng.binomial(outstanding, link.packet_loss_rate))
            if lost == 0:
                break
            if attempt == link.max_retries:
                failed = True
                break
            outstanding = lost
            transmitted_packets += lost
            retry_rounds += 1
        retransmission_bytes = max(transmitted_packets - packet_count, 0) * config.packet_payload_bytes
        serialization_s = 8.0 * (byte_count + retransmission_bytes) / bps
        sampled_jitter = (
            float(rng.gamma(shape=2.0, scale=link.jitter_s / 2.0))
            if link.jitter_s > 0.0
            else 0.0
        )
        elapsed_s = (
            serialization_s
            + latency_s
            + sampled_jitter
            + retry_rounds * (latency_s + sampled_jitter)
        )
        return elapsed_s, serialization_s, sampled_jitter, failed

    uplink_s, uplink_serialization_s, uplink_jitter, uplink_failed = direction(
        task.uplink_bytes, link.uplink_bps, link.uplink_latency_s, 1
    )
    downlink_s, downlink_serialization_s, downlink_jitter, downlink_failed = direction(
        task.downlink_bytes, link.downlink_bps, link.downlink_latency_s, 2
    )
    base_bytes = task.uplink_bytes + task.downlink_bytes
    transmitted_bytes = (
        uplink_serialization_s * link.uplink_bps / 8.0
        + downlink_serialization_s * link.downlink_bps / 8.0
    )
    return NetworkTransfer(
        communication_s=uplink_s + downlink_s,
        uplink_s=uplink_s,
        downlink_s=downlink_s,
        uplink_serialization_s=uplink_serialization_s,
        downlink_serialization_s=downlink_serialization_s,
        jitter_s=uplink_jitter + downlink_jitter,
        retransmission_bytes=max(transmitted_bytes - base_bytes, 0.0),
        failed=uplink_failed or downlink_failed,
    )


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
    *,
    network_failed: bool = False,
    transfer: NetworkTransfer | None = None,
    packet_loss_rate: float = 0.0,
    jitter_s: float = 0.0,
    retransmission_bytes: float = 0.0,
) -> dict[str, Any]:
    if transfer is not None:
        jitter_s = transfer.jitter_s
        retransmission_bytes = transfer.retransmission_bytes
        if venue is not None and venue.link is not None:
            packet_loss_rate = venue.link.packet_loss_rate
    return {
        "seed": scenario.seed,
        "load": scenario.load,
        "qos_regime": scenario.qos_regime,
        "algorithm": variant.name,
        "task_id": task.task_id,
        "arrival_s": task.arrival_s,
        "completion_s": completion_s if completion_s is not None else "",
        "relative_deadline_s": task.relative_deadline_s,
        "missed": int(missed),
        "rejected": int(rejected),
        "network_failed": int(network_failed),
        "energy_j": energy_j,
        "monetary_cost": monetary_cost,
        "selected_venue": venue.name if venue is not None else "",
        "venue_kind": venue.kind if venue is not None else "rejected",
        "packet_loss_rate": packet_loss_rate,
        "jitter_ms": 1000.0 * jitter_s,
        "retransmission_bytes": retransmission_bytes,
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
