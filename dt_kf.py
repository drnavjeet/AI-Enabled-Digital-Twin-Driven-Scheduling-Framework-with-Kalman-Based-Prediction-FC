from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite, prod
from typing import Iterable, Literal, Sequence

import numpy as np


VenueKind = Literal["local", "fog", "cloud"]


@dataclass(frozen=True)
class Task:
    task_id: str
    arrival_s: float
    deadline_s: float
    compute_mi: float
    uplink_bytes: float
    downlink_bytes: float
    min_accuracy: float = 0.0
    min_trust: float = 0.0
    energy_budget_j: float = inf
    inference: bool = False

    @property
    def relative_deadline_s(self) -> float:
        return self.deadline_s - self.arrival_s


@dataclass(frozen=True)
class LinkState:
    uplink_bps: float
    downlink_bps: float
    uplink_latency_s: float
    downlink_latency_s: float
    packet_loss_rate: float = 0.0
    jitter_s: float = 0.0
    max_retries: int = 3


@dataclass(frozen=True)
class VenueState:
    name: str
    kind: VenueKind
    mips: float
    energy_coefficient: float
    price_per_mi: float
    trust: float
    predicted_accuracy: float
    queue_work_mi: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    link: LinkState | None = None
    battery_fraction: float = 1.0


@dataclass(frozen=True)
class SchedulerConfig:
    latency_weight: float = 0.25
    energy_weight: float = 0.25
    cost_weight: float = 0.25
    accuracy_weight: float = 0.25
    minimum_bandwidth_bps: float = 1_000_000.0
    minimum_lhs: float = 0.50
    minimum_battery_fraction: float = 0.10
    reference_uplink_bps: float = 10_000_000.0
    reference_downlink_bps: float = 20_000_000.0
    reference_uplink_latency_s: float = 0.050
    reference_downlink_latency_s: float = 0.050
    reference_packet_loss_rate: float = 0.150
    reference_jitter_s: float = 0.100
    maximum_packet_loss_rate: float = 0.150
    maximum_jitter_s: float = 0.100
    device_tx_power_w: float = 1.3
    device_rx_power_w: float = 0.9

    def __post_init__(self) -> None:
        weights = (
            self.latency_weight,
            self.energy_weight,
            self.cost_weight,
            self.accuracy_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Objective weights must be non-negative")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Objective weights must sum to one")


@dataclass(frozen=True)
class CandidateEvaluation:
    venue: str
    kind: VenueKind
    feasible: bool
    reasons: tuple[str, ...]
    latency_s: float
    queue_s: float
    communication_s: float
    compute_s: float
    system_energy_j: float
    device_energy_j: float
    monetary_cost: float
    accuracy: float
    lhs: float | None
    objective: float | None = None


@dataclass(frozen=True)
class Decision:
    task_id: str
    selected_venue: str | None
    rejected: bool
    evaluations: tuple[CandidateEvaluation, ...]


class ScalarKalman:
    """Random-walk scalar Kalman filter with F=H=1."""

    def __init__(self, process_variance: float, measurement_variance: float) -> None:
        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("Kalman variances must be positive")
        self.q = float(process_variance)
        self.r = float(measurement_variance)

    def one_step_predictions(self, observations: Sequence[float]) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Expected a non-empty one-dimensional series")
        predictions = np.full(values.shape, np.nan, dtype=float)
        estimate = float(values[0])
        covariance = 1.0
        for index in range(1, len(values)):
            covariance_prior = covariance + self.q
            predictions[index] = estimate
            gain = covariance_prior / (covariance_prior + self.r)
            estimate = estimate + gain * (float(values[index]) - estimate)
            covariance = (1.0 - gain) * covariance_prior
        return predictions


def last_value_predictions(observations: Sequence[float]) -> np.ndarray:
    values = np.asarray(observations, dtype=float)
    predictions = np.full(values.shape, np.nan, dtype=float)
    predictions[1:] = values[:-1]
    return predictions


def ema_predictions(observations: Sequence[float], alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("EMA alpha must be in (0, 1]")
    values = np.asarray(observations, dtype=float)
    predictions = np.full(values.shape, np.nan, dtype=float)
    estimate = float(values[0])
    for index in range(1, len(values)):
        predictions[index] = estimate
        estimate = alpha * float(values[index]) + (1.0 - alpha) * estimate
    return predictions


def error_metrics(
    observations: Sequence[float], predictions: Sequence[float], epsilon: float = 1e-6
) -> dict[str, float]:
    actual = np.asarray(observations, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(valid):
        raise ValueError("No valid observation/prediction pairs")
    error = actual[valid] - predicted[valid]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float(np.mean(np.abs(error) / (np.abs(actual[valid]) + epsilon))),
        "n": int(np.sum(valid)),
    }


def link_health_score(link: LinkState, config: SchedulerConfig) -> float:
    legacy_components = (
        min(link.uplink_bps / config.reference_uplink_bps, 1.0),
        min(link.downlink_bps / config.reference_downlink_bps, 1.0),
        1.0 - min(link.uplink_latency_s / config.reference_uplink_latency_s, 1.0),
        1.0 - min(link.downlink_latency_s / config.reference_downlink_latency_s, 1.0),
    )
    if link.packet_loss_rate == 0.0 and link.jitter_s == 0.0:
        components = legacy_components
    else:
        components = legacy_components + (
            1.0 - min(link.packet_loss_rate / config.reference_packet_loss_rate, 1.0),
            1.0 - min(link.jitter_s / config.reference_jitter_s, 1.0),
        )
    return float(prod(max(0.0, value) for value in components) ** (1.0 / len(components)))


def expected_retransmission_factor(packet_loss_rate: float, max_retries: int) -> float:
    """Expected transmissions per packet for a retry-capped independent-loss model."""
    if not 0.0 <= packet_loss_rate < 1.0:
        raise ValueError("Packet-loss rate must be in [0, 1)")
    if max_retries < 0:
        raise ValueError("Maximum retries must be non-negative")
    return float(sum(packet_loss_rate**attempt for attempt in range(max_retries + 1)))


def edf_queue_delay_s(task: Task, venue: VenueState) -> float:
    earlier_work = sum(
        remaining_mi
        for absolute_deadline_s, remaining_mi in venue.queue_work_mi
        if absolute_deadline_s < task.deadline_s
    )
    return earlier_work / venue.mips


def _evaluate_candidate(
    task: Task, venue: VenueState, config: SchedulerConfig
) -> CandidateEvaluation:
    reasons: list[str] = []
    if venue.mips <= 0:
        raise ValueError(f"Venue {venue.name} must have positive MIPS")

    compute_s = task.compute_mi / venue.mips
    queue_s = 0.0 if venue.kind == "local" else edf_queue_delay_s(task, venue)
    communication_s = 0.0
    device_energy_j = 0.0
    lhs: float | None = None

    if venue.kind == "local":
        device_energy_j = venue.energy_coefficient * task.compute_mi * venue.mips**2
        if venue.battery_fraction < config.minimum_battery_fraction:
            reasons.append("battery_below_minimum")
    else:
        if venue.link is None:
            reasons.append("missing_link_state")
        else:
            link = venue.link
            if not 0.0 <= link.packet_loss_rate < 1.0:
                raise ValueError(f"Venue {venue.name} has an invalid packet-loss rate")
            if link.jitter_s < 0.0:
                raise ValueError(f"Venue {venue.name} has negative jitter")
            if min(link.uplink_bps, link.downlink_bps) < config.minimum_bandwidth_bps:
                reasons.append("bandwidth_below_minimum")
            # Task sizes are bytes while bandwidth is bits/s, so the factor eight is required.
            retry_factor = expected_retransmission_factor(
                link.packet_loss_rate, link.max_retries
            )
            uplink_tx_s = 8.0 * task.uplink_bytes / link.uplink_bps * retry_factor
            downlink_tx_s = 8.0 * task.downlink_bytes / link.downlink_bps * retry_factor
            communication_s = (
                uplink_tx_s
                + link.uplink_latency_s
                + link.jitter_s
                + downlink_tx_s
                + link.downlink_latency_s
                + link.jitter_s
            )
            device_energy_j = (
                config.device_tx_power_w * uplink_tx_s
                + config.device_rx_power_w * downlink_tx_s
            )
            if venue.kind == "fog":
                lhs = link_health_score(link, config)
                if lhs < config.minimum_lhs:
                    reasons.append("lhs_below_minimum")
            if link.packet_loss_rate > config.maximum_packet_loss_rate:
                reasons.append("packet_loss_above_maximum")
            if link.jitter_s > config.maximum_jitter_s:
                reasons.append("jitter_above_maximum")

    latency_s = communication_s + queue_s + compute_s
    venue_energy_j = venue.energy_coefficient * task.compute_mi * venue.mips**2
    system_energy_j = device_energy_j + (0.0 if venue.kind == "local" else venue_energy_j)
    monetary_cost = 0.0 if venue.kind == "local" else venue.price_per_mi * task.compute_mi

    if venue.trust < task.min_trust:
        reasons.append("trust_below_minimum")
    if task.inference and venue.predicted_accuracy < task.min_accuracy:
        reasons.append("accuracy_below_minimum")
    if latency_s > task.relative_deadline_s:
        reasons.append("deadline_miss_predicted")
    if device_energy_j > task.energy_budget_j:
        reasons.append("device_energy_budget_exceeded")

    return CandidateEvaluation(
        venue=venue.name,
        kind=venue.kind,
        feasible=not reasons,
        reasons=tuple(reasons),
        latency_s=latency_s,
        queue_s=queue_s,
        communication_s=communication_s,
        compute_s=compute_s,
        system_energy_j=system_energy_j,
        device_energy_j=device_energy_j,
        monetary_cost=monetary_cost,
        accuracy=venue.predicted_accuracy,
        lhs=lhs,
    )


def _positive_max(values: Iterable[float]) -> float:
    maximum = max(values, default=0.0)
    return maximum if maximum > 0 else 1.0


def select_venue(
    task: Task, venues: Sequence[VenueState], config: SchedulerConfig
) -> Decision:
    evaluations = [_evaluate_candidate(task, venue, config) for venue in venues]
    feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
    if not feasible:
        return Decision(task.task_id, None, True, tuple(evaluations))

    max_latency = _positive_max(item.latency_s for item in feasible)
    max_energy = _positive_max(item.system_energy_j for item in feasible)
    max_cost = _positive_max(item.monetary_cost for item in feasible)
    max_accuracy = _positive_max(item.accuracy for item in feasible) if task.inference else 1.0

    scored: list[CandidateEvaluation] = []
    for item in evaluations:
        if not item.feasible:
            scored.append(item)
            continue
        normalized_accuracy = item.accuracy / max_accuracy if task.inference else 1.0
        objective = (
            config.latency_weight * item.latency_s / max_latency
            + config.energy_weight * item.system_energy_j / max_energy
            + config.cost_weight * item.monetary_cost / max_cost
            + config.accuracy_weight * (1.0 - normalized_accuracy)
        )
        scored.append(
            CandidateEvaluation(**{**item.__dict__, "objective": float(objective)})
        )

    eligible = [item for item in scored if item.feasible and item.objective is not None]
    selected = min(eligible, key=lambda item: (item.objective, item.latency_s, item.venue))
    return Decision(task.task_id, selected.venue, False, tuple(scored))
