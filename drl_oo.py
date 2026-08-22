from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from dt_kf import SchedulerConfig, Task, VenueState, _evaluate_candidate
from experiment import (
    AlgorithmVariant,
    ExperimentConfig,
    ExperimentData,
    NodeRuntime,
    ScheduledJob,
    _build_venues,
    _communication_times,
    generate_scenario,
    predictor_matrix,
    predictor_series,
)


FEATURE_COUNT = 23


@dataclass(frozen=True)
class DRLOOConfig:
    implementation_version: int = 2
    hidden_units: int = 256
    candidate_actions: int = 100
    top_actions: int = 10
    learning_rate: float = 5e-4
    discount_factor: float = 0.99
    batch_size: int = 32
    soft_update: float = 0.005
    target_update_interval: int = 3
    replay_capacity: int = 60_000
    episodes: int = 200
    tasks_per_episode: int = 160
    updates_per_episode: int = 2


class DenseNetwork:
    def __init__(self, input_size: int, hidden_units: int, rng: np.random.Generator) -> None:
        self.w1 = rng.normal(0.0, math.sqrt(2.0 / input_size), (input_size, hidden_units))
        self.b1 = np.zeros(hidden_units)
        self.w2 = rng.normal(0.0, math.sqrt(2.0 / hidden_units), (hidden_units, hidden_units))
        self.b2 = np.zeros(hidden_units)
        self.w3 = rng.normal(0.0, math.sqrt(2.0 / hidden_units), (hidden_units, 1))
        self.b3 = np.zeros(1)
        self._adam_m = [np.zeros_like(value) for value in self.parameters]
        self._adam_v = [np.zeros_like(value) for value in self.parameters]
        self._adam_step = 0

    @property
    def parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def copy(self) -> "DenseNetwork":
        clone = object.__new__(DenseNetwork)
        clone.w1, clone.b1, clone.w2, clone.b2, clone.w3, clone.b3 = [
            value.copy() for value in self.parameters
        ]
        clone._adam_m = [np.zeros_like(value) for value in clone.parameters]
        clone._adam_v = [np.zeros_like(value) for value in clone.parameters]
        clone._adam_step = 0
        return clone

    def predict(self, features: np.ndarray, *, binary: bool = False) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        output = (h2 @ self.w3 + self.b3).reshape(-1)
        if binary:
            output = 1.0 / (1.0 + np.exp(-np.clip(output, -30.0, 30.0)))
        return output

    def train_batch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        *,
        binary: bool,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float).reshape(-1, 1)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        weights = (
            np.ones_like(y)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=float).reshape(-1, 1)
        )
        if binary:
            predictions = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            loss = -np.mean(
                weights
                * (y * np.log(predictions + 1e-9) + (1.0 - y) * np.log(1.0 - predictions + 1e-9))
            )
            gradient = weights * (predictions - y) / len(x)
        else:
            residual = logits - y
            loss = float(np.mean(weights * residual**2))
            gradient = 2.0 * weights * residual / len(x)

        gradients: list[np.ndarray] = []
        grad_w3 = h2.T @ gradient
        grad_b3 = np.sum(gradient, axis=0)
        grad_h2 = (gradient @ self.w3.T) * (1.0 - h2**2)
        grad_w2 = h1.T @ grad_h2
        grad_b2 = np.sum(grad_h2, axis=0)
        grad_h1 = (grad_h2 @ self.w2.T) * (1.0 - h1**2)
        grad_w1 = x.T @ grad_h1
        grad_b1 = np.sum(grad_h1, axis=0)
        gradients.extend([grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3])

        norm = math.sqrt(sum(float(np.sum(gradient_value**2)) for gradient_value in gradients))
        if norm > 5.0:
            gradients = [gradient_value * (5.0 / norm) for gradient_value in gradients]
        self._adam_step += 1
        beta1, beta2 = 0.9, 0.999
        for index, (parameter, gradient_value) in enumerate(zip(self.parameters, gradients)):
            self._adam_m[index] = beta1 * self._adam_m[index] + (1.0 - beta1) * gradient_value
            self._adam_v[index] = beta2 * self._adam_v[index] + (1.0 - beta2) * gradient_value**2
            corrected_m = self._adam_m[index] / (1.0 - beta1**self._adam_step)
            corrected_v = self._adam_v[index] / (1.0 - beta2**self._adam_step)
            parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        return float(loss)

    def soft_update_from(self, source: "DenseNetwork", coefficient: float) -> None:
        for target_value, source_value in zip(self.parameters, source.parameters):
            target_value *= 1.0 - coefficient
            target_value += coefficient * source_value


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.features = np.zeros((capacity, FEATURE_COUNT), dtype=np.float32)
        self.actor_targets = np.zeros(capacity, dtype=np.float32)
        self.critic_targets = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.cursor = 0

    def add(self, features: np.ndarray, actor_targets: np.ndarray, critic_targets: np.ndarray) -> None:
        for feature, actor_target, critic_target in zip(features, actor_targets, critic_targets):
            self.features[self.cursor] = feature
            self.actor_targets[self.cursor] = actor_target
            self.critic_targets[self.cursor] = critic_target
            self.cursor = (self.cursor + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.integers(0, self.size, size=count)
        return self.features[indices], self.actor_targets[indices], self.critic_targets[indices]


class DRLOOPolicy:
    def __init__(self, config: DRLOOConfig, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.actor = DenseNetwork(FEATURE_COUNT, config.hidden_units, rng)
        self.critic = DenseNetwork(FEATURE_COUNT, config.hidden_units, rng)
        self.target_actor = self.actor.copy()
        self.target_critic = self.critic.copy()
        self.scheduler_config = SchedulerConfig(
            latency_weight=0.5,
            energy_weight=0.5,
            cost_weight=0.0,
            accuracy_weight=0.0,
            minimum_bandwidth_bps=800_000.0,
            minimum_lhs=0.0,
            minimum_battery_fraction=0.08,
            reference_uplink_bps=25_000_000.0,
            reference_downlink_bps=50_000_000.0,
            reference_uplink_latency_s=0.060,
            reference_downlink_latency_s=0.060,
        )

    def candidate_data(
        self, task: Task, venues: list[VenueState]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        rewards: list[float] = []
        feasible: list[bool] = []
        for venue in venues:
            evaluation = _evaluate_candidate(task, venue, self.scheduler_config)
            features.append(candidate_features(task, venue, evaluation))
            latency_ratio = evaluation.latency_s / max(task.relative_deadline_s, 1e-6)
            energy_ratio = evaluation.system_energy_j / 1.5
            reward = 0.5 * (1.0 - latency_ratio) - 0.5 * energy_ratio
            if not evaluation.feasible:
                reward -= 2.0 + 0.1 * len(evaluation.reasons)
            rewards.append(float(np.clip(reward, -4.0, 1.0)))
            feasible.append(evaluation.feasible)
        return np.asarray(features), np.asarray(rewards), np.asarray(feasible, dtype=bool)

    def select_venue(
        self, task: Task, venues: list[VenueState], *, decision_seed: int = 0
    ) -> str | None:
        features, objective_rewards, feasible = self.candidate_data(task, venues)
        feasible_indices = np.flatnonzero(feasible)
        if len(feasible_indices) == 0:
            return None
        # When the feasible action space already fits in the published top-s set,
        # OO retains every action and neural ranking cannot change the final choice.
        if len(feasible_indices) <= self.config.top_actions:
            selected = int(feasible_indices[np.argmax(objective_rewards[feasible_indices])])
            return venues[selected].name
        actor_scores = self.actor.predict(features, binary=True)
        preliminary = int(feasible_indices[np.argmax(actor_scores[feasible_indices])])
        candidates = self._ordinal_candidates(
            preliminary, feasible_indices, len(venues), decision_seed
        )
        critic_scores = self.critic.predict(features[candidates])
        top_count = min(self.config.top_actions, len(candidates))
        top_indices = np.asarray(candidates)[np.argsort(critic_scores)[-top_count:]]
        selected = int(top_indices[np.argmax(objective_rewards[top_indices])])
        return venues[selected].name

    def _ordinal_candidates(
        self,
        preliminary: int,
        feasible_indices: np.ndarray,
        action_count: int,
        decision_seed: int,
    ) -> list[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, decision_seed, 20_025]))
        bit_count = max(int(math.ceil(math.log2(max(action_count, 2)))), 1)
        feasible_set = {int(value) for value in feasible_indices}
        sampled = [preliminary]
        for _ in range(self.config.candidate_actions - 1):
            flip_count = int(rng.integers(1, min(bit_count, 3) + 1))
            positions = rng.choice(bit_count, size=flip_count, replace=False)
            mask = sum(1 << int(position) for position in positions)
            candidate = preliminary ^ mask
            if candidate in feasible_set:
                sampled.append(candidate)
        unique = list(dict.fromkeys(sampled))
        if len(unique) < min(self.config.top_actions, len(feasible_set)):
            unique.extend(index for index in sorted(feasible_set) if index not in unique)
        return unique

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for prefix, network in (
            ("actor", self.actor),
            ("critic", self.critic),
            ("target_actor", self.target_actor),
            ("target_critic", self.target_critic),
        ):
            for index, value in enumerate(network.parameters):
                arrays[f"{prefix}_{index}"] = value
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "policy_seed": self.seed,
                    "config": asdict(self.config),
                    "method": (
                        "Paper-aligned discrete DRL-OO adaptation with simulator replay "
                        "and target-critic bootstrapping"
                    ),
                    **metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "DRLOOPolicy":
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        policy = cls(DRLOOConfig(**metadata["config"]), int(metadata["policy_seed"]))
        values = np.load(path)
        for prefix, network in (
            ("actor", policy.actor),
            ("critic", policy.critic),
            ("target_actor", policy.target_actor),
            ("target_critic", policy.target_critic),
        ):
            for index, parameter in enumerate(network.parameters):
                parameter[...] = values[f"{prefix}_{index}"]
        return policy


def candidate_features(task: Task, venue: VenueState, evaluation: Any) -> np.ndarray:
    kind = [float(venue.kind == name) for name in ("local", "fog", "cloud")]
    link = venue.link
    loss = link.packet_loss_rate if link is not None else 0.0
    jitter = link.jitter_s if link is not None else 0.0
    lhs = evaluation.lhs if evaluation.lhs is not None else 1.0
    values = [
        task.compute_mi / 4500.0,
        task.uplink_bytes / 2_000_000.0,
        task.downlink_bytes / 4_000_000.0,
        task.relative_deadline_s / 2.6,
        task.min_accuracy,
        task.min_trust,
        min(task.energy_budget_j / 1.6, 2.0),
        float(task.inference),
        *kind,
        venue.mips / 15_000.0,
        min(evaluation.queue_s / 2.6, 2.0),
        min(evaluation.communication_s / 2.6, 2.0),
        min(evaluation.compute_s / 2.6, 2.0),
        min(evaluation.latency_s / max(task.relative_deadline_s, 1e-6), 2.0),
        min(evaluation.system_energy_j / 1.5, 2.0),
        min(evaluation.monetary_cost / 0.08, 2.0),
        venue.trust,
        venue.predicted_accuracy,
        float(lhs),
        min(loss / 0.15, 2.0),
        min(jitter / 0.10, 2.0),
    ]
    array = np.asarray(values, dtype=np.float32)
    if len(array) != FEATURE_COUNT:
        raise AssertionError(f"Expected {FEATURE_COUNT} features, received {len(array)}")
    return array


def train_policy(
    data: ExperimentData,
    experiment_config: ExperimentConfig,
    *,
    policy_seed: int,
    config: DRLOOConfig | None = None,
) -> tuple[DRLOOPolicy, list[dict[str, float]]]:
    config = config or DRLOOConfig()
    policy = DRLOOPolicy(config, policy_seed)
    replay = ReplayBuffer(config.replay_capacity)
    rng = np.random.default_rng(policy_seed + 700_001)
    training_config = replace(experiment_config, warmup_s=2.0, measurement_s=8.0)
    variant = AlgorithmVariant(
        "DRL-OO-2025", "last", "drl_oo", qos_aware=True, use_lhs=False
    )
    log: list[dict[str, float]] = []
    loads = ("low", "medium", "high")
    regimes = ("clean", "moderate", "impaired")

    for episode in range(config.episodes):
        load = loads[episode % len(loads)]
        regime = regimes[(episode // len(loads)) % len(regimes)]
        scenario = generate_scenario(
            data,
            training_config,
            load,
            policy_seed * 100_000 + episode,
            fixed_task_count=config.tasks_per_episode,
            qos_regime=regime,
        )
        predicted_load = predictor_series(scenario.true_load, variant, training_config)
        predicted_loss = predictor_matrix(
            scenario.packet_loss, variant, training_config, lower=0.0, upper=0.20
        )
        predicted_jitter = predictor_matrix(
            scenario.jitter_s, variant, training_config, lower=0.0, upper=0.15
        )
        runtimes = {"local": NodeRuntime(), "cloud": NodeRuntime()}
        runtimes.update({profile.name: NodeRuntime() for profile in scenario.fog_profiles})
        episode_rewards: list[float] = []
        pending_transition: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for order, sim_task in enumerate(scenario.tasks):
            task = sim_task.task
            telemetry_index = min(
                int(task.arrival_s / training_config.telemetry_step_s),
                len(scenario.true_load) - 1,
            )
            for runtime in runtimes.values():
                runtime.prune(task.arrival_s)
            venues = _build_venues(
                task,
                scenario.user_coordinates[sim_task.user_index],
                scenario,
                runtimes,
                telemetry_index,
                float(predicted_load[telemetry_index]),
                variant,
                actual=False,
                qos_loss=predicted_loss[telemetry_index],
                qos_jitter=predicted_jitter[telemetry_index],
            )
            features, rewards, feasible = policy.candidate_data(task, venues)
            feasible_indices = np.flatnonzero(feasible)
            if len(feasible_indices) == 0:
                continue
            best = int(feasible_indices[np.argmax(rewards[feasible_indices])])
            actor_targets = np.zeros(len(venues), dtype=float)
            actor_targets[best] = 1.0
            if pending_transition is not None:
                previous_features, previous_actor_targets, previous_rewards = pending_transition
                bootstrap = float(
                    np.max(policy.target_critic.predict(features[feasible_indices]))
                )
                critic_targets = np.clip(
                    previous_rewards + config.discount_factor * bootstrap,
                    -6.0,
                    3.0,
                )
                replay.add(previous_features, previous_actor_targets, critic_targets)
            pending_transition = (features, actor_targets, rewards)
            episode_rewards.append(float(rewards[best]))

            exploration = max(0.05, 0.30 * (1.0 - episode / config.episodes))
            if rng.random() < exploration:
                selected_index = int(rng.choice(feasible_indices))
            else:
                selected_index = best
            selected = venues[selected_index]
            communication_s, uplink_s, downlink_s = _communication_times(task, selected)
            venue_energy = selected.energy_coefficient * task.compute_mi * selected.mips**2
            job = ScheduledJob(
                task=task,
                venue=selected,
                service_arrival_s=task.arrival_s + uplink_s,
                service_s=task.compute_mi / selected.mips,
                downlink_s=downlink_s,
                energy_j=venue_energy,
                monetary_cost=selected.price_per_mi * task.compute_mi,
                measured=True,
                order=order,
            )
            runtimes[selected.name].schedule(job, task.arrival_s, "edf")

        if pending_transition is not None:
            replay.add(*pending_transition)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        if replay.size >= config.batch_size:
            for _ in range(config.updates_per_episode):
                x, actor_y, critic_y = replay.sample(rng, config.batch_size)
                actor_losses.append(
                    policy.actor.train_batch(
                        x,
                        actor_y,
                        config.learning_rate,
                        binary=True,
                        sample_weights=np.where(actor_y > 0.5, 6.0, 1.0),
                    )
                )
                critic_losses.append(
                    policy.critic.train_batch(
                        x, critic_y, config.learning_rate, binary=False
                    )
                )
        if (episode + 1) % config.target_update_interval == 0:
            policy.target_actor.soft_update_from(policy.actor, config.soft_update)
            policy.target_critic.soft_update_from(policy.critic, config.soft_update)
        log.append(
            {
                "policy_seed": float(policy_seed),
                "episode": float(episode + 1),
                "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else math.nan,
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else math.nan,
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else math.nan,
                "replay_size": float(replay.size),
            }
        )
    return policy, log
