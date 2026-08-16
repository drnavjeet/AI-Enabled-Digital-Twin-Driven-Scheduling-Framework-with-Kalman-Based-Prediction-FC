from __future__ import annotations

import unittest

import numpy as np

from dt_kf import (
    LinkState,
    ScalarKalman,
    SchedulerConfig,
    Task,
    VenueState,
    error_metrics,
    link_health_score,
    select_venue,
)
from recompute import point_estimate_comparisons
from experiment import (
    AlgorithmVariant,
    ExperimentConfig,
    NodeRuntime,
    ScheduledJob,
    jain_fairness,
    predictor_series,
)
from run_independent_experiments import holm_adjust, student_t_cdf


class PredictorTests(unittest.TestCase):
    def test_kalman_predictions_are_one_step_ahead(self) -> None:
        observations = np.array([0.2, 0.3, 0.4, 0.5])
        predictions = ScalarKalman(1e-3, 1e-2).one_step_predictions(observations)
        self.assertTrue(np.isnan(predictions[0]))
        self.assertAlmostEqual(predictions[1], observations[0])
        self.assertTrue(np.all(np.isfinite(predictions[1:])))

    def test_metrics_use_fractional_mape(self) -> None:
        metrics = error_metrics([1.0, 2.0], [0.5, 1.0])
        self.assertAlmostEqual(metrics["mae"], 0.75)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.625))
        self.assertAlmostEqual(metrics["mape"], 0.5, places=5)


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SchedulerConfig()
        self.good_link = LinkState(
            uplink_bps=20_000_000,
            downlink_bps=40_000_000,
            uplink_latency_s=0.005,
            downlink_latency_s=0.005,
        )

    def test_link_health_is_bounded(self) -> None:
        score = link_health_score(self.good_link, self.config)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_infeasible_candidate_is_rejected_before_ranking(self) -> None:
        task = Task("t1", 0.0, 0.5, 500.0, 10_000, 2_000, min_trust=0.8)
        venues = [
            VenueState("fog-untrusted", "fog", 6000, 1e-12, 1e-5, 0.2, 0.9, link=self.good_link),
            VenueState("cloud", "cloud", 20000, 1e-13, 2e-5, 0.95, 0.95, link=self.good_link),
        ]
        decision = select_venue(task, venues, self.config)
        self.assertEqual(decision.selected_venue, "cloud")
        untrusted = next(item for item in decision.evaluations if item.venue == "fog-untrusted")
        self.assertIn("trust_below_minimum", untrusted.reasons)

    def test_bytes_to_bits_factor_is_applied(self) -> None:
        task = Task("t2", 0.0, 10.0, 1.0, 1_000_000, 0)
        venue = VenueState(
            "fog",
            "fog",
            6000,
            1e-12,
            0.0,
            1.0,
            1.0,
            link=LinkState(8_000_000, 8_000_000, 0.0, 0.0),
        )
        decision = select_venue(task, [venue], self.config)
        evaluation = decision.evaluations[0]
        self.assertAlmostEqual(evaluation.communication_s, 1.0)


class ResultAuditTests(unittest.TestCase):
    def test_point_estimate_comparisons_are_complete(self) -> None:
        rows = point_estimate_comparisons()
        self.assertEqual(len(rows), 74)

    def test_high_load_latency_vs_dt_opt_is_recomputed(self) -> None:
        row = next(
            item
            for item in point_estimate_comparisons()
            if item["table"] == "performance"
            and item["load"] == "high"
            and item["metric"] == "mean_latency_ms"
            and item["comparator"] == "DT-OPT"
        )
        self.assertAlmostEqual(row["relative_improvement_pct"], 10.8416, places=4)


class IndependentExperimentTests(unittest.TestCase):
    def test_jain_fairness_uses_node_counts(self) -> None:
        self.assertAlmostEqual(jain_fairness([10, 10, 10]), 1.0)
        self.assertAlmostEqual(jain_fairness([10, 0]), 0.5)

    def test_predictors_are_strictly_one_step_ahead(self) -> None:
        values = np.asarray([0.2, 0.8, 0.3, 0.7])
        variant = AlgorithmVariant("test", "last", "cost")
        predictions = predictor_series(values, variant, ExperimentConfig())
        self.assertAlmostEqual(predictions[1], values[0])
        self.assertAlmostEqual(predictions[2], values[1])

    def test_nonpreemptive_edf_reorders_only_waiting_jobs(self) -> None:
        venue = VenueState("fog", "fog", 1000, 1e-12, 1e-5, 1.0, 1.0)
        runtime = NodeRuntime()

        def job(task_id: str, arrival: float, deadline: float, order: int) -> ScheduledJob:
            return ScheduledJob(
                Task(task_id, arrival, deadline, 1000, 0, 0),
                venue,
                arrival,
                1.0,
                0.0,
                0.0,
                0.0,
                True,
                order,
            )

        running = job("running", 0.0, 10.0, 0)
        waiting = job("waiting", 0.1, 9.0, 1)
        urgent = job("urgent", 0.2, 2.0, 2)
        runtime.schedule(running, 0.0, "edf")
        runtime.schedule(waiting, 0.1, "edf")
        runtime.schedule(urgent, 0.2, "edf")
        self.assertAlmostEqual(running.service_end_s, 1.0)
        self.assertAlmostEqual(urgent.service_start_s, 1.0)
        self.assertAlmostEqual(waiting.service_start_s, 2.0)

    def test_statistical_helpers_match_reference_values(self) -> None:
        self.assertAlmostEqual(student_t_cdf(2.04523, 29), 0.975, places=3)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
