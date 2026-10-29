import unittest

import numpy as np

from src.evaluation.metrics import calculate_classification_metrics
from src.evaluation.metrics import calculate_lead_time_to_failure
from src.models.attention_lstm_ae import AttentionLSTMAutoencoder
from src.scheduling.maintenance_planner import MaintenanceScheduler
import torch


class ClassificationMetricsTests(unittest.TestCase):
    def test_confusion_counts_and_scores_are_consistent(self):
        result = calculate_classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0])
        )

        self.assertEqual((result["TN"], result["FP"], result["FN"], result["TP"]), (1, 1, 1, 1))
        self.assertEqual(result["Accuracy"], 0.5)
        self.assertEqual(result["Precision"], 0.5)
        self.assertEqual(result["Recall"], 0.5)
        self.assertEqual(result["F1_score"], 0.5)

    def test_confusion_counts_exist_when_one_class_is_absent(self):
        result = calculate_classification_metrics(
            np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        )

        self.assertEqual((result["TN"], result["FP"], result["FN"], result["TP"]), (3, 0, 0, 0))

    def test_lead_time_does_not_call_out_of_range_events_missed(self):
        results = calculate_lead_time_to_failure(
            np.array(["2020-06-01T00:00:00"], dtype="datetime64[s]"),
            np.array([0]),
        )

        self.assertEqual(results[0]["Status"], "Not evaluated in supplied data")


class AttentionModelTests(unittest.TestCase):
    def test_model_reconstructs_input_and_attention_is_normalized(self):
        model = AttentionLSTMAutoencoder(
            input_dim=15, seq_len=30, hidden_dim1=16, hidden_dim2=8
        )
        values = torch.randn(2, 30, 15)
        reconstruction, weights = model(values)

        self.assertEqual(tuple(reconstruction.shape), tuple(values.shape))
        self.assertEqual(tuple(weights.shape), (2, 30))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))


class MaintenanceSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = MaintenanceScheduler(
            threshold_warning=0.03, threshold_critical=0.04
        )

    def test_rul_is_unavailable_without_observed_degradation(self):
        self.assertIsNone(
            self.scheduler.estimate_remaining_useful_life(
                np.array([95.0, 94.0, 96.0, 95.0])
            )
        )

    def test_rul_estimates_degrading_health_trend(self):
        health = np.linspace(80.0, 50.0, 10)
        rul = self.scheduler.estimate_remaining_useful_life(health, 1.0)

        self.assertIsNotNone(rul)
        self.assertGreater(rul, 0)


if __name__ == "__main__":
    unittest.main()
