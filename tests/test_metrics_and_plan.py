import json
from pathlib import Path
import tempfile
import unittest

from scheduler_lab.analysis import holm_adjust, paired_permutation_p
from scheduler_lab.experiment import load_config, make_plan
from scheduler_lab.linux import parse_cpus
from scheduler_lab.metrics import latency_metrics, percentile


class MetricTests(unittest.TestCase):
    def test_known_gaps_separate_ttft_from_itl(self):
        metrics = latency_metrics([100, 110, 130, 160])
        self.assertEqual(metrics["ttft_ms"], 100)
        self.assertEqual(metrics["itl_mean_ms"], 20)
        self.assertAlmostEqual(metrics["itl_p99_ms"], 29.8)
        self.assertAlmostEqual(metrics["tail_jitter_ms"], 9.8)
        self.assertEqual(metrics["itl_std_ms"], 10)

    def test_short_stream_is_not_zero_jitter(self):
        self.assertIsNone(latency_metrics([10])["itl_p99_ms"])
        self.assertIsNone(percentile([], 99))

    def test_five_pairs_cannot_attain_point_zero_five(self):
        self.assertEqual(paired_permutation_p([1, 2, 3, 4, 5]), 0.0625)
        self.assertEqual(paired_permutation_p([0, 0, 0, 0, 0]), 1)

    def test_holm_preserves_original_order(self):
        self.assertEqual(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


class PlanningTests(unittest.TestCase):
    def test_proposal_has_35_trials_with_balanced_blocks(self):
        config = load_config("configs/proposal.json")
        plan = make_plan(config, 3)
        self.assertEqual(len(plan), 35)
        self.assertEqual(plan, make_plan(config, 3))
        self.assertEqual(len({t["trial_id"] for t in plan}), 35)
        for replication in range(1, 6):
            self.assertEqual(len([t for t in plan if t["replication"] == replication]), 7)
        self.assertEqual({t["stress_workers"] for t in plan}, {0, 3, 6})

    def test_full_factorial_and_fractional_worker_rounding(self):
        plan = make_plan(load_config("configs/full.json"), 3)
        self.assertEqual(len(plan), 60)
        row = next(t for t in plan if t["contention_pct"] == 50)
        self.assertEqual(row["stress_workers"], 2)
        self.assertAlmostEqual(row["actual_worker_density_pct"], 200 / 3)

    def test_cpu_lists_and_invalid_ranges(self):
        self.assertEqual(parse_cpus("1-3,6,3"), {1, 2, 3, 6})
        for value in ("3-1", "-1", "", "1-2-3"):
            with self.assertRaises(ValueError):
                parse_cpus(value)

    def test_reject_duplicate_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            condition = {"policy": "other", "contention_pct": 0}
            path.write_text(json.dumps({"replications": 5, "conditions": [condition, condition]}))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
