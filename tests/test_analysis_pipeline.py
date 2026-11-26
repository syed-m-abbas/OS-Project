import json
from pathlib import Path
import tempfile
import unittest

from scheduler_lab.analysis import analyze_results
from scheduler_lab.demo import create_demo


class AnalysisTests(unittest.TestCase):
    def test_demo_provenance_figures_and_trial_level_statistics(self):
        try:
            import matplotlib
            import numpy
        except ImportError:
            self.skipTest("Analysis extras not installed")
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "demo", Path(temp) / "analysis"
            create_demo(source)
            analyze_results(source, output)
            data = json.loads((output / "analysis.json").read_text())
            self.assertEqual(data["data_kind"], "synthetic")
            self.assertEqual(data["valid_trials"], 35)
            self.assertTrue(data["complete_matrix"])
            self.assertEqual(len(data["summaries"]), 7)
            self.assertEqual(len(data["comparisons"]), 4)
            for name in ("throughput", "jitter_cdf", "context_switches", "ttft"):
                self.assertGreater((output / (name + ".png")).stat().st_size, 1000)
                self.assertGreater((output / (name + ".pdf")).stat().st_size, 1000)
            self.assertIn("Synthetic", (output / "results_table.tex").read_text())
            with self.assertRaises(ValueError):
                analyze_results(source, output)

    def test_recorded_condition_cannot_be_changed(self):
        try:
            import matplotlib
            import numpy
        except ImportError:
            self.skipTest("Analysis extras not installed")
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "demo", Path(temp) / "analysis"
            create_demo(source)
            rows = (source / "trials.jsonl").read_text().splitlines()
            row = json.loads(rows[0])
            row["policy"] = "fifo" if row["policy"] != "fifo" else "batch"
            rows[0] = json.dumps(row)
            (source / "trials.jsonl").write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "recorded plan"):
                analyze_results(source, output)
