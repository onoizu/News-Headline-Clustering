import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts/cluster_headlines.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_sampling_keeps_exact_size_unique_source_rows_and_years(self):
        # Non-contiguous source indices expose index-reset sampling mistakes.
        frame = pd.DataFrame({"publish_date": ["20200101"] * 7 + ["20210101"] * 3,
                              "headline_text": [f"sample headline {chr(97+i)}" for i in range(10)]})
        frame = pd.concat([frame, frame.iloc[[0]], pd.DataFrame({
            "publish_date": ["invalid", "20200101"], "headline_text": ["bad date", "123"]})], ignore_index=True)
        a, audit = pipeline.prepare_data(frame, 5, 42)
        b, _ = pipeline.prepare_data(frame, 5, 42)
        self.assertEqual(len(a), 5)
        self.assertTrue(a.source_row.is_unique)
        self.assertTrue(a.headline_text.is_unique)
        self.assertEqual(set(a.year), {2020, 2021})
        self.assertEqual(audit["duplicate_headlines_removed"], 1)
        self.assertEqual(audit["invalid_or_empty_rows"], 2)
        pd.testing.assert_frame_equal(a, b)

    def test_all_noise_and_single_cluster_have_no_silhouette(self):
        values = np.array([[0., 0.], [1., 1.], [2., 2.]])
        for labels in [np.array([-1, -1, -1]), np.array([0, 0, 0])]:
            result = pipeline.evaluate(values, labels, 42)
            self.assertIsNone(result["silhouette"])
            json.dumps(result, allow_nan=False)

    def test_invalid_data_reports_clear_error(self):
        with self.assertRaisesRegex(ValueError, "CSV must contain"):
            pipeline.prepare_data(pd.DataFrame({"wrong": []}), 5, 42)

    def test_cli_writes_consistent_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run([sys.executable, str(ROOT / "scripts/cluster_headlines.py"),
                            "--input", str(ROOT / "data/demo_headlines.csv"),
                            "--output", str(output), "--embedding", "tfidf",
                            "--reducer", "none", "--clusterer", "dbscan",
                            "--eps", "0.9", "--min-samples", "2", "--plot"],
                           check=True, capture_output=True, text=True)
            assignments = pd.read_csv(output / "assignments.csv")
            metrics = json.loads((output / "metrics.json").read_text())
            topics = json.loads((output / "topics.json").read_text())
            metadata = json.loads((output / "run_info.json").read_text())
            self.assertEqual(len(assignments), metadata["audit"]["sampled_rows"])
            self.assertTrue(assignments.source_row.is_unique)
            self.assertEqual(int((assignments.cluster == -1).sum()), metrics["noise_count"])
            self.assertEqual(sum(t["size"] for t in topics) + metrics["noise_count"], len(assignments))
            self.assertEqual(len(metadata["input_sha256"]), 64)
            self.assertTrue((output / "clusters.png").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
