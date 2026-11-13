import json
import unittest
from pathlib import Path


class AnalysisInputsTests(unittest.TestCase):
    def load_data(self):
        return json.loads((Path(__file__).resolve().parents[1] / "fixtures/analysis/input.json").read_text())

    def test_censored_service(self):
        data = self.load_data()
        self.assertTrue(any(row['served_hours'] == 0 and row['requested_hours'] > 0 for row in data['tasks']))

    def test_reassigned_task(self):
        data = self.load_data()
        self.assertTrue(any('replaces' in row for row in data['tasks']))

    def test_unfinished_tasks(self):
        data = self.load_data()
        self.assertTrue(any(row['closed_at'] is None for row in data['tasks']))

    def test_source_versions_and_cutoff(self):
        data = self.load_data()
        self.assertEqual(data["schema_version"], 1)
        self.assertTrue(data["source_versions"])
        self.assertIn("+00:00", data["as_of"])
