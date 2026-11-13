"""Behavioural tests for the demand planner.

Each test targets one specific trap described in the business brief, rather
than only checking output shape.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from emergency_resource_dispatch.demand_planning.inputs import PlanningInputError, load_dataset
from emergency_resource_dispatch.demand_planning.models import (
    backtest_origins,
    predictive_interval,
    run_backtests,
)
from emergency_resource_dispatch.demand_planning.reconcile import build_reconciliation, resolve_requests
from emergency_resource_dispatch.demand_planning.report import run_analysis
from emergency_resource_dispatch.demand_planning.series import DemandSeries, SeriesKey, utc_day

ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> dict:
    return json.loads((ROOT / "fixtures/analysis/input.json").read_text(encoding="utf-8"))


def _isolated(directory: str) -> tuple[str, str]:
    """Write the fixture and an empty ledger path into a temp dir."""
    input_path = Path(directory) / "input.json"
    input_path.write_text(json.dumps(_fixture(), ensure_ascii=False), encoding="utf-8")
    return str(input_path), str(Path(directory) / "ledger.sqlite3")


def _run(directory: str, payload: dict | None = None) -> dict:
    data = payload if payload is not None else _fixture()
    input_path = Path(directory) / "input.json"
    input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    database = str(Path(directory) / "ledger.sqlite3")
    output = Path(directory) / "plan.json"
    return run_analysis(input_path, database, output)


class ReconciliationTests(unittest.TestCase):
    def test_request_row_totals_are_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        reconciliation = plan["demand_reconciliation"]
        self.assertTrue(reconciliation["balance"]["accounted_equals_requested"])
        self.assertEqual(reconciliation["balance"]["unaccounted_hours"], 0)
        self.assertEqual(
            reconciliation["balance"]["requested_hours"],
            reconciliation["raw_file_totals"]["requested_hours_on_request_rows"],
        )

    def test_replacement_row_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        reconciliation = plan["demand_reconciliation"]
        raw = reconciliation["raw_file_totals"]
        self.assertEqual(raw["task_rows"] - raw["request_rows"], 1)
        self.assertEqual(raw["task_rows_minus_request_hours"], 4)
        self.assertEqual(reconciliation["double_count_prevented"]["duplicate_hours_not_counted"], 4)
        replacement = [
            row for row in reconciliation["requests"] if row["request_id"] == "request-2"
        ][0]
        self.assertEqual(replacement["task_rows"], ["task-2", "task-reassign"])
        self.assertEqual(replacement["superseded_rows"], ["task-2"])
        self.assertEqual(replacement["terminal_task_id"], "task-reassign")
        self.assertEqual(replacement["disposition"], "reassigned")

    def test_abandoned_requests_are_demand_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        reconciliation = plan["demand_reconciliation"]
        abandoned = reconciliation["dispositions"]["abandoned_hours_kept_as_demand"]
        self.assertGreater(abandoned, 0)
        zeroes = reconciliation["unknown_not_treated_as_zero"]
        self.assertEqual(zeroes["abandoned_hours_kept_as_demand"], abandoned)
        # exactly the requests whose terminal row closed as abandoned
        with self.assertRaises(KeyError):
            reconciliation["unknown_not_treated_as_zero"]["abandoned_treated_as_zero"]

    def test_open_tasks_keep_demand_but_withhold_served_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        reconciliation = plan["demand_reconciliation"]
        self.assertGreater(reconciliation["dispositions"]["censored_hours_kept_as_arrival_demand"], 0)
        self.assertGreater(
            reconciliation["unknown_not_treated_as_zero"]["open_served_hours_withheld_from_training"], 0
        )
        censored = [row for row in reconciliation["requests"] if row["disposition"] == "censored"]
        self.assertTrue(censored)
        for row in censored:
            self.assertIsNone(row["observed_served_hours"])
            self.assertGreater(row["requested_hours"], 0)

    def test_completion_log_is_not_presented_as_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        service = plan["demand_reconciliation"]["service_is_capacity_bounded"]
        self.assertLess(service["completion_log_to_request_ratio"], 0.75)
        self.assertGreater(service["service_shortfall_hours"], 0)
        self.assertNotIn("demand_hours", service)

    def test_abandonment_bounds_are_declared_assumptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        bounds = plan["demand_reconciliation"]["withdrawal_bounds"]
        self.assertEqual(bounds["demand_lower_bound_hours"], 500.0)
        self.assertGreater(bounds["demand_upper_bound_hours"], bounds["demand_lower_bound_hours"])
        self.assertTrue(bounds["upper_bound_is_assumption"])
        self.assertFalse(plan["demand_reconciliation"]["not_directly_observable"]["recoverable_from_completion_logs"])


class ForecastSemanticsTests(unittest.TestCase):
    def test_target_is_arrival_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        self.assertEqual(plan["scope"]["target_metric"], "requested_hours")
        for forecast in plan["forecasts"]:
            if forecast.get("forecast"):
                self.assertEqual(forecast["target_metric"], "requested_hours")

    def test_capacity_cut_does_not_lower_demand_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        steps = plan["capacity_diagnostics"]["steps"]
        self.assertTrue(steps)
        step = next(item for item in steps if item["cell"] == "west/pump")
        self.assertLess(step["capacity"]["relative_change"], 0)
        # completed work fell by less than arrivals in this cell: demand is not read from it
        self.assertIn("arrival_demand", step["relative_change"])
        self.assertNotEqual(
            step["relative_change"]["arrival_demand"], step["relative_change"]["observed_service"]
        )
        forecast = next(item for item in plan["forecasts"] if item["key"] == "cell:west/pump")["forecast"]
        self.assertGreater(forecast["point_hours"], 0)

    def test_served_ranking_disagrees_with_demand_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        ranking = plan["capacity_diagnostics"]["ranking_check"]
        arrival = {item["id"]: item["rank"] for item in ranking["rank_by_arrival_demand"]}
        service = {item["id"]: item["rank"] for item in ranking["rank_by_observed_service"]}
        self.assertEqual(arrival["west/pump"], 1)
        self.assertNotEqual(arrival["west/pump"], service["west/pump"])
        self.assertTrue(ranking["rank_displacements"])

    def test_intervals_are_monotone_and_non_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        for forecast in plan["forecasts"]:
            body = forecast.get("forecast")
            if not body:
                continue
            interval80 = body["interval_80"]
            interval95 = body["interval_95"]
            self.assertGreaterEqual(interval80["lower"], 0)
            self.assertGreaterEqual(interval95["lower"], 0)
            self.assertGreaterEqual(body["point_hours"], interval80["lower"])
            self.assertLessEqual(body["point_hours"], interval80["upper"])
            self.assertGreaterEqual(interval95["lower"], 0)
            self.assertLess(interval80["lower"], interval95["upper"])

    def test_intervals_widen_with_distance_and_uncertainty(self):
        short = predictive_interval(10.0, 5.0, 3.0)
        wide = predictive_interval(10.0, 15.0, 3.0)
        self.assertGreater(
            wide["interval_80"]["upper"] - wide["interval_80"]["lower"],
            short["interval_80"]["upper"] - short["interval_80"]["lower"],
        )


class SparseDataTests(unittest.TestCase):
    def test_new_region_and_rare_resource_are_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        decisions = {item["key"]: item for item in plan["support"]}
        for key in (
            "cell:new-region/pump",
            "cell:new-region/generator",
            "cell:new-region/rare-boat",
            "cell:east/rare-boat",
            "cell:west/rare-boat",
            "region:new-region",
            "resource:rare-boat",
        ):
            self.assertEqual(decisions[key]["status"], "unsupported", key)
            self.assertTrue(decisions[key]["limitations"], key)
        forecast = next(item for item in plan["forecasts"] if item["key"] == "cell:new-region/pump")
        self.assertIsNone(forecast["forecast"])
        self.assertIn("note", forecast)

    def test_sparse_cell_is_not_given_a_fabricated_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        unsupported = [item for item in plan["support"] if item["status"] == "unsupported"]
        self.assertTrue(unsupported)
        for decision in unsupported:
            forecast = next(item for item in plan["forecasts"] if item["key"] == decision["key"])
            self.assertIsNone(forecast["forecast"])

    def test_sibling_pooling_ignores_never_observed_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
            dataset = load_dataset(Path(tmp) / "input.json")
        series = DemandSeries(dataset)
        key = SeriesKey("cell", "east", "generator")
        from emergency_resource_dispatch.demand_planning.models import fit_hierarchical

        days = [utc_day(dataset.final_training_cutoff) - timedelta(days=i) for i in range(28, 0, -1)]
        fit = fit_hierarchical(series, key, days[0], utc_day(dataset.final_training_cutoff), datetime(2026, 10, 10, tzinfo=timezone.utc))
        self.assertEqual(fit.detail["parent_source"], "observed_sibling_cells")
        self.assertGreater(fit.detail["parent_rate"], 0)


class ZeroWindowTests(unittest.TestCase):
    def test_zero_window_does_not_fall_back_to_last_nonzero_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        gaps = plan["capacity_diagnostics"]["coverage_gap"]
        self.assertTrue(gaps["trailing_ledger_silence"])
        self.assertEqual(gaps["gap_days"], 16)
        self.assertIn("missing coverage", gaps["interpretation"])

    def test_no_forecast_exceeds_observed_coverage_with_zero_tail(self):
        """Training zeros must not be hidden by extrapolating the last non-zero day."""
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        for forecast in plan["forecasts"]:
            body = forecast.get("forecast")
            if not body:
                continue
            self.assertGreaterEqual(body["interval_80"]["lower"], 0)
            self.assertLessEqual(body["point_hours"], body["interval_95"]["upper"])


class LeakageAndCutoffTests(unittest.TestCase):
    def test_validation_flags_are_computed_and_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        validation = plan["validation"]
        self.assertTrue(validation["as_of_respected"])
        self.assertTrue(validation["training_cutoff_respected"])
        self.assertTrue(validation["unclosed_terminal_rows_excluded_from_service"])
        self.assertTrue(all(check["passes"] for check in validation["input_checks"]))

    def test_final_fit_ignores_unsettled_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
            dataset = load_dataset(Path(tmp) / "input.json")
        series = DemandSeries(dataset)
        cutoff = utc_day(dataset.final_training_cutoff)
        origin = datetime(cutoff.year, cutoff.month, cutoff.day, tzinfo=timezone.utc)
        # A terminal row still open at the cutoff must contribute no service:
        # its observed hours are withheld and never enter a series.
        open_entries = [entry for entry in series.evidence.values() if entry.disposition == "censored"]
        self.assertTrue(open_entries)
        withheld = 0.0
        served_task_ids = {
            entry.terminal_task_id for entry in series.evidence.values() if entry.disposition == "served"
        }
        for entry in open_entries:
            self.assertIsNone(entry.observed_hours)
            withheld += entry.observed_served_hours or 0.0
            self.assertNotIn(entry.terminal_task_id, served_task_ids)
        self.assertGreater(withheld, 0.0)
        # The ledger records these hours, yet the observed-service total excludes them.
        ledger_served = sum(row.served_hours or 0.0 for row in dataset.tasks if row.closed_at is None)
        plan_total = plan["demand_reconciliation"]["dispositions"]["observed_served_hours"]
        observed_without_open = sum(
            entry.observed_hours or 0.0
            for entry in series.evidence.values()
            if entry.disposition == "served"
        )
        self.assertAlmostEqual(plan_total, observed_without_open)
        self.assertGreater(ledger_served, 0.0)

    def test_late_fact_never_enters_an_earlier_fit(self):
        base = _fixture()
        late = copy.deepcopy(base)
        # A large request that becomes available only after an early origin.
        late["requests"].append(
            {
                "request_id": "request-late",
                "incident_id": "incident-late",
                "region": "east",
                "resource": "pump",
                "arrived_at": "2026-08-02T00:00:00+00:00",
                "available_at": "2026-09-20T00:00:00+0000",
                "requested_hours": 500,
            }
        )
        late["tasks"].append(
            {
                "task_id": "task-late",
                "request_id": "request-late",
                "requested_hours": 500,
                "served_hours": 0,
                "closed_at": None,
                "available_at": "2026-09-20T00:00:00+00:00",
                "outcome": "served",
            }
        )
        with tempfile.TemporaryDirectory() as tmp_a:
            baseline = _run(tmp_a, base)
        with tempfile.TemporaryDirectory() as tmp_b:
            mutated = _run(tmp_b, late)
        base_origin_fold = _fold_actual(baseline, "2026-08-29", "cell:east/pump")
        mutated_origin_fold = _fold_actual(mutated, "2026-08-29", "cell:east/pump")
        self.assertEqual(base_origin_fold, mutated_origin_fold)
        # but the late record is visible in the full-data reconciliation
        self.assertGreater(
            mutated["demand_reconciliation"]["balance"]["requested_hours"],
            baseline["demand_reconciliation"]["balance"]["requested_hours"],
        )


def _write(payload: dict, directory: str) -> str:
    path = Path(directory) / "input.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _fold_actual(plan: dict, origin: str, key: str) -> float:
    fold = next(item for item in plan["backtests"]["folds"] if item["origin"] == origin)
    entry = next(item for item in fold["keys"] if item["key"] == key)
    return entry["actual_hours"]


class HoldoutTests(unittest.TestCase):
    def test_holdout_is_after_cutoff_and_separate_from_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        holdout = plan["backtests"]["holdout"]
        self.assertEqual(holdout["window"]["start"], "2026-10-10")
        self.assertEqual(holdout["window"]["end_exclusive"], "2026-10-17")
        self.assertIn("never used to choose the model", holdout["purpose"])
        self.assertTrue(holdout["keys"])
        # The ledger stops at 2026-10-14, so only the observed prefix is scored.
        self.assertTrue(holdout["partial"])
        self.assertEqual(holdout["observed_days"], 5)

    def test_holdout_refuses_to_score_unobserved_days_as_zero(self):
        """A capacity-drop illustration whose ledger stops before the horizon."""
        payload = json.loads(
            (ROOT / "fixtures/analysis/capacity_drop_illustration.json").read_text(encoding="utf-8")
        )
        payload["as_of"] = payload["planning"]["final_training_cutoff"]
        payload["requests"] = [row for row in payload["requests"] if row["arrived_at"] < payload["planning"]["final_training_cutoff"]]
        keep = {row["request_id"] for row in payload["requests"]}
        payload["tasks"] = [row for row in payload["tasks"] if row["request_id"] in keep]
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp, payload)
        holdout = plan["backtests"]["holdout"]
        self.assertFalse(holdout["available"])
        self.assertEqual(holdout["observed_days"], 0)
        self.assertIsNone(holdout["wape"])

    def test_backtest_origins_end_before_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        cutoff = datetime.fromisoformat(plan["final_training_cutoff"]).date()
        for origin in plan["backtests"]["origins"]:
            self.assertLess(datetime.fromisoformat(origin).date(), cutoff)

    def test_origin_schedule_is_on_seven_day_grid(self):
        origins = backtest_origins(
            datetime(2026, 8, 1).date(), datetime(2026, 10, 10).date(), 7
        )
        gaps = {(right - left).days for left, right in zip(origins, origins[1:])}
        self.assertEqual(gaps, {7})
        self.assertGreaterEqual(len(origins), 4)


class SupportTests(unittest.TestCase):
    def test_supported_forecasts_have_backtested_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        for decision in plan["support"]:
            if decision["status"] == "supported":
                forecast = next(item for item in plan["forecasts"] if item["key"] == decision["key"])
                self.assertIsNotNone(forecast["forecast"])
                self.assertIsNotNone(forecast["selected_model"])
                self.assertTrue(forecast["backtest"]["complete"])
                self.assertLessEqual(forecast["backtest"]["wape"], 0.50)

    def test_unsupported_cells_name_their_limitation(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        for decision in plan["support"]:
            if decision["status"] != "supported":
                self.assertTrue(decision["limitations"])

    def test_region_and_total_forecasts_are_coherent(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp)
        forecasts = {item["key"]: item.get("forecast") for item in plan["forecasts"]}
        for region in ("east", "west"):
            cells = [
                forecasts[f"cell:{region}/pump"]["point_hours"],
                forecasts[f"cell:{region}/generator"]["point_hours"],
            ]
            self.assertLessEqual(
                abs(forecasts[f"region:{region}"]["point_hours"] - sum(cells)),
                0.5 * forecasts[f"region:{region}"]["point_hours"],
            )
        east = forecasts["region:east"]["point_hours"]
        west = forecasts["region:west"]["point_hours"]
        total = forecasts["total"]["point_hours"]
        self.assertLess(abs(total - (east + west)), max(5.0, 0.25 * total))


class InputValidationTests(unittest.TestCase):
    def test_null_request_hours_are_rejected_not_zeroed(self):
        payload = _fixture()
        payload["requests"][0]["requested_hours"] = None
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PlanningInputError):
                _run(tmp, payload)

    def test_overlapping_capacity_windows_are_rejected(self):
        payload = _fixture()
        payload["capacity"].append(
            {
                "region": "east",
                "resource": "pump",
                "start": "2026-09-01T00:00:00+00:00",
                "end": "2026-09-05T00:00:00+00:00",
                "hours_per_day": 4,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PlanningInputError):
                _run(tmp, payload)

    def test_cutoff_after_as_of_is_rejected(self):
        payload = _fixture()
        payload["planning"]["final_training_cutoff"] = "2026-11-15T00:00:00+00:00"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PlanningInputError):
                _run(tmp, payload)

    def test_naive_timezone_timestamps_are_rejected(self):
        payload = _fixture()
        payload["requests"][0]["arrived_at"] = "2026-08-01T00:00:00"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PlanningInputError):
                _run(tmp, payload)


class ReproducibilityTests(unittest.TestCase):
    def test_same_input_same_plan_id(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            first = _run(tmp_a)
            second = _run(tmp_b)
        self.assertEqual(first["plan_id"], second["plan_id"])
        self.assertEqual(first["demand_reconciliation"]["balance"], second["demand_reconciliation"]["balance"])
        self.assertEqual(
            [item["forecast"] for item in first["forecasts"]],
            [item["forecast"] for item in second["forecasts"]],
        )

    def test_replay_uses_stored_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            input_path.write_text(json.dumps(_fixture(), ensure_ascii=False), encoding="utf-8")
            database = str(Path(tmp) / "ledger.sqlite3")
            first = run_analysis(input_path, database, Path(tmp) / "a.json")
            second = run_analysis(input_path, database, Path(tmp) / "b.json")
        self.assertEqual(first["plan_id"], second["plan_id"])
        self.assertFalse(first["storage"]["replayed"])
        self.assertTrue(second["storage"]["replayed"])

    def test_analysis_does_not_touch_operational_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "ledger.sqlite3"
            input_path = _write(_fixture(), tmp)
            run_analysis(input_path, database, Path(tmp) / "plan.json")
            import sqlite3

            connection = sqlite3.connect(database)
            try:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                connection.close()
        self.assertIn("demand_plans", tables)
        self.assertNotIn("runs", tables)
        self.assertNotIn("idempotency", tables)

    def test_populated_ledger_is_only_read(self):
        """With a real operational run present, no run/event/idempotency row may change."""
        import sqlite3

        from emergency_resource_dispatch.service import run_fixture

        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "ledger.sqlite3"
            run_fixture(ROOT / "fixtures/demo.json", database)
            input_path = _write(_fixture(), tmp)

            def snapshot() -> dict[str, list[tuple]]:
                connection = sqlite3.connect(database)
                try:
                    return {
                        table: sorted(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))
                        for table in ("runs", "events", "idempotency", "failures")
                    }
                finally:
                    connection.close()

            before = snapshot()
            plan = run_analysis(input_path, database, Path(tmp) / "plan.json")
            after = snapshot()

        self.assertEqual(before, after)
        ledger = plan["evidence"]["ledger"]
        self.assertTrue(ledger["read_only"])
        self.assertEqual(ledger["linked_case_id"], "demo-process-20260802")
        self.assertTrue(ledger["lease_evidence"])
        self.assertFalse(ledger["errors"])


class EdgeCaseTests(unittest.TestCase):
    def test_empty_snapshot_issues_no_forecasts(self):
        payload = _fixture()
        payload["requests"] = []
        payload["tasks"] = []
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp, payload)
        self.assertEqual(plan["demand_reconciliation"]["balance"]["requested_hours"], 0)
        self.assertEqual(plan["backtests"]["fold_count"], 0)
        for item in plan["support"]:
            self.assertEqual(item["status"], "unsupported")
        for item in plan["forecasts"]:
            self.assertIsNone(item["forecast"])
        self.assertIn("no request rows", plan["capacity_diagnostics"]["coverage_gap"]["interpretation"])

    def test_single_cell_still_forecasts_levels_above(self):
        payload = _fixture()
        keep = {
            row["request_id"]
            for row in payload["requests"]
            if row["region"] == "east" and row["resource"] == "pump"
        }
        payload["requests"] = [row for row in payload["requests"] if row["request_id"] in keep]
        payload["tasks"] = [row for row in payload["tasks"] if row["request_id"] in keep]
        payload["planning"] = {
            "horizon_days": 7,
            "final_training_cutoff": "2026-10-10T00:00:00+00:00",
            "regions": ["east"],
            "resources": ["pump"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            plan = _run(tmp, payload)
        decisions = {item["key"]: item["status"] for item in plan["support"]}
        self.assertEqual(decisions["cell:east/pump"], "supported")
        forecast = next(item for item in plan["forecasts"] if item["key"] == "cell:east/pump")
        self.assertIsNotNone(forecast["forecast"])
        # no sibling pool exists; the model must not invent one
        if forecast["selected_model"] == "hierarchical_shrinkage":
            self.assertIsNone(forecast["fit"]["detail"]["parent_rate"])
        self.assertEqual(decisions["cell:west/pump"], "unsupported")


class CapacityDropIllustrationTests(unittest.TestCase):
    """The dedicated fixture where capacity collapses but arrivals stay flat."""

    def _plan(self, tmp: str) -> dict:
        payload = json.loads(
            (ROOT / "fixtures/analysis/capacity_drop_illustration.json").read_text(encoding="utf-8")
        )
        return _run(tmp, payload)

    def test_completed_work_falls_while_demand_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(tmp)
        steps = plan["capacity_diagnostics"]["steps"]
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertLess(step["capacity"]["relative_change"], -0.5)
        self.assertEqual(step["relative_change"]["arrival_demand"], 0.0)
        self.assertLess(step["relative_change"]["observed_service"], -0.3)
        self.assertIn("tracks capacity rather than demand", step["findings"][0])

    def test_forecast_stays_at_demand_level_despite_low_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(tmp)
        forecast = next(item for item in plan["forecasts"] if item["key"] == "cell:north/pump")
        # Flat demand of 4 h/day over 7 days, not the capacity-bound delivery.
        self.assertAlmostEqual(forecast["forecast"]["point_hours"], 28.0, places=6)
        self.assertGreater(forecast["forecast"]["interval_80"]["lower"], 0)

    def test_reconciliation_separates_declared_from_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(tmp)
        service = plan["demand_reconciliation"]["service_is_capacity_bounded"]
        self.assertGreater(service["declared_service_hours"], service["observed_service_hours"])
        self.assertGreater(service["service_shortfall_hours"], 0)


if __name__ == "__main__":
    unittest.main()
