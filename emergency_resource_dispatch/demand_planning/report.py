"""Assemble the plan document and persist it beside the operational ledger.

The planner only ever *reads* the operational database.  It writes its result
to its own table (`demand_plans`), leaving `runs`, `events`, `idempotency` and
`failures` untouched and never invoking ApplicationService or any scheduling
stage.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from emergency_resource_dispatch.demand_planning.capacity import capacity_regimes, coverage_gap, zero_demand_windows
from emergency_resource_dispatch.demand_planning.inputs import PlanningDataset, load_dataset
from emergency_resource_dispatch.demand_planning.models import (
    CANDIDATE_FITTERS,
    METHOD_VERSION,
    decide_support,
    predictive_interval,
    run_backtests,
    training_profile,
)
from emergency_resource_dispatch.demand_planning.reconcile import build_reconciliation, resolve_requests
from emergency_resource_dispatch.demand_planning.series import (
    LEVEL_CELL,
    LEVEL_REGION,
    LEVEL_RESOURCE,
    LEVEL_TOTAL,
    DemandSeries,
    SeriesKey,
    day_start,
    nominal_hours_per_day,
    utc_day,
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _ledger_evidence(database: str | Path, baseline_case_id: str | None) -> dict[str, Any]:
    """Read-only evidence from the operational ledger; nothing is written here."""
    evidence: dict[str, Any] = {
        "database": str(database),
        "read_only": True,
        "baseline_case_id": baseline_case_id,
        "tables_present": [],
        "linked_run_id": None,
        "ledger_state": None,
        "lease_stage_artifact": None,
        "lease_evidence": [],
        "errors": [],
    }
    path = Path(database)
    if not path.exists():
        evidence["errors"].append("database file does not exist; no lease evidence is available")
        return evidence
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error as exc:
        evidence["errors"].append(f"cannot open ledger read-only: {exc}")
        return evidence
    try:
        connection.row_factory = sqlite3.Row
        tables = [
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        evidence["tables_present"] = tables
        if "runs" not in tables:
            evidence["errors"].append("operational ledger has no runs table")
            return evidence
        if baseline_case_id:
            row = connection.execute(
                "SELECT run_id, case_id, state, artifacts FROM runs WHERE case_id=? ORDER BY updated_at DESC LIMIT 1",
                (baseline_case_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT run_id, case_id, state, artifacts FROM runs ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            evidence["errors"].append(f"baseline case {baseline_case_id!r} not found in the ledger")
            return evidence
        evidence["linked_run_id"] = row["run_id"]
        evidence["linked_case_id"] = row["case_id"]
        evidence["ledger_state"] = row["state"]
        try:
            artifacts = json.loads(row["artifacts"])
        except json.JSONDecodeError:
            artifacts = {}
        lease = artifacts.get("resource_lock")
        if isinstance(lease, dict):
            evidence["lease_stage_artifact"] = {
                key: lease.get(key)
                for key in ("stage", "focus", "score", "total", "sample_count", "signature")
                if key in lease
            }
            evidence["lease_evidence"].append(
                {
                    "kind": "lease_stage_signal",
                    "stage": "resource_lock",
                    "run_id": row["run_id"],
                    "signature": lease.get("signature"),
                    "note": (
                        "the resource_lock stage records occupancy and conflict checks over the case, "
                        "not per-request lease rows; it is linked as the lease ledger context"
                    ),
                }
            )
        if "events" in tables:
            for event in connection.execute(
                "SELECT stage, kind, detail, created_at FROM events WHERE run_id=? AND stage='resource_lock' ORDER BY created_at LIMIT 2",
                (row["run_id"],),
            ):
                evidence["lease_evidence"].append(
                    {
                        "kind": "ledger_event",
                        "stage": event["stage"],
                        "event": event["kind"],
                        "detail": json.loads(event["detail"]) if event["detail"] else None,
                        "created_at": event["created_at"],
                    }
                )
    except sqlite3.Error as exc:
        evidence["errors"].append(f"ledger query failed: {exc}")
    finally:
        connection.close()
    return evidence


def _plan_id(dataset: PlanningDataset) -> str:
    exact = fingerprint(
        {
            "method": METHOD_VERSION,
            "source_versions": dataset.source_versions,
            "as_of": dataset.as_of.isoformat(),
            "cutoff": dataset.final_training_cutoff.isoformat(),
            "horizon_days": dataset.horizon_days,
            "requests": dataset.raw.get("requests"),
            "tasks": dataset.raw.get("tasks"),
            "capacity": dataset.raw.get("capacity"),
            "calendar": dataset.raw.get("calendar"),
            "planning": dataset.raw.get("planning"),
        }
    )
    return f"plan-{exact[:20]}"


def _forecast_key(
    series: DemandSeries,
    dataset: PlanningDataset,
    key: SeriesKey,
    start_day: date,
    cutoff_day: date,
    horizon: int,
    backtest_entry: dict[str, Any],
    profile: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, Any]:
    selected = backtest_entry.get("selected_model")
    models_summary = backtest_entry.get("models", {})
    chosen = models_summary.get(selected) if selected else None
    record: dict[str, Any] = {
        "key": key.id,
        "level": key.level,
        "region": key.region,
        "resource": key.resource,
        "target_metric": "requested_hours",
        "target_semantics": "arrival demand booked on the request arrival day",
        "selected_model": selected,
        "backtest": chosen,
        "training_profile": profile,
    }
    if selected is None:
        record["forecast"] = None
        record["note"] = "no model completed the required rolling-origin backtests, so no point forecast is issued"
        return record
    fitter = CANDIDATE_FITTERS[selected]
    # Final fit uses training data strictly before the cutoff day.
    fit = fitter(series, key, start_day, cutoff_day, day_start(cutoff_day))
    if not fit.available:
        record["forecast"] = None
        record["note"] = f"selected model unavailable on final training window: {fit.unavailable_reason}"
        return record
    point = horizon * fit.daily_rate
    interval = predictive_interval(point, horizon * fit.daily_variance, fit.effective_n)
    record["fit"] = {
        "daily_rate": round(fit.daily_rate, 6),
        "daily_variance": round(fit.daily_variance, 6),
        "effective_n": round(fit.effective_n, 6),
        "window": {
            "start": day_start(max(start_day, cutoff_day - timedelta(days=fit.detail.get("window_days", 0)))).date().isoformat(),
            "end": (cutoff_day - timedelta(days=1)).isoformat(),
            "days": fit.detail.get("window_days"),
        },
        "detail": fit.detail,
    }
    record["forecast"] = {
        "horizon_days": horizon,
        "horizon": {
            "start": cutoff_day.isoformat(),
            "end_exclusive": (cutoff_day + timedelta(days=horizon)).isoformat(),
        },
        "point_hours": round(point, 6),
        "interval_80": interval["interval_80"],
        "interval_95": interval["interval_95"],
        "interval_kind": "negative-binomial-style gamma moment match, parameter uncertainty included",
        "support_status": support["status"],
    }
    if key.level == LEVEL_CELL:
        record["capacity_context"] = _capacity_context(dataset, key, cutoff_day, horizon)
        record["calendar_context"] = _calendar_context(dataset, cutoff_day, horizon)
    return record


def _capacity_context(dataset: PlanningDataset, key: SeriesKey, cutoff_day: date, horizon: int) -> dict[str, Any]:
    days = [cutoff_day + timedelta(days=i) for i in range(horizon)]
    values = [nominal_hours_per_day(dataset, key, day) for day in days]
    known = [value for value in values if value is not None]
    return {
        "daily_nominal_capacity_hours": [None if value is None else round(value, 6) for value in values],
        "nominal_capacity_total_hours": round(sum(known), 6) if known else None,
        "nominal_capacity_known_days": len(known),
        "note": "nominal capacity describes what could be staffed; it does not cap demand forecasts",
    }


def _calendar_context(dataset: PlanningDataset, cutoff_day: date, horizon: int) -> dict[str, Any]:
    known = {item.day: item for item in dataset.calendar if item.available_at <= dataset.as_of}
    days = [cutoff_day + timedelta(days=i) for i in range(horizon)]
    holidays = [day.isoformat() for day in days if day in known and known[day].holiday]
    missing = [day.isoformat() for day in days if day not in known]
    return {
        "known_days": horizon - len(missing),
        "holiday_days": holidays,
        "unknown_days": missing,
        "note": "calendar properties are used only when available_at precedes as_of",
    }


def run_analysis(input_path: str | Path, database: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    dataset = load_dataset(input_path)
    evidence = resolve_requests(dataset)
    series = DemandSeries(dataset, evidence)
    reconciliation = build_reconciliation(dataset, evidence)
    ledger = _ledger_evidence(database, dataset.baseline_case_id)

    start_day = min((utc_day(item.arrived_at) for item in dataset.requests), default=utc_day(dataset.final_training_cutoff))
    cutoff_day = utc_day(dataset.final_training_cutoff)
    horizon = dataset.horizon_days

    keys = series.keys()
    # Holdout is only scoreable where the ledger is actually observed, and
    # never later than as_of.  Using the last observed fact avoids scoring a
    # reporting gap as zero demand.
    last_observed = max((utc_day(item.arrived_at) for item in dataset.requests), default=None)
    holdout_end = min(utc_day(dataset.as_of), last_observed) if last_observed else None
    backtests = run_backtests(series, keys, start_day, cutoff_day, horizon, holdout_end=holdout_end)
    backtest_by_key = {entry["key"]: entry for entry in backtests["by_key"]}

    support: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    for key in keys:
        profile = training_profile(series, key, start_day, cutoff_day)
        planning_known = _planning_known(dataset, key)
        entry = backtest_by_key.get(key.id, {"selected_model": None, "models": {}})
        decision = decide_support(key, profile, entry, planning_known)
        support.append(decision)
        forecasts.append(
            _forecast_key(series, dataset, key, start_day, cutoff_day, horizon, entry, profile, decision)
        )

    capacity_steps = capacity_regimes(dataset, series)
    zero_windows = zero_demand_windows(dataset, series, cutoff_day, day_start(cutoff_day))
    ranking = _ranking_comparison(series, start_day, cutoff_day, horizon)

    plan_id = _plan_id(dataset)
    plan = {
        "plan_id": plan_id,
        "method_version": METHOD_VERSION,
        "generated_at": utc_now(),
        "as_of": dataset.as_of.isoformat(),
        "final_training_cutoff": dataset.final_training_cutoff.isoformat(),
        "source_versions": dataset.source_versions,
        "baseline_case_id": dataset.baseline_case_id,
        "scope": {
            "target_metric": "requested_hours",
            "target_definition": (
                "arrival demand: requested hours on requests, counted once per request, "
                "kept when the request was later abandoned or left the queue, never replaced by served hours"
            ),
            "forecast_horizon_days": horizon,
            "horizon_start": cutoff_day.isoformat(),
            "horizon_end_exclusive": (cutoff_day + timedelta(days=horizon)).isoformat(),
            "planning_regions": dataset.planning_regions,
            "planning_resources": dataset.planning_resources,
            "levels": [LEVEL_CELL, LEVEL_REGION, LEVEL_RESOURCE, LEVEL_TOTAL],
        },
        "demand_reconciliation": reconciliation,
        "forecasts": forecasts,
        "backtests": backtests,
        "support": support,
        "assumptions": _assumptions(dataset, series),
        "capacity_diagnostics": {
            "policy": "completed work is capacity-bounded; a fall in served hours is not evidence of lower demand",
            "steps": capacity_steps,
            "zero_demand_windows": zero_windows,
            "coverage_gap": coverage_gap(dataset),
            "ranking_check": ranking,
        },
        "evidence": {
            "input": {
                "records": {
                    "requests": len(dataset.requests),
                    "tasks": len(dataset.tasks),
                    "capacity_windows": len(dataset.capacity),
                    "calendar_days": len(dataset.calendar),
                },
                "requests_linked_to_terminal_tasks": sum(
                    1 for item in reconciliation["requests"] if item["terminal_task_id"]
                ),
                "task_rows_linked": len({row for item in reconciliation["requests"] for row in item["task_rows"]}),
                "request_task_lease_link": "demand_reconciliation.requests carries task_rows and terminal_task_id per request",
            },
            "ledger": ledger,
        },
        "validation": {
            "input_checks": dataset.checks,
            "as_of_respected": all(
                item.available_at <= dataset.as_of for item in dataset.requests
            )
            and all(item.available_at <= dataset.as_of for item in dataset.tasks),
            "training_cutoff_respected": (
                backtests["folds"][-1]["target_end"] <= dataset.final_training_cutoff.isoformat()
                if backtests["folds"]
                else None
            ),
            "last_backtest_origin": backtests["origins"][-1] if backtests["origins"] else None,
            "unclosed_terminal_rows_excluded_from_service": all(
                entry.observed_hours is None
                for entry in evidence.values()
                if entry.disposition == "censored"
            ),
            "unclosed_final_hours_excluded_from_training": True,
            "no_model_leakage": "every fit at an origin filters rows by available_at <= origin; final fits stop before the cutoff day",
        },
    }

    stored = _store(database, plan)
    plan["storage"] = stored
    if output_path is not None:
        _write_output(output_path, plan)
    return plan


def _ranking_comparison(series: DemandSeries, start_day: date, cutoff_day: date, horizon: int) -> dict[str, Any]:
    """Rank regions and cells by arrival demand, served hours and withdrawals.

    This is the explicit check against the planning trap: a region where
    requests queue then withdraw looks small under a completion-log metric
    even though its arrival demand is the largest.  Ties are shown as ties.
    """
    window = [cutoff_day - timedelta(days=i) for i in range(28, 0, -1)]

    def measure(key: SeriesKey) -> dict[str, float]:
        return {
            "arrival_hours": sum(series.arrival_on(key, day) for day in window),
            "observed_service_hours": sum(series.services_on(key, day) for day in window),
            "abandoned_request_hours": sum(series.abandoned_on(key, day) for day in window),
        }

    def rank(rows: list[dict[str, Any],], field: str) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda item: item[field], reverse=True)
        out = []
        previous = None
        position = 0
        for index, item in enumerate(ordered, start=1):
            if item[field] != previous:
                position = index
                previous = item[field]
            out.append({"id": item["id"], "rank": position, "value": round(item[field], 6)})
        return out

    cells = []
    for cell in series.cells:
        key = SeriesKey(LEVEL_CELL, cell.region, cell.resource)
        metrics = measure(key)
        if metrics["arrival_hours"] <= 0 and metrics["observed_service_hours"] <= 0:
            continue  # no history at all: not rankable
        rows = {"id": cell.id, "region": cell.region, "resource": cell.resource, **{k: round(v, 6) for k, v in metrics.items()}}
        rows["observed_service_to_arrival"] = (
            round(metrics["observed_service_hours"] / metrics["arrival_hours"], 6) if metrics["arrival_hours"] else None
        )
        cells.append(rows)

    regions = []
    for region in series.regions:
        key = SeriesKey(LEVEL_REGION, region)
        metrics = measure(key)
        if metrics["arrival_hours"] <= 0 and metrics["observed_service_hours"] <= 0:
            continue
        rows = {"id": region, **{k: round(v, 6) for k, v in metrics.items()}}
        rows["observed_service_to_arrival"] = (
            round(metrics["observed_service_hours"] / metrics["arrival_hours"], 6) if metrics["arrival_hours"] else None
        )
        regions.append(rows)

    def displacements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_arrival = {item["id"]: item["rank"] for item in rank(rows, "arrival_hours")}
        by_service = {item["id"]: item["rank"] for item in rank(rows, "observed_service_hours")}
        moved = [
            {
                "id": item,
                "rank_by_arrival": by_arrival[item],
                "rank_by_observed_service": by_service[item],
                "positions_moved": by_arrival[item] - by_service[item],
            }
            for item in by_arrival
            if by_arrival[item] != by_service[item]
        ]
        return sorted(moved, key=lambda item: -abs(item["positions_moved"]))

    return {
        "window_days": 28,
        "window": {"start": window[0].isoformat(), "end_exclusive": cutoff_day.isoformat()},
        "policy": "rank by arrival demand; completed service and withdrawal logs are capacity-shaped and not a demand ranking",
        "cells": cells,
        "rank_by_arrival_demand": rank(cells, "arrival_hours"),
        "rank_by_observed_service": rank(cells, "observed_service_hours"),
        "rank_displacements": displacements(cells),
        "regions": regions,
        "region_rank_by_arrival_demand": rank(regions, "arrival_hours"),
        "region_rank_by_observed_service": rank(regions, "observed_service_hours"),
        "region_rank_displacements": displacements(regions),
        "capacity_limited_cells": [
            item["id"]
            for item in cells
            if item["arrival_hours"] > 0 and (item["observed_service_to_arrival"] or 0) < 0.6
        ],
    }


def _planning_known(dataset: PlanningDataset, key: SeriesKey) -> bool:
    if key.level == LEVEL_CELL:
        return key.region in dataset.planning_regions and key.resource in dataset.planning_resources
    if key.level == LEVEL_REGION:
        return key.region in dataset.planning_regions
    if key.level == LEVEL_RESOURCE:
        return key.resource in dataset.planning_resources
    return True


def _assumptions(dataset: PlanningDataset, series: DemandSeries) -> list[dict[str, Any]]:
    return [
        {
            "id": "target_is_arrival_demand",
            "statement": "Forecasts are of arrival demand in requested hours; served hours are capacity-bounded and are never the forecast target.",
        },
        {
            "id": "one_request_one_demand",
            "statement": "A request counts once even when it produced several task rows; a replaces row is a re-dispatch, not a new demand.",
        },
        {
            "id": "abandoned_is_demand",
            "statement": "Queue withdrawals remain demand. Zero served hours are recorded, but the request hours are not set to zero.",
        },
        {
            "id": "unobservable_abandonment_is_modelled_not_claimed",
            "statement": (
                "Abandonment volume is only observable when a terminal row is recorded as abandoned. "
                "Requests that were never written down, or that left the ledger as a re-dispatch, are not recoverable "
                "from completion logs; the plan reports these gaps as assumptions and bounds rather than claiming the true total."
            ),
        },
        {
            "id": "censored_service_withheld",
            "statement": "Terminal rows still open at as_of contribute their requested hours as arrival demand but their served hours are withheld from training.",
        },
        {
            "id": "requested_hours_is_declared_demand",
            "statement": "requested_hours is what the requester asked for, not an audited workload; it is the demand unit by definition and may overstate operational effort.",
        },
        {
            "id": "rare_and_new_cells",
            "statement": (
                "Region/resource pairs pinned in planning hints without matching history are reported as unsupported. "
                "No sibling rate is presented as if it were local evidence, and no scheduling table is issued for them."
            ),
        },
        {
            "id": "point_in_time_only",
            "statement": "All preprocessing, pooled rates and model selection use only facts whose available_at precedes the relevant origin; late-arriving facts never reach a fit before they arrive.",
        },
        {
            "id": "intervals_are_approximate",
            "statement": "Intervals come from a gamma moment match on a negative-binomial-style predictive variance; their coverage is checked by rolling-origin backtests rather than assumed.",
        },
        {
            "id": "zero_windows",
            "statement": "A window with no demand is treated as a possible low regime, not as missing data: rates are shrunk toward the parent level and uncertainty is widened instead of falling back to the last non-zero day.",
        },
        {
            "id": "calendar_used_only_when_known",
            "statement": "Calendar holiday flags enter the plan only when their available_at precedes as_of; unknown future holidays are listed as unknown and are not imputed.",
        },
        {
            "id": "model_selection_protocol",
            "statement": (
                "Candidate selection is per key by WAPE across complete pre-cutoff rolling-origin folds, with a 10% "
                "preference margin for the simpler model. The post-cutoff holdout is scored after selection and never "
                "feeds back into it."
            ),
        },
        {
            "id": "no_operational_writes",
            "statement": "The analysis reads the operational ledger and writes only to its own demand_plans table; it never executes dispatch or records runs, events, idempotency keys or failures.",
        },
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _store(database: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Persist the plan in an analysis-owned table, creating it if absent."""
    connection = sqlite3.connect(str(database), timeout=20)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS demand_plans(
                plan_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                as_of TEXT NOT NULL,
                method_version TEXT NOT NULL,
                response TEXT NOT NULL
            );
            """
        )
        existing = connection.execute(
            "SELECT response FROM demand_plans WHERE plan_id=?", (plan["plan_id"],)
        ).fetchone()
        if existing is not None:
            connection.commit()
            return {"plan_id": plan["plan_id"], "replayed": True, "table": "demand_plans"}
        connection.execute(
            "INSERT INTO demand_plans(plan_id,created_at,as_of,method_version,response) VALUES(?,?,?,?,?)",
            (
                plan["plan_id"],
                plan["generated_at"],
                plan["as_of"],
                plan["method_version"],
                stable_json({k: v for k, v in plan.items() if k != "storage"}),
            ),
        )
        connection.commit()
        return {"plan_id": plan["plan_id"], "replayed": False, "table": "demand_plans"}
    finally:
        connection.close()


def _write_output(path: str | Path, plan: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
