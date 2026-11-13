"""Read and validate the read-only planning dataset.

The planner consumes a snapshot document (requests, tasks, capacity, calendar,
planning hints).  It never mutates the document and never calls dispatch.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class PlanningInputError(ValueError):
    """Raised when the snapshot cannot be interpreted without guessing."""


def _parse_time(value: Any, field: str, issues: list[str]) -> datetime | None:
    if value is None:
        issues.append(f"{field}: missing timestamp")
        return None
    if not isinstance(value, str):
        issues.append(f"{field}: timestamp must be an ISO 8601 string")
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        issues.append(f"{field}: invalid ISO 8601 timestamp {value!r}")
        return None
    if parsed.tzinfo is None:
        issues.append(f"{field}: timestamp must carry a timezone offset")
        return None
    return parsed.astimezone(timezone.utc)


def _parse_day(value: Any, field: str, issues: list[str]) -> date | None:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{field}: day must be a non-empty string")
        return None
    try:
        if "T" in value:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                issues.append(f"{field}: day timestamp must carry a timezone offset")
                return None
            return parsed.astimezone(timezone.utc).date()
        return date.fromisoformat(value)
    except ValueError:
        issues.append(f"{field}: invalid day {value!r}")
        return None


def _parse_hours(value: Any, field: str, issues: list[str], required: bool = True) -> float | None:
    if value is None:
        if required:
            issues.append(f"{field}: null hours are unknown, not zero")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(f"{field}: hours must be numeric")
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        issues.append(f"{field}: hours must be finite and non-negative")
        return None
    return number


def _parse_string(value: Any, field: str, issues: list[str], required: bool = True) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            issues.append(f"{field}: value is required")
        return None
    if not isinstance(value, str):
        issues.append(f"{field}: value must be a string")
        return None
    return value.strip()


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    incident_id: str | None
    region: str
    resource: str
    arrived_at: datetime
    available_at: datetime
    requested_hours: float


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    request_id: str
    replaces: str | None
    requested_hours: float | None
    served_hours: float | None
    closed_at: datetime | None
    available_at: datetime
    outcome: str


@dataclass(frozen=True, slots=True)
class CapacityWindow:
    region: str
    resource: str
    start: datetime
    end: datetime
    hours_per_day: float


@dataclass(frozen=True, slots=True)
class CalendarDay:
    day: date
    holiday: bool
    available_at: datetime


@dataclass(frozen=True, slots=True)
class PlanningDataset:
    as_of: datetime
    source_versions: dict[str, Any]
    baseline_case_id: str | None
    requests: list[Request]
    tasks: list[Task]
    capacity: list[CapacityWindow]
    calendar: list[CalendarDay]
    horizon_days: int
    final_training_cutoff: datetime
    planning_regions: list[str]
    planning_resources: list[str]
    raw: dict[str, Any]
    checks: list[dict[str, Any]]


def _unique(values: list[str], label: str, issues: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            issues.append(f"{label}: duplicate identifier {value!r}")
        seen.add(value)


def load_dataset(path: str | Path) -> PlanningDataset:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanningInputError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanningInputError(f"input file is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanningInputError("analysis input must be a single JSON object")

    issues: list[str] = []
    checks: list[dict[str, Any]] = []

    as_of = _parse_time(raw.get("as_of"), "as_of", issues)
    source_versions = raw.get("source_versions")
    if not isinstance(source_versions, dict) or not source_versions:
        issues.append("source_versions: non-empty object is required")
        source_versions = {}
    baseline_case_id = raw.get("baseline_case_id")
    if baseline_case_id is not None and not isinstance(baseline_case_id, str):
        issues.append("baseline_case_id: must be a string when present")
        baseline_case_id = None

    planning = raw.get("planning")
    if not isinstance(planning, dict):
        issues.append("planning: object is required")
        planning = {}
    horizon_days = planning.get("horizon_days")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
        issues.append("planning.horizon_days: positive integer is required")
        horizon_days = 0
    final_cutoff = _parse_time(planning.get("final_training_cutoff"), "planning.final_training_cutoff", issues)
    planning_regions = planning.get("regions")
    if not isinstance(planning_regions, list) or not planning_regions:
        issues.append("planning.regions: non-empty list is required")
        planning_regions = []
    planning_resources = planning.get("resources")
    if not isinstance(planning_resources, list) or not planning_resources:
        issues.append("planning.resources: non-empty list is required")
        planning_resources = []
    for label, values in (("planning.regions", planning_regions), ("planning.resources", planning_resources)):
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{label}[{index}]: non-empty string is required")

    # ---------------- requests -------------
    requests_raw = raw.get("requests")
    if not isinstance(requests_raw, list):
        issues.append("requests: list is required")
        requests_raw = []
    requests: list[Request] = []
    for index, row in enumerate(requests_raw):
        field = f"requests[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{field}: object is required")
            continue
        request_id = _parse_string(row.get("request_id"), f"{field}.request_id", issues)
        incident_id = _parse_string(row.get("incident_id"), f"{field}.incident_id", issues, required=False)
        region = _parse_string(row.get("region"), f"{field}.region", issues)
        resource = _parse_string(row.get("resource"), f"{field}.resource", issues)
        arrived_at = _parse_time(row.get("arrived_at"), f"{field}.arrived_at", issues)
        available_at = (
            _parse_time(row.get("available_at"), f"{field}.available_at", issues)
            if "available_at" in row
            else arrived_at
        )
        hours = _parse_hours(row.get("requested_hours"), f"{field}.requested_hours", issues)
        if None in (request_id, region, resource, arrived_at, available_at, hours):
            continue
        if available_at < arrived_at:
            issues.append(f"{field}.available_at: cannot precede arrived_at")
            continue
        requests.append(Request(request_id, incident_id, region, resource, arrived_at, available_at, hours))
    _unique([item.request_id for item in requests], "requests.request_id", issues)

    # ---------------- tasks -------------
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list):
        issues.append("tasks: list is required")
        tasks_raw = []
    tasks: list[Task] = []
    for index, row in enumerate(tasks_raw):
        field = f"tasks[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{field}: object is required")
            continue
        task_id = _parse_string(row.get("task_id"), f"{field}.task_id", issues)
        request_id = _parse_string(row.get("request_id"), f"{field}.request_id", issues)
        replaces = _parse_string(row.get("replaces"), f"{field}.replaces", issues, required=False)
        requested_hours = _parse_hours(row.get("requested_hours"), f"{field}.requested_hours", issues)
        served_hours = _parse_hours(row.get("served_hours"), f"{field}.served_hours", issues, required=False)
        closed_at = _parse_time(row.get("closed_at"), f"{field}.closed_at", issues) if row.get("closed_at") else None
        available_at = (
            _parse_time(row.get("available_at"), f"{field}.available_at", issues)
            if "available_at" in row
            else (closed_at or (requests and requests[0].available_at))
        )
        outcome = _parse_string(row.get("outcome"), f"{field}.outcome", issues, required=False) or "unknown"
        if None in (task_id, request_id, requested_hours, available_at):
            continue
        tasks.append(
            Task(task_id, request_id, replaces, requested_hours, served_hours, closed_at, available_at, outcome.lower())
        )
    _unique([item.task_id for item in tasks], "tasks.task_id", issues)

    request_ids = {item.request_id for item in requests}
    task_ids = {item.task_id for item in tasks}
    for task in tasks:
        if task.request_id not in request_ids:
            issues.append(f"tasks.{task.task_id}.request_id: unknown request {task.request_id!r}")
        if task.replaces and task.replaces not in task_ids:
            issues.append(f"tasks.{task.task_id}.replaces: unknown task {task.replaces!r}")
        if task.replaces and task.replaces == task.task_id:
            issues.append(f"tasks.{task.task_id}.replaces: a task cannot replace itself")

    # ---------------- capacity -------------
    capacity_raw = raw.get("capacity")
    if not isinstance(capacity_raw, list):
        issues.append("capacity: list is required")
        capacity_raw = []
    capacity: list[CapacityWindow] = []
    for index, row in enumerate(capacity_raw):
        field = f"capacity[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{field}: object is required")
            continue
        region = _parse_string(row.get("region"), f"{field}.region", issues)
        resource = _parse_string(row.get("resource"), f"{field}.resource", issues)
        start = _parse_time(row.get("start"), f"{field}.start", issues)
        end = _parse_time(row.get("end"), f"{field}.end", issues)
        hours_per_day = _parse_hours(row.get("hours_per_day"), f"{field}.hours_per_day", issues)
        if None in (region, resource, start, end, hours_per_day):
            continue
        if end <= start:
            issues.append(f"{field}: end must be after start")
            continue
        capacity.append(CapacityWindow(region, resource, start, end, hours_per_day))
    by_cell: dict[tuple[str, str], list[CapacityWindow]] = {}
    for window in capacity:
        by_cell.setdefault((window.region, window.resource), []).append(window)
    for key, windows in by_cell.items():
        windows = sorted(windows, key=lambda item: item.start)
        for left, right in zip(windows, windows[1:]):
            if left.end > right.start:
                issues.append(
                    f"capacity[{key[0]}/{key[1]}]: overlapping windows {left.start.isoformat()}..{left.end.isoformat()} and {right.start.isoformat()}..{right.end.isoformat()}"
                )

    # ---------------- calendar -------------
    calendar_raw = raw.get("calendar")
    if not isinstance(calendar_raw, list):
        issues.append("calendar: list is required")
        calendar_raw = []
    calendar: list[CalendarDay] = []
    for index, row in enumerate(calendar_raw):
        field = f"calendar[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{field}: object is required")
            continue
        day = _parse_day(row.get("day"), f"{field}.day", issues)
        holiday = row.get("holiday")
        if not isinstance(holiday, bool):
            issues.append(f"{field}.holiday: boolean is required")
            holiday = False
        available_at = (
            _parse_time(row.get("available_at"), f"{field}.available_at", issues)
            if "available_at" in row
            else (as_of if as_of is not None else datetime(1970, 1, 1, tzinfo=timezone.utc))
        )
        if day is None or available_at is None:
            continue
        calendar.append(CalendarDay(day, holiday, available_at))
    _unique([item.day.isoformat() for item in calendar], "calendar.day", issues)

    if issues:
        raise PlanningInputError("input validation failed:\n - " + "\n - ".join(issues))

    assert as_of is not None and final_cutoff is not None
    checks.append(
        {
            "id": "time_order",
            "passes": final_cutoff <= as_of,
            "detail": {"final_training_cutoff": final_cutoff.isoformat(), "as_of": as_of.isoformat()},
        }
    )
    checks.append(
        {
            "id": "records_before_as_of",
            "passes": all(item.available_at <= as_of for item in requests)
            and all(item.available_at <= as_of for item in tasks),
            "detail": "every request/task fact is available by as_of",
        }
    )
    if final_cutoff > as_of:
        raise PlanningInputError("planning.final_training_cutoff cannot be later than as_of")

    return PlanningDataset(
        as_of=as_of,
        source_versions=source_versions,
        baseline_case_id=baseline_case_id,
        requests=requests,
        tasks=tasks,
        capacity=capacity,
        calendar=calendar,
        horizon_days=horizon_days,
        final_training_cutoff=final_cutoff,
        planning_regions=[item.strip() for item in planning_regions],
        planning_resources=[item.strip() for item in planning_resources],
        raw=raw,
        checks=checks,
    )
