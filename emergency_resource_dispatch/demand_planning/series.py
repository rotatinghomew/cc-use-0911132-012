"""Daily demand aggregation with strict point-in-time availability gates.

Two different clocks are kept apart:

* ``requested_hours`` is booked on the request's *arrival day*, but a request
  row only enters any computation once ``available_at`` has passed.
* ``served_hours`` is booked on the terminal task's *closed day*, and only
  closed terminal rows whose ``available_at`` has passed are used.

Training windows always end strictly before the forecast origin.  Records that
were not yet available at an origin can never influence a fit at that origin.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from emergency_resource_dispatch.demand_planning.inputs import PlanningDataset
from emergency_resource_dispatch.demand_planning.reconcile import (
    DISP_SERVED,
    RequestEvidence,
    resolve_requests,
)

LEVEL_CELL = "cell"
LEVEL_REGION = "region"
LEVEL_RESOURCE = "resource"
LEVEL_TOTAL = "total"


def utc_day(value: datetime) -> date:
    return value.astimezone(timezone.utc).date()


def day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True, order=True)
class SeriesKey:
    level: str
    region: str | None = None
    resource: str | None = None

    @property
    def id(self) -> str:
        if self.level == LEVEL_CELL:
            return f"cell:{self.region}/{self.resource}"
        if self.level == LEVEL_REGION:
            return f"region:{self.region}"
        if self.level == LEVEL_RESOURCE:
            return f"resource:{self.resource}"
        return "total"

    @property
    def label(self) -> str:
        return self.id

    def matches(self, region: str, resource: str) -> bool:
        if self.level == LEVEL_CELL:
            return self.region == region and self.resource == resource
        if self.level == LEVEL_REGION:
            return self.region == region
        if self.level == LEVEL_RESOURCE:
            return self.resource == resource
        return True


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    region: str
    resource: str

    @property
    def id(self) -> str:
        return f"{self.region}/{self.resource}"


HISTORY_AVAILABILITY = "available_at <= origin"
SERVICE_METRIC = "served_hours"


class DemandSeries:
    """Daily arrival-demand and observed-service series for the whole dataset."""

    def __init__(self, dataset: PlanningDataset, evidence: dict[str, RequestEvidence] | None = None):
        self.dataset = dataset
        self.evidence = evidence if evidence is not None else resolve_requests(dataset)
        self.cells = [
            Cell(region, resource)
            for region, resource in sorted(
                {(item.region, item.resource) for item in dataset.requests}
                | {(item.region, item.resource) for item in dataset.capacity}
                | {(region, resource) for region in dataset.planning_regions for resource in dataset.planning_resources}
            )
        ]
        self.regions = sorted({item.region for item in self.cells})
        self.resources = sorted({item.resource for item in self.cells})
        self._arrival: dict[Cell, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        self._arrival_count: dict[Cell, dict[date, int]] = defaultdict(lambda: defaultdict(int))
        self._service: dict[Cell, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        self._abandoned: dict[Cell, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        # (day, available_at) pairs so point-in-time views can filter on arrival knowledge.
        self._arrival_knowledge: dict[Cell, list[tuple[date, datetime, float]]] = defaultdict(list)
        self._service_knowledge: dict[Cell, list[tuple[date, datetime, float]]] = defaultdict(list)
        for request in dataset.requests:
            cell = Cell(request.region, request.resource)
            day = utc_day(request.arrived_at)
            self._arrival[cell][day] += request.requested_hours
            self._arrival_count[cell][day] += 1
            self._arrival_knowledge[cell].append((day, request.available_at, request.requested_hours))
        for entry in self.evidence.values():
            cell = Cell(entry.request.region, entry.request.resource)
            if entry.disposition == "abandoned":
                self._abandoned[cell][utc_day(entry.request.arrived_at)] += entry.request.requested_hours
            if entry.disposition != DISP_SERVED or entry.terminal_closed_at is None:
                continue
            self._service[cell][utc_day(entry.terminal_closed_at)] += entry.observed_served_hours or 0.0
            # Settled service becomes known when the terminal row itself is available.
            terminal = next(item for item in dataset.tasks if item.task_id == entry.terminal_task_id)
            self._service_knowledge[cell].append(
                (utc_day(entry.terminal_closed_at), terminal.available_at, entry.observed_served_hours or 0.0)
            )

    # ------------------------- series lookups -------------------------
    def keys(self) -> list[SeriesKey]:
        keys = [SeriesKey(LEVEL_CELL, cell.region, cell.resource) for cell in self.cells]
        keys += [SeriesKey(LEVEL_REGION, region) for region in self.regions]
        keys += [SeriesKey(LEVEL_RESOURCE, resource=resource) for resource in self.resources]
        keys.append(SeriesKey(LEVEL_TOTAL))
        return keys

    def _sum_cells(
        self,
        mapping: dict[Cell, dict[date, float]],
        day: date,
        key: SeriesKey,
        knowledge: dict[Cell, list[tuple[date, datetime, float]]] | None = None,
        origin: datetime | None = None,
    ) -> float:
        if knowledge is not None and origin is not None:
            return sum(
                value
                for cell in self.cells
                if key.matches(cell.region, cell.resource)
                for observed_day, available_at, value in knowledge[cell]
                if observed_day == day and available_at <= origin
            )
        return sum(
            mapping[cell].get(day, 0.0) for cell in self.cells if key.matches(cell.region, cell.resource)
        )

    def arrival_on(self, key: SeriesKey, day: date, origin: datetime | None = None) -> float:
        return self._sum_cells(self._arrival, day, key, self._arrival_knowledge, origin)

    def services_on(self, key: SeriesKey, day: date, origin: datetime | None = None) -> float:
        return self._sum_cells(self._service, day, key, self._service_knowledge, origin)

    def abandoned_on(self, key: SeriesKey, day: date) -> float:
        return self._sum_cells(self._abandoned, day, key)

    def first_day(self, key: SeriesKey, origin: datetime | None = None) -> date | None:
        if origin is not None:
            days = [
                observed_day
                for cell in self.cells
                if key.matches(cell.region, cell.resource)
                for observed_day, available_at, _ in self._arrival_knowledge[cell]
                if available_at <= origin
            ]
        else:
            days = [
                observed_day
                for cell in self.cells
                if key.matches(cell.region, cell.resource)
                for observed_day, _, _ in self._arrival_knowledge[cell]
            ]
        return min(days) if days else None


def capacity_for_day(dataset: PlanningDataset, cell_key: SeriesKey, day: date) -> list[Any]:
    """Capacity windows covering a day, for cells only."""
    if cell_key.level != LEVEL_CELL:
        return []
    day_begin = day_start(day)
    day_end = day_begin + timedelta(days=1)
    return [
        window
        for window in dataset.capacity
        if window.region == cell_key.region
        and window.resource == cell_key.resource
        and window.start < day_end
        and window.end > day_begin
    ]


def nominal_hours_per_day(dataset: PlanningDataset, key: SeriesKey, day: date) -> float | None:
    """Sum of nominal capacity hours for a cell on a day; None if unknown."""
    if key.level != LEVEL_CELL:
        raise ValueError("nominal capacity is defined per cell")
    windows = capacity_for_day(dataset, key, day)
    if not windows:
        return None
    day_begin = day_start(day)
    day_end = day_begin + timedelta(days=1)
    total = 0.0
    for window in windows:
        overlap_start = max(day_begin, window.start)
        overlap_end = min(day_end, window.end)
        covered = max(0.0, (overlap_end - overlap_start).total_seconds() / 86400.0)
        total += covered * window.hours_per_day
    return total


def longest_zero_run(values: list[float]) -> tuple[int, int | None, int | None]:
    best = 0
    best_start = None
    best_end = None
    run = 0
    start = None
    for index, value in enumerate(values):
        if value == 0:
            if run == 0:
                start = index
            run += 1
            if run > best:
                best = run
                best_start = start
                best_end = index
        else:
            run = 0
    return best, best_start, best_end
