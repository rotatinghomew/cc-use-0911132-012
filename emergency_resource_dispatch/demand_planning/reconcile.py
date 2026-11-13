"""Reconcile raw requests and task rows into measurable demand.

Counting rules used everywhere downstream:

* One request is one demand unit, regardless of how many task rows it produced.
  A replacement row (``replaces``) is a re-dispatch of the same request, not a
  second request.
* Each request resolves to exactly one *terminal* task row: the latest live
  row.  A row named by another row's ``replaces`` is superseded and excluded
  from the choice of terminal row, so a re-dispatch chain collapses onto the
  final live row.
* ``requested_hours`` on the terminal row is the declared demand reached while
  the resource could be committed.  ``served_hours`` is what was delivered and
  is bounded by capacity, not demand.
* A terminal row closed as ``abandoned`` is a queue withdrawal.  It remains
  demand and must never be counted as zero.
* A terminal row that is itself a ``reassigned`` re-dispatch means the request
  left this ledger; its final service realisation is *unknown*, not zero.
  Earlier service observed on the superseded row is kept as pre-replacement
  evidence but is not added to final realised service.
* A request whose terminal row is still open (``closed_at`` null) is censored.
  Its declared hours are counted as arrival demand, but its served hours are
  withheld from training because settlement is not final and using them would
  leak a not-yet-closed outcome into the training era.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from emergency_resource_dispatch.demand_planning.inputs import PlanningDataset, Request, Task

DISP_SERVED = "served"
DISP_ABANDONED = "abandoned"
DISP_REASSIGNED = "reassigned"
DISP_CENSORED = "censored"
DISP_UNLINKED = "unlinked"


@dataclass
class RequestEvidence:
    """Per-request accounting with explicit task-row pointers."""

    request: Request
    task_rows: list[str] = field(default_factory=list)
    superseded_rows: list[str] = field(default_factory=list)
    terminal_task_id: str | None = None
    terminal_outcome: str = "unlinked"
    terminal_closed_at: datetime | None = None
    declared_hours: float | None = None
    observed_served_hours: float | None = None
    pre_replacement_observed_hours: float = 0.0

    @property
    def disposition(self) -> str:
        if self.terminal_task_id is None:
            return DISP_UNLINKED
        if self.terminal_closed_at is None:
            return DISP_CENSORED
        if self.terminal_outcome == "abandoned":
            return DISP_ABANDONED
        if self.terminal_outcome == "reassigned":
            return DISP_REASSIGNED
        return DISP_SERVED

    @property
    def observed_hours(self) -> float | None:
        """Hours usable as a settled service observation for training."""
        if self.disposition == DISP_SERVED:
            return self.observed_served_hours
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "region": self.request.region,
            "resource": self.request.resource,
            "arrived_at": self.request.arrived_at.isoformat(),
            "requested_hours": self.request.requested_hours,
            "disposition": self.disposition,
            "task_rows": self.task_rows,
            "superseded_rows": self.superseded_rows,
            "terminal_task_id": self.terminal_task_id,
            "terminal_outcome": self.terminal_outcome,
            "terminal_closed_at": self.terminal_closed_at.isoformat() if self.terminal_closed_at else None,
            "declared_hours": self.declared_hours,
            "observed_served_hours": self.observed_hours,
            "pre_replacement_observed_hours": self.pre_replacement_observed_hours or None,
        }


def _index(tasks: list[Task]) -> dict[str, Task]:
    return {item.task_id: item for item in tasks}


def resolve_requests(dataset: PlanningDataset) -> dict[str, RequestEvidence]:
    """Attach every task row to its request and pick the terminal row."""
    by_request: dict[str, list[Task]] = defaultdict(list)
    for task in dataset.tasks:
        by_request[task.request_id].append(task)

    indexed = _index(dataset.tasks)
    superseded: set[str] = set()
    pending = [task.replaces for task in dataset.tasks if task.replaces]
    while pending:
        target = pending.pop()
        if target in superseded or target not in indexed:
            continue
        superseded.add(target)
        parent = indexed[target]
        if parent.replaces:
            pending.append(parent.replaces)

    evidence: dict[str, RequestEvidence] = {}
    for request in dataset.requests:
        rows = sorted(by_request.get(request.request_id, []), key=lambda item: (item.available_at, item.task_id))
        entry = RequestEvidence(
            request=request,
            task_rows=[item.task_id for item in rows],
            superseded_rows=[item.task_id for item in rows if item.task_id in superseded],
        )
        live = [item for item in rows if item.task_id not in superseded]
        if not live:
            evidence[request.request_id] = entry
            continue
        terminal = max(
            live,
            key=lambda item: (
                item.available_at,
                item.closed_at or datetime.min.replace(tzinfo=timezone.utc),
                item.task_id,
            ),
        )
        entry.terminal_task_id = terminal.task_id
        entry.terminal_outcome = terminal.outcome
        entry.terminal_closed_at = terminal.closed_at
        entry.declared_hours = terminal.requested_hours
        entry.observed_served_hours = terminal.served_hours
        # Service observed before a re-dispatch is evidence, but the final
        # realisation is unknown, so keep it separate from settled service.
        if entry.disposition == DISP_REASSIGNED:
            entry.pre_replacement_observed_hours = sum(
                item.served_hours or 0.0 for item in rows if item.task_id in superseded
            )
        evidence[request.request_id] = entry
    return evidence


def build_reconciliation(dataset: PlanningDataset, evidence: dict[str, RequestEvidence]) -> dict[str, Any]:
    """Request-level reconciliation with observable and modelled components."""
    requests = dataset.requests
    task_rows = len(dataset.tasks)
    raw_request_hours = sum(item.requested_hours for item in requests)
    raw_task_hours = sum(item.requested_hours or 0.0 for item in dataset.tasks)

    superseded_hours = 0.0
    superseded_rows = 0
    for entry in evidence.values():
        for task_id in entry.superseded_rows:
            row = next((item for item in dataset.tasks if item.task_id == task_id), None)
            if row is not None:
                superseded_hours += row.requested_hours or 0.0
                superseded_rows += 1

    counts = defaultdict(int)
    hours_by_disposition: dict[str, float] = defaultdict(float)
    served_declared = 0.0
    observed_served = 0.0
    service_shortfall = 0.0
    service_excess = 0.0
    pre_replacement_hours = 0.0
    unlinked_requests: list[str] = []
    unclosed_terminal_task_ids: list[str] = []
    open_served_hours_withheld = 0.0

    for entry in evidence.values():
        disposition = entry.disposition
        counts[disposition] += 1
        hours_by_disposition[disposition] += entry.request.requested_hours
        if disposition == DISP_UNLINKED:
            unlinked_requests.append(entry.request.request_id)
        if disposition == DISP_CENSORED:
            unclosed_terminal_task_ids.append(entry.terminal_task_id or "")
            open_served_hours_withheld += entry.observed_served_hours or 0.0
        if disposition == DISP_SERVED:
            declared = entry.declared_hours if entry.declared_hours is not None else entry.request.requested_hours
            observed = entry.observed_served_hours or 0.0
            served_declared += declared
            observed_served += observed
            if observed < declared:
                service_shortfall += declared - observed
            else:
                service_excess += observed - declared
        if disposition == DISP_REASSIGNED:
            pre_replacement_hours += entry.pre_replacement_observed_hours

    accounted = sum(hours_by_disposition.values())
    total_requests = len(requests)

    # Bounds on unobservable withdrawal demand.  The observed ledger already
    # counts every recorded abandonment as demand.  Any request that never
    # reached a task row, or that left as a re-dispatch, could additionally
    # have withdrawn unseen; that amount is not recoverable from completion
    # logs, so it is bounded from explicit, stated scenarios rather than
    # asserted.
    recorded = raw_request_hours
    unseen_low = 0.0
    unseen_high = round(hours_by_disposition[DISP_ABANDONED] * 0.25, 6)
    acknowledged_ceiling = round(max(recorded, recorded + unseen_high), 6)

    return {
        "unit": "requested_hours",
        "counting_rule": (
            "one request = one demand unit; a replaces row is a re-dispatch of the same request and is not counted again"
        ),
        "period": {
            "request_arrival_start": min((item.arrived_at for item in requests), default=None).isoformat()
            if requests
            else None,
            "request_arrival_end": max((item.arrived_at for item in requests), default=None).isoformat()
            if requests
            else None,
            "training_cutoff": dataset.final_training_cutoff.isoformat(),
            "observable_service_end": dataset.as_of.isoformat(),
            "calendar_end": max((item.day for item in dataset.calendar), default=None).isoformat()
            if dataset.calendar
            else None,
        },
        "raw_file_totals": {
            "request_rows": total_requests,
            "task_rows": task_rows,
            "requested_hours_on_request_rows": round(raw_request_hours, 6),
            "requested_hours_on_task_rows": round(raw_task_hours, 6),
            "task_rows_minus_request_hours": round(raw_task_hours - raw_request_hours, 6),
            "superseded_task_rows": superseded_rows,
            "superseded_requested_hours": round(superseded_hours, 6),
        },
        "dispositions": {
            "counts": dict(counts),
            "requested_hours": {key: round(value, 6) for key, value in hours_by_disposition.items()},
            "served_declared_hours": round(served_declared, 6),
            "observed_served_hours": round(observed_served, 6),
            "abandoned_hours_kept_as_demand": round(hours_by_disposition[DISP_ABANDONED], 6),
            "reassigned_hours_kept_as_demand": round(hours_by_disposition[DISP_REASSIGNED], 6),
            "censored_hours_kept_as_arrival_demand": round(hours_by_disposition[DISP_CENSORED], 6),
            "unlinked_hours": round(hours_by_disposition[DISP_UNLINKED], 6),
        },
        "balance": {
            "requested_hours": round(raw_request_hours, 6),
            "accounted_hours": round(accounted, 6),
            "unaccounted_hours": round(raw_request_hours - accounted, 6),
            "accounted_equals_requested": abs(accounted - raw_request_hours) < 1e-9,
            "composition": "served_declared + abandoned + reassigned + censored + unlinked",
        },
        "measurable_demand": {
            "definition": (
                "Demand that the ledger can measure: requested hours on every request row, each request counted once, "
                "split by whether the request was served, withdrew from the queue, left as a re-dispatch, or is still open."
            ),
            "total_hours": round(raw_request_hours, 6),
            "served_request_hours": round(hours_by_disposition[DISP_SERVED], 6),
            "withdrawn_request_hours": round(hours_by_disposition[DISP_ABANDONED], 6),
            "reassigned_request_hours": round(hours_by_disposition[DISP_REASSIGNED], 6),
            "open_request_hours": round(hours_by_disposition[DISP_CENSORED], 6),
            "unlinked_request_hours": round(hours_by_disposition[DISP_UNLINKED], 6),
            "observed_service_hours": round(observed_served, 6),
            "gap_between_measured_demand_and_observed_service": round(raw_request_hours - observed_served, 6),
            "state": "measured",
        },
        "not_directly_observable": {
            "definition": "Withdrawals that never produced a request or task row cannot be read out of the ledger.",
            "claim": "no true-population estimate is made from completion logs",
            "bounds": {
                "lower_hours": round(recorded, 6),
                "upper_hours": acknowledged_ceiling,
                "upper_is_assumption": True,
            },
            "recoverable_from_completion_logs": False,
        },
        "double_count_prevented": {
            "duplicate_task_rows_not_counted": superseded_rows,
            "duplicate_hours_not_counted": round(superseded_hours, 6),
            "explanation": "the replaces row repeats the same request's hours; only the terminal row counts",
        },
        "unknown_not_treated_as_zero": {
            "abandoned_hours_kept_as_demand": round(hours_by_disposition[DISP_ABANDONED], 6),
            "censored_hours_kept_as_arrival_demand": round(hours_by_disposition[DISP_CENSORED], 6),
            "reassigned_hours_kept_as_demand": round(hours_by_disposition[DISP_REASSIGNED], 6),
            "pre_replacement_observed_hours_reported_separately": round(pre_replacement_hours, 6),
            "open_served_hours_withheld_from_training": round(open_served_hours_withheld, 6),
        },
        "withdrawal_bounds": {
            "explanation": (
                "recorded abandonment is already inside requested hours; the ledger cannot reveal withdrawals "
                "that never produced a request or task row. The upper bound adds a stated scenario where 25% of "
                "recorded abandonment hours were also lost unseen; it is an assumption, not a recovery of truth."
            ),
            "recorded_demand_hours": round(recorded, 6),
            "unseen_withdrawal_lower_hours": unseen_low,
            "unseen_withdrawal_upper_hours": unseen_high,
            "demand_lower_bound_hours": round(recorded, 6),
            "demand_upper_bound_hours": acknowledged_ceiling,
            "upper_bound_is_assumption": True,
        },
        "service_is_capacity_bounded": {
            "declared_service_hours": round(served_declared, 6),
            "observed_service_hours": round(observed_served, 6),
            "service_shortfall_hours": round(service_shortfall, 6),
            "observed_service_excess_hours": round(service_excess, 6),
            "completion_log_to_request_ratio": round(observed_served / raw_request_hours, 6) if raw_request_hours else None,
            "note": "declared and observed service are both constrained by capacity; neither is the demand level",
        },
        "unlinked_request_ids": unlinked_requests,
        "unclosed_terminal_task_ids": unclosed_terminal_task_ids,
        "requests": [entry.as_dict() for entry in evidence.values()],
    }
