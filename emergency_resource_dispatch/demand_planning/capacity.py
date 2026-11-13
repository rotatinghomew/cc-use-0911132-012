"""Explain whether a drop in completed work tracks demand or capacity.

The diagnostic compares consecutive capacity regimes per cell.  It reports
arrival demand, declared service and observed service side by side so that a
capacity cut is not silently read as a demand fall.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from emergency_resource_dispatch.demand_planning.inputs import PlanningDataset
from emergency_resource_dispatch.demand_planning.series import (
    LEVEL_CELL,
    Cell,
    DemandSeries,
    SeriesKey,
    utc_day,
)


def capacity_regimes(dataset: PlanningDataset, series: DemandSeries) -> list[dict[str, Any]]:
    """Detect step changes in nominal capacity for each cell with history."""
    out: list[dict[str, Any]] = []
    for cell in series.cells:
        windows = sorted(
            [item for item in dataset.capacity if item.region == cell.region and item.resource == cell.resource],
            key=lambda item: item.start,
        )
        for left, right in zip(windows, windows[1:]):
            if right.hours_per_day >= left.hours_per_day:
                continue
            out.append(
                _regime_comparison(dataset, series, cell, left, right)
            )
    return out


def _regime_comparison(
    dataset: PlanningDataset,
    series: DemandSeries,
    cell: Cell,
    before: Any,
    after: Any,
) -> dict[str, Any]:
    step_day = utc_day(after.start)
    window_days = min(14, max(7, (step_day - utc_day(before.start)).days // 2))
    pre_start = step_day - timedelta(days=window_days)
    post_end = step_day + timedelta(days=window_days)
    key = SeriesKey(LEVEL_CELL, cell.region, cell.resource)

    def totals(start: date, end: date) -> dict[str, float]:
        arrival = served = abandoned = 0.0
        day = start
        while day < end:
            arrival += series.arrival_on(key, day)
            served += series.services_on(key, day)
            abandoned += series.abandoned_on(key, day)
            day += timedelta(days=1)
        return {"arrival": arrival, "served": served, "abandoned": abandoned, "days": float((end - start).days)}

    pre = totals(pre_start, step_day)
    post = totals(step_day, post_end)

    def change(metric: str) -> float | None:
        if pre[metric] <= 0:
            return None
        return round((post[metric] - pre[metric]) / pre[metric], 6)

    capacity_change = round(after.hours_per_day / before.hours_per_day - 1.0, 6) if before.hours_per_day else None
    arrival_change = change("arrival")
    served_change = change("served")
    abandoned_change = change("abandoned")

    utilisation_pre = pre["served"] / (before.hours_per_day * pre["days"]) if before.hours_per_day else None
    utilisation_post = post["served"] / (after.hours_per_day * post["days"]) if after.hours_per_day else None

    findings: list[str] = []
    if arrival_change is not None and served_change is not None:
        if served_change <= -0.10 and arrival_change > -0.05:
            findings.append(
                "completed work fell while arrivals did not, so the drop tracks capacity rather than demand"
            )
        elif served_change <= -0.10 and arrival_change <= -0.10:
            findings.append("both completed work and arrivals fell; the two effects are entangled")
        elif served_change > -0.10 and arrival_change <= -0.10:
            findings.append("completed work held up while arrivals fell, so service is not a demand gauge")
    if utilisation_pre is not None and utilisation_post is not None and utilisation_post > 0.95 and after.hours_per_day > 0:
        findings.append("observed service is at the nominal capacity ceiling after the step, so served hours are censored by capacity")

    return {
        "cell": cell.id,
        "region": cell.region,
        "resource": cell.resource,
        "step_at": after.start.isoformat(),
        "window_days": window_days,
        "capacity": {
            "before_hours_per_day": before.hours_per_day,
            "after_hours_per_day": after.hours_per_day,
            "relative_change": capacity_change,
        },
        "pre_step": {
            "period": f"{pre_start.isoformat()}..{step_day.isoformat()}",
            "arrival_hours_per_day": round(pre["arrival"] / pre["days"], 6) if pre["days"] else None,
            "declared_service_hours_per_day": round(pre["served"] / pre["days"], 6) if pre["days"] else None,
            "abandoned_request_hours_per_day": round(pre["abandoned"] / pre["days"], 6) if pre["days"] else None,
            "observed_service_to_arrival": round(pre["served"] / pre["arrival"], 6) if pre["arrival"] else None,
            "nominal_utilisation": round(utilisation_pre, 6) if utilisation_pre is not None else None,
        },
        "post_step": {
            "period": f"{step_day.isoformat()}..{post_end.isoformat()}",
            "arrival_hours_per_day": round(post["arrival"] / post["days"], 6) if post["days"] else None,
            "declared_service_hours_per_day": round(post["served"] / post["days"], 6) if post["days"] else None,
            "abandoned_request_hours_per_day": round(post["abandoned"] / post["days"], 6) if post["days"] else None,
            "observed_service_to_arrival": round(post["served"] / post["arrival"], 6) if post["arrival"] else None,
            "nominal_utilisation": round(utilisation_post, 6) if utilisation_post is not None else None,
        },
        "relative_change": {
            "arrival_demand": arrival_change,
            "observed_service": served_change,
            "abandoned_request_demand": abandoned_change,
        },
        "findings": findings,
        "reading": (
            "arrival demand is the planning target; observed service and abandoned demand are capacity-constrained "
            "and must not be substituted for it"
        ),
    }


def zero_demand_windows(
    dataset: PlanningDataset, series: DemandSeries, cutoff: date, origin: datetime | None = None
) -> dict[str, Any]:
    """Report cells whose history ends in a zero-demand run.

    A zero run shared by every cell is treated as a likely reporting gap, not a
    true demand collapse: the run is reported but not described as demand.
    """
    per_cell = []
    for key in series.keys():
        if key.level != LEVEL_CELL:
            continue
        start = series.first_day(key)
        if start is None:
            continue
        days = [start + timedelta(days=i) for i in range((cutoff - start).days)]
        values = [series.arrival_on(key, day, origin) for day in days]
        trailing = 0
        for value in reversed(values):
            if value == 0:
                trailing += 1
            else:
                break
        longest = 0
        run = 0
        for value in values:
            run = run + 1 if value == 0 else 0
            longest = max(longest, run)
        per_cell.append(
            {
                "key": key.id,
                "trailing_zero_days": trailing,
                "longest_zero_run_days": longest,
                "training_days": len(days),
                "training_end": days[-1].isoformat() if days else None,
            }
        )
    last_common = None
    if per_cell:
        last_common = min(item["trailing_zero_days"] for item in per_cell)
    ledger_tail_zero = bool(per_cell) and last_common is not None and last_common >= 3
    return {
        "per_cell": per_cell,
        "shared_trailing_zero_days": last_common,
        "likely_reporting_gap": ledger_tail_zero,
        "interpretation": (
            "every cell's ledger goes quiet for the same trailing days, which points to a reporting cutoff "
            "rather than simultaneous demand collapse; those days are excluded from training, not counted as zero demand"
        )
        if ledger_tail_zero
        else "zero runs are cell-specific and are handled as low-demand regimes",
        "effect_on_forecasts": (
            "rates are estimated on the last non-zero window and shrunk toward the parent level; "
            "intervals widen through the dispersion term instead of the last non-zero day being carried forward"
        ),
    }


def coverage_gap(dataset: PlanningDataset) -> dict[str, Any]:
    """Compare the ledger's last observed fact with the analysis as_of clock.

    as_of is the information cutoff, not an upper bound on business time.  When
    the ledger simply stops several days before as_of, those days are missing
    coverage, not zero demand; the plan says so instead of training on them.
    """
    last_arrival = max((item.arrived_at for item in dataset.requests), default=None)
    last_available = max((item.available_at for item in dataset.requests), default=None)
    last_task = max((item.closed_at for item in dataset.tasks if item.closed_at), default=None)
    gap = (dataset.as_of.date() - last_arrival.date()).days if last_arrival else None
    if last_arrival is None:
        interpretation = "the ledger contains no request rows, so nothing can be said about coverage or demand"
        silence = False
    elif gap is not None and gap >= 7:
        interpretation = (
            "the ledger stops before as_of for every cell; the trailing silent days are treated as missing coverage, "
            "not as zero demand, and they are not used as training zeros"
        )
        silence = True
    else:
        interpretation = "ledger coverage reaches as_of"
        silence = False
    return {
        "as_of": dataset.as_of.isoformat(),
        "last_request_arrival": last_arrival.isoformat() if last_arrival else None,
        "last_request_available_at": last_available.isoformat() if last_available else None,
        "last_task_closure": last_task.isoformat() if last_task else None,
        "gap_days": gap,
        "trailing_ledger_silence": silence,
        "interpretation": interpretation,
        "handling": "training ends at final_training_cutoff; silently missing days beyond it never enter a fit",
    }
