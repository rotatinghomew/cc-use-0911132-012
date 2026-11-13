"""Forecast candidates, rolling-origin backtests and support decisions.

Target metric: **arrival demand in requested hours**, booked on the day a
request arrived.  Observed ``served_hours`` is never the forecast target: it is
bounded by capacity and partly censored, so it cannot measure demand.

Candidates, all fit only on data whose ``available_at`` precedes the origin and
whose arrival day is strictly before the training cutoff:

* ``trailing_mean``      - 14-day mean daily rate (transparent baseline).
* ``hierarchical_shrinkage`` - 28-day local rate shrunk toward a parent rate
  (sibling cells sharing the resource, or the level above) with a fixed prior
  weight of 14 days.  This is the borrowing mechanism: a thin or newly quiet
  cell is pulled toward its siblings instead of trusting a handful of points.
* ``seasonal_naive``     - repeat the detected cycle when a period is reliably
  detectable in the *trailing* data only; unavailable otherwise.

Intervals are approximate negative-binomial predictive intervals obtained from a
gamma moment match (mean H, variance V), widened by parameter uncertainty
(``h**2 * var / n_eff``).  Coverage is checked, not assumed.

Model selection is per key, by mean weighted absolute percentage error (WAPE)
over complete rolling-origin folds.  A simpler candidate wins ties within 10%
relative WAPE, which prevents selecting a fragile seasonal pattern on noise.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from emergency_resource_dispatch.demand_planning.series import (
    LEVEL_CELL,
    LEVEL_REGION,
    LEVEL_RESOURCE,
    LEVEL_TOTAL,
    DemandSeries,
    SeriesKey,
    day_start,
    longest_zero_run,
    utc_day,
)

METHOD_VERSION = "demand-planning/1"
BASELINE_WINDOW_DAYS = 14
SHRINKAGE_WINDOW_DAYS = 28
PRIOR_WEIGHT_DAYS = 14.0
MAX_SEASONAL_PERIOD = 10
SEASONAL_RELATIVE_ERROR = 0.30
MIN_TRAINING_DAYS = 21
MIN_NONZERO_DAYS = 10
MIN_FOLDS = 4
FOLD_STEP_DAYS = 7
SUPPORT_WAPE_LIMIT = 0.50
LIMITED_WAPE_LIMIT = 0.80
SUPPORT_COVERAGE_LIMIT = 0.50
ZERO_WINDOW_LIMIT = 21
PREFERENCE_MARGIN = 0.10
PARSIMONY = ["trailing_mean", "hierarchical_shrinkage", "seasonal_naive"]

NORMAL = statistics.NormalDist()


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def gamma_quantile(probability: float, shape: float, scale: float) -> float:
    """Wilson-Hilferty approximation of a gamma quantile."""
    if shape <= 0 or scale <= 0:
        return 0.0
    z = NORMAL.inv_cdf(probability)
    k = shape
    return max(0.0, k * (1.0 - 1.0 / (9.0 * k) + z * math.sqrt(1.0 / (9.0 * k))) ** 3 * scale)


def predictive_interval(point: float, variance: float, effective_n: float) -> dict[str, Any]:
    horizon_variance = variance + variance / max(effective_n, 1.0)
    if point <= 0 or horizon_variance <= 0:
        return {
            "interval_80": {"lower": 0.0, "upper": 0.0},
            "interval_95": {"lower": 0.0, "upper": 0.0},
            "shape": None,
            "scale": None,
        }
    shape = point * point / horizon_variance
    scale = horizon_variance / point
    return {
        "interval_80": {
            "lower": round(gamma_quantile(0.10, shape, scale), 6),
            "upper": round(gamma_quantile(0.90, shape, scale), 6),
        },
        "interval_95": {
            "lower": round(gamma_quantile(0.025, shape, scale), 6),
            "upper": round(gamma_quantile(0.975, shape, scale), 6),
        },
        "shape": round(shape, 6),
        "scale": round(scale, 6),
    }


# --------------------------------------------------------------------------
# candidate fits
# --------------------------------------------------------------------------
@dataclass
class CandidateFit:
    name: str
    available: bool
    daily_rate: float = 0.0
    daily_variance: float = 0.0
    effective_n: float = 1.0
    detail: dict[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None


def _values(series: DemandSeries, key: SeriesKey, days: list[date], origin: datetime) -> list[float]:
    """Arrival demand known at the origin, booked on each day."""
    return [series.arrival_on(key, day, origin) for day in days]


def _window(start_day: date, origin_day: date, width: int) -> list[date]:
    end = origin_day
    begin = max(start_day, origin_day - timedelta(days=width))
    return [begin + timedelta(days=i) for i in range((end - begin).days)]


def _parent_rate(series: DemandSeries, key: SeriesKey, days: list[date], origin: datetime) -> tuple[float | None, str]:
    """Rate borrowed from a sibling pool, computed from the same days only.

    Only siblings with actual observations in this window contribute.  A cell
    that appears in planning hints but has never recorded a request is not a
    sibling with a known zero rate; it is simply absent from the pool.
    """
    if not days:
        return None, "no_window"
    if key.level == LEVEL_TOTAL:
        return None, "top_level"
    if key.level == LEVEL_REGION:
        observed = [
            region
            for region in series.regions
            if region != key.region
            and any(series.arrival_on(SeriesKey(LEVEL_REGION, region), day, origin) for day in days)
        ]
        if not observed:
            return None, "no_observed_sibling_regions"
        total = sum(series.arrival_on(SeriesKey(LEVEL_REGION, region), day, origin) for region in observed for day in days)
        return total / (len(days) * len(observed)), "observed_sibling_regions"
    if key.level == LEVEL_RESOURCE:
        observed = [
            resource
            for resource in series.resources
            if resource != key.resource
            and any(series.arrival_on(SeriesKey(LEVEL_RESOURCE, resource=resource), day, origin) for day in days)
        ]
        if not observed:
            return None, "no_observed_sibling_resources"
        total = sum(
            series.arrival_on(SeriesKey(LEVEL_RESOURCE, resource=resource), day, origin)
            for resource in observed
            for day in days
        )
        return total / (len(days) * len(observed)), "observed_sibling_resources"
    # cell: average daily rate among sibling cells sharing the resource and
    # an active request history in this window.
    observed = [
        cell
        for cell in series.cells
        if cell.resource == key.resource
        and cell.region != key.region
        and any(series.arrival_on(SeriesKey(LEVEL_CELL, cell.region, cell.resource), day, origin) for day in days)
    ]
    if not observed:
        return None, "no_observed_sibling_cells"
    sibling_total = sum(
        series.arrival_on(SeriesKey(LEVEL_CELL, cell.region, cell.resource), day, origin)
        for cell in observed
        for day in days
    )
    return sibling_total / (len(days) * len(observed)), "observed_sibling_cells"


def _dispersion(values: list[float], rate: float) -> float:
    """Daily variance floored at the mean: treat hours as overdispersed counts."""
    return max(_variance(values), rate, 1e-9)


def fit_trailing_mean(series: DemandSeries, key: SeriesKey, start_day: date, origin_day: date, origin: datetime | None = None) -> CandidateFit:
    days = _window(start_day, origin_day, BASELINE_WINDOW_DAYS)
    origin_ts = origin or day_start(origin_day)
    values = _values(series, key, days, origin_ts)
    if not values:
        return CandidateFit("trailing_mean", False, unavailable_reason="no_training_days")
    rate = _mean(values)
    return CandidateFit(
        "trailing_mean",
        True,
        daily_rate=rate,
        daily_variance=_dispersion(values, rate),
        effective_n=float(len(values)),
        detail={"window_days": len(values), "window_start": days[0].isoformat(), "window_end": days[-1].isoformat()},
    )


def fit_hierarchical(series: DemandSeries, key: SeriesKey, start_day: date, origin_day: date, origin: datetime | None = None) -> CandidateFit:
    days = _window(start_day, origin_day, SHRINKAGE_WINDOW_DAYS)
    origin_ts = origin or day_start(origin_day)
    values = _values(series, key, days, origin_ts)
    if not values:
        return CandidateFit("hierarchical_shrinkage", False, unavailable_reason="no_training_days")
    local_rate = _mean(values)
    parent, parent_source = _parent_rate(series, key, days, origin_ts)
    if parent is None:
        rate = local_rate
        borrowed = 0.0
    else:
        borrowed = PRIOR_WEIGHT_DAYS
        rate = (sum(values) + borrowed * parent) / (len(values) + borrowed)
    return CandidateFit(
        "hierarchical_shrinkage",
        True,
        daily_rate=rate,
        daily_variance=_dispersion(values, rate),
        effective_n=float(len(values)) + borrowed,
        detail={
            "window_days": len(values),
            "window_start": days[0].isoformat(),
            "window_end": days[-1].isoformat(),
            "local_rate": round(local_rate, 6),
            "parent_rate": None if parent is None else round(parent, 6),
            "parent_source": parent_source,
            "prior_weight_days": borrowed,
        },
    )


def detect_period(values: list[float]) -> tuple[int, float] | None:
    """Return (period, relative error) if a cycle is reliable in this window."""
    best: tuple[int, float] | None = None
    for period in range(2, MAX_SEASONAL_PERIOD + 1):
        if len(values) < period + 8:
            continue
        errors = [abs(values[i] - values[i - period]) for i in range(period, len(values))]
        mae = sum(errors) / len(errors)
        level = max(1e-9, _mean(values[-period:]))
        relative = mae / level
        if best is None or relative < best[1]:
            best = (period, relative)
    return best


def fit_seasonal(series: DemandSeries, key: SeriesKey, start_day: date, origin_day: date, origin: datetime | None = None) -> CandidateFit:
    days = _window(start_day, origin_day, SHRINKAGE_WINDOW_DAYS)
    origin_ts = origin or day_start(origin_day)
    values = _values(series, key, days, origin_ts)
    if len(values) < 12:
        return CandidateFit("seasonal_naive", False, unavailable_reason="not_enough_days")
    if all(value == 0 for value in values[-3:]):
        return CandidateFit("seasonal_naive", False, unavailable_reason="trailing_zero_window")
    detected = detect_period(values)
    if detected is None or detected[1] > SEASONAL_RELATIVE_ERROR:
        return CandidateFit("seasonal_naive", False, unavailable_reason="no_reliable_period")
    period, relative = detected
    cycle = values[-period:]
    rate = _mean(cycle)
    residual = _variance(values[-SHRINKAGE_WINDOW_DAYS:]) if len(values) > 2 else 0.0
    return CandidateFit(
        "seasonal_naive",
        True,
        daily_rate=rate,
        daily_variance=max(residual, rate, 1e-9),
        effective_n=max(2.0, len(values) / period),
        detail={
            "period_days": period,
            "relative_error": round(relative, 6),
            "cycle_values": [round(value, 6) for value in cycle],
            "window_days": len(values),
        },
    )


CANDIDATE_FITTERS = {
    "trailing_mean": fit_trailing_mean,
    "hierarchical_shrinkage": fit_hierarchical,
    "seasonal_naive": fit_seasonal,
}


# --------------------------------------------------------------------------
# rolling-origin backtests
# --------------------------------------------------------------------------
def backtest_origins(start_day: date, cutoff_day: date, horizon: int) -> list[date]:
    """Origins every 7 days whose target window closes by the training cutoff."""
    origins: list[date] = []
    origin = start_day + timedelta(days=SHRINKAGE_WINDOW_DAYS)
    while origin + timedelta(days=horizon) <= cutoff_day:
        origins.append(origin)
        origin += timedelta(days=FOLD_STEP_DAYS)
    return origins


def _score(point: float, interval: dict[str, Any], actual: float) -> dict[str, Any]:
    error = point - actual
    return {
        "point_hours": round(point, 6),
        "actual_hours": round(actual, 6),
        "error_hours": round(error, 6),
        "absolute_error": round(abs(error), 6),
        "absolute_percentage_error": round(abs(error) / actual, 6) if actual > 0 else None,
        "covered_80": interval["interval_80"]["lower"] <= actual <= interval["interval_80"]["upper"],
        "covered_95": interval["interval_95"]["lower"] <= actual <= interval["interval_95"]["upper"],
        "lower_80": interval["interval_80"]["lower"],
        "upper_80": interval["interval_80"]["upper"],
    }


def _fold_key(
    series: DemandSeries,
    key: SeriesKey,
    start_day: date,
    origin: date,
    origin_dt: datetime,
    horizon: int,
    per_key_models: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    target_days = [origin + timedelta(days=i) for i in range(horizon)]
    # Realised outcome for scoring: all rows are observable after the fact.
    actual = sum(series.arrival_on(key, day) for day in target_days)
    models: dict[str, Any] = {}
    for name, fitter in CANDIDATE_FITTERS.items():
        fit = fitter(series, key, start_day, origin, origin_dt)
        if not fit.available:
            models[name] = {"available": False, "reason": fit.unavailable_reason}
            continue
        point = horizon * fit.daily_rate
        interval = predictive_interval(point, horizon * fit.daily_variance, fit.effective_n)
        scored = _score(point, interval, actual)
        scored["available"] = True
        models[name] = scored
        if per_key_models is not None:
            per_key_models[key.id][name].append(scored)
    return {"key": key.id, "actual_hours": round(actual, 6), "models": models}


def run_backtests(
    series: DemandSeries,
    keys: list[SeriesKey],
    start_day: date,
    cutoff_day: date,
    horizon: int,
    holdout_end: date | None = None,
) -> dict[str, Any]:
    """Rolling-origin folds inside training, plus an optional post-cutoff holdout.

    Folds never cross the cutoff.  When ``holdout_end`` extends beyond the
    cutoff, a separate single fold starts at the cutoff and is reported apart
    from selection: selection uses it only after the model is fixed, as the
    time-forward check that the plan actually generalises.
    """
    origins = backtest_origins(start_day, cutoff_day, horizon)
    folds = []
    per_key_models: dict[str, dict[str, list[dict[str, Any]]]] = {
        key.id: {name: [] for name in CANDIDATE_FITTERS} for key in keys
    }
    for origin in origins:
        origin_dt = day_start(origin)
        fold = {"origin": origin.isoformat(), "target_start": origin.isoformat(), "target_end": (origin + timedelta(days=horizon)).isoformat(), "keys": []}
        for key in keys:
            fold["keys"].append(_fold_key(series, key, start_day, origin, origin_dt, horizon, per_key_models))
        folds.append(fold)

    by_key = []
    for key in keys:
        summary_models: dict[str, Any] = {}
        for name in CANDIDATE_FITTERS:
            results = per_key_models[key.id][name]
            total_actual = sum(item["actual_hours"] for item in results)
            summary_models[name] = {
                "folds_available": len(results),
                "folds_expected": len(origins),
                "complete": len(results) == len(origins) and bool(origins),
                "mae": round(sum(item["absolute_error"] for item in results) / len(results), 6) if results else None,
                "wape": round(
                    sum(item["absolute_error"] for item in results) / total_actual, 6
                )
                if results and total_actual > 0
                else None,
                "bias_hours": round(sum(item["error_hours"] for item in results) / len(results), 6) if results else None,
                "coverage_80": f"{sum(item['covered_80'] for item in results)}/{len(results)}" if results else None,
                "coverage_95": f"{sum(item['covered_95'] for item in results)}/{len(results)}" if results else None,
                "coverage_80_rate": round(sum(item["covered_80"] for item in results) / len(results), 6) if results else None,
                "mean_interval_80_width": round(
                    sum(item["upper_80"] - item["lower_80"] for item in results) / len(results), 6
                )
                if results
                else None,
            }
        selected, basis = select_model(summary_models, len(origins))
        by_key.append({"key": key.id, "selected_model": selected, "selection_basis": basis, "models": summary_models})

    total_scored = sum(len(per_key_models[key.id][name]) for key in keys for name in CANDIDATE_FITTERS)
    holdout = None
    if holdout_end is not None:
        holdout = _holdout(series, keys, cutoff_day, horizon, by_key, holdout_end)
    return {
        "protocol": (
            "rolling origin; at each origin only arrivals whose available_at precedes the origin are used; "
            "target windows never extend past planning.final_training_cutoff; "
            "the post-cutoff holdout is scored only after selection"
        ),
        "horizon_days": horizon,
        "origins": [origin.isoformat() for origin in origins],
        "fold_count": len(origins),
        "minimum_required_folds": MIN_FOLDS,
        "folds": folds,
        "by_key": by_key,
        "scored_forecasts": total_scored,
        "holdout": holdout,
    }


def _holdout(
    series: DemandSeries,
    keys: list[SeriesKey],
    cutoff_day: date,
    horizon: int,
    by_key: list[dict[str, Any]],
    coverage_end: date | None = None,
) -> dict[str, Any]:
    """Score the *selected* model on the first window after the cutoff.

    The window is scored on its observed prefix only.  When the ledger stops
    inside the window the fold is labelled partial; when it stops before the
    window begins nothing is scored.  Unobserved days are never treated as zero
    demand.
    """
    origin_dt = day_start(cutoff_day)
    window_end = cutoff_day + timedelta(days=horizon)
    observed_end = window_end if coverage_end is None else min(window_end, coverage_end + timedelta(days=1))
    observed_days = max(0, (observed_end - cutoff_day).days)
    if observed_days == 0:
        return {
            "window": {"start": cutoff_day.isoformat(), "end_exclusive": window_end.isoformat()},
            "purpose": "time-forward check after the cutoff; never used to choose the model",
            "available": False,
            "partial": False,
            "observed_days": 0,
            "reason": (
                f"ledger coverage ends {coverage_end.isoformat() if coverage_end else 'unknown'}, "
                "before the holdout window begins; nothing is scored"
            ),
            "keys": [],
            "available_keys": 0,
            "wape": None,
            "coverage_80": None,
        }
    rows = []
    for key in keys:
        selection = next(entry for entry in by_key if entry["key"] == key.id)
        selected = selection["selected_model"]
        if selected is None:
            rows.append({"key": key.id, "selected_model": None, "available": False, "reason": "no selected model"})
            continue
        fit = CANDIDATE_FITTERS[selected](series, key, series.first_day(key) or cutoff_day, cutoff_day, origin_dt)
        if not fit.available:
            rows.append({"key": key.id, "selected_model": selected, "available": False, "reason": fit.unavailable_reason})
            continue
        target_days = [cutoff_day + timedelta(days=i) for i in range(observed_days)]
        actual = sum(series.arrival_on(key, day) for day in target_days)
        point = observed_days * fit.daily_rate
        interval = predictive_interval(point, observed_days * fit.daily_variance, fit.effective_n)
        scored = _score(point, interval, actual)
        rows.append({"key": key.id, "selected_model": selected, "available": True, **scored})
    available = [row for row in rows if row.get("available")]
    total_actual = sum(row["actual_hours"] for row in available)
    return {
        "window": {"start": cutoff_day.isoformat(), "end_exclusive": window_end.isoformat()},
        "observed_window": {
            "start": cutoff_day.isoformat(),
            "end_exclusive": observed_end.isoformat(),
            "days": observed_days,
        },
        "purpose": "time-forward check after the cutoff; never used to choose the model",
        "available": True,
        "partial": observed_days < horizon,
        "observed_days": observed_days,
        "keys": rows,
        "available_keys": len(available),
        "wape": round(sum(row["absolute_error"] for row in available) / total_actual, 6) if total_actual else None,
        "coverage_80": f"{sum(row['covered_80'] for row in available)}/{len(available)}" if available else None,
    }


def select_model(summary: dict[str, Any], expected_folds: int) -> tuple[str | None, str]:
    eligible = [
        (name, item)
        for name, item in summary.items()
        if item["complete"] and item["wape"] is not None and expected_folds >= MIN_FOLDS
    ]
    if not eligible:
        return None, "no candidate completed every fold"
    best_wape = min(item["wape"] for _, item in eligible)
    threshold = best_wape * (1.0 + PREFERENCE_MARGIN)
    near_best = [name for name, item in eligible if item["wape"] <= threshold]
    for name in PARSIMONY:
        if name in near_best:
            chosen = name
            break
    else:  # pragma: no cover - PARSIMONY covers every candidate
        chosen = min(eligible, key=lambda pair: pair[1]["wape"])[0]
    chosen_wape = summary[chosen]["wape"]
    if chosen_wape <= best_wape + 1e-12:
        basis = f"lowest WAPE ({chosen_wape:.3f}); margin rule kept the simplest candidate"
    else:
        basis = (
            f"WAPE {chosen_wape:.3f} is within {PREFERENCE_MARGIN:.0%} of the best "
            f"({best_wape:.3f}); the more parsimonious candidate was preferred"
        )
    return chosen, basis


# --------------------------------------------------------------------------
# support decision
# --------------------------------------------------------------------------
def training_profile(series: DemandSeries, key: SeriesKey, start_day: date, cutoff_day: date) -> dict[str, Any]:
    days = _window(start_day, cutoff_day, (cutoff_day - start_day).days)
    origin_ts = day_start(cutoff_day)
    values = _values(series, key, days, origin_ts)
    nonzero = sum(1 for value in values if value > 0)
    longest, _, _ = longest_zero_run(values)
    trailing = 0
    for value in reversed(values):
        if value == 0:
            trailing += 1
        else:
            break
    first_nonzero = next((days[i] for i, value in enumerate(values) if value > 0), None)
    last_nonzero = next((days[i] for i, value in reversed(list(enumerate(values))) if value > 0), None)
    # Requests in the profile must be the same observable set the fit sees:
    # arrived before the cutoff day and available by the cutoff instant.
    requests = sum(
        1
        for entry in series.evidence.values()
        if key.matches(entry.request.region, entry.request.resource)
        and entry.request.available_at <= origin_ts
        and utc_day(entry.request.arrived_at) < cutoff_day
    )
    return {
        "days": len(values),
        "start": days[0].isoformat() if days else None,
        "end": days[-1].isoformat() if days else None,
        "nonzero_days": nonzero,
        "zero_day_share": round(1 - nonzero / len(values), 6) if values else None,
        "longest_zero_run_days": longest,
        "trailing_zero_days": trailing,
        "first_nonzero_day": first_nonzero.isoformat() if first_nonzero else None,
        "last_nonzero_day": last_nonzero.isoformat() if last_nonzero else None,
        "requested_hours": round(sum(values), 6),
        "requests": requests,
        "observed_through": origin_ts.isoformat(),
    }


def decide_support(
    key: SeriesKey,
    profile: dict[str, Any],
    backtest_entry: dict[str, Any],
    planning_known: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    limitations: list[str] = []
    status = "supported"

    if profile["requests"] == 0:
        status = "unsupported"
        reasons.append("no_observations")
        limitations.append("no request rows exist for this region/resource, so no rate can be estimated")
    elif profile["nonzero_days"] < MIN_NONZERO_DAYS:
        status = "unsupported"
        reasons.append("sparse_nonzero_history")
        limitations.append(
            f"only {profile['nonzero_days']} non-zero days are observable; a rate estimate would be fabricated"
        )
    elif profile["days"] < MIN_TRAINING_DAYS:
        status = "unsupported"
        reasons.append("short_history")
        limitations.append(f"only {profile['days']} training days are observable")
    else:
        selected = backtest_entry.get("selected_model")
        models = backtest_entry.get("models", {})
        chosen = models.get(selected) if selected else None
        if selected is None or chosen is None or not chosen.get("complete"):
            status = "limited"
            reasons.append("backtest_incomplete")
            limitations.append("no forecast candidate completed every rolling-origin fold")
        else:
            wape = chosen.get("wape")
            coverage = chosen.get("coverage_80_rate")
            if wape is None:
                status = "unsupported"
                reasons.append("no_positive_actuals_in_backtest")
            elif wape > LIMITED_WAPE_LIMIT:
                status = "unsupported"
                reasons.append("backtest_error_too_high")
                limitations.append(f"backtest WAPE {wape:.2f} exceeds {LIMITED_WAPE_LIMIT:.2f}")
            elif wape > SUPPORT_WAPE_LIMIT:
                status = "limited"
                reasons.append("backtest_error_moderate")
                limitations.append(f"backtest WAPE {wape:.2f} exceeds {SUPPORT_WAPE_LIMIT:.2f}")
            elif coverage is not None and coverage < SUPPORT_COVERAGE_LIMIT:
                status = "limited"
                reasons.append("interval_coverage_low")
                limitations.append(f"80% interval covered only {coverage:.0%} of folds")
        if profile["trailing_zero_days"] >= ZERO_WINDOW_LIMIT:
            if status == "supported":
                status = "limited"
            reasons.append("consecutive_zero_demand_window")
            limitations.append(
                f"{profile['trailing_zero_days']} consecutive zero-demand days precede the cutoff; "
                "the point estimate is shrunk toward the parent and the interval widened"
            )
        if profile["longest_zero_run_days"] >= ZERO_WINDOW_LIMIT:
            limitations.append(
                f"history contains a zero-demand run of {profile['longest_zero_run_days']} days, "
                "so the level is not stable"
            )

    if not planning_known:
        limitations.append("cell appears in planning hints only; it has no capacity or request history")
    return {
        "key": key.id,
        "level": key.level,
        "region": key.region,
        "resource": key.resource,
        "status": status,
        "reasons": reasons,
        "limitations": limitations,
    }
