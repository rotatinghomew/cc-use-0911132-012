"""Read-only demand planning analysis.

This package deliberately does not execute dispatch.  It reads the operational
ledger for evidence and writes only a separate ``demand_plans`` record; it never
calls ApplicationService or the scheduling stages.
"""

from emergency_resource_dispatch.demand_planning.report import run_analysis

__all__ = ["run_analysis"]
