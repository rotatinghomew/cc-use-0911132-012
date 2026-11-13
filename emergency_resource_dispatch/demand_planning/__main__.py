"""CLI: python3 -m emergency_resource_dispatch.demand_planning --input ... --database ... --output ..."""
from __future__ import annotations

import argparse
import json
import sys

from emergency_resource_dispatch.demand_planning.inputs import PlanningInputError
from emergency_resource_dispatch.demand_planning.report import run_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="供给受限记录中的资源到达需求预测（只读分析，不执行调度）"
    )
    parser.add_argument("--input", required=True, help="analysis snapshot JSON")
    parser.add_argument("--database", required=True, help="operational ledger database (read-only for evidence)")
    parser.add_argument("--output", required=True, help="where to write the plan JSON")
    args = parser.parse_args(argv)
    try:
        plan = run_analysis(args.input, args.database, args.output)
    except PlanningInputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "plan_id": plan["plan_id"],
        "output": args.output,
        "forecasts": len(plan["forecasts"]),
        "supported": sum(1 for item in plan["support"] if item["status"] == "supported"),
        "limited": sum(1 for item in plan["support"] if item["status"] == "limited"),
        "unsupported": sum(1 for item in plan["support"] if item["status"] == "unsupported"),
        "storage": plan.get("storage"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
