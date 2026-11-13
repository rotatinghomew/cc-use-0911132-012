from __future__ import annotations
from typing import Any
from emergency_resource_dispatch.resources.runtime import RunState, SQLiteLedger, fingerprint, number, sequence

class DependencyDispatcher:
    artifact_key = 'dispatch_dependencies'

    def execute(self, state: RunState, stages: list[Any], ledger: SQLiteLedger) -> None:
        dependencies = {stages[0].name: [], stages[1].name: [stages[0].name], stages[2].name: [stages[0].name], stages[3].name: [stages[1].name, stages[2].name], stages[4].name: [stages[3].name], stages[5].name: [stages[2].name], stages[6].name: [stages[4].name, stages[5].name]}
        completed = set()
        schedule = []
        while len(completed) < len(stages):
            ready = [stage for stage in stages if stage.name not in completed and set(dependencies[stage.name]) <= completed]
            if not ready:
                raise ValueError("dependency cycle")
            for stage in ready:
                stage.apply(state)
                completed.add(stage.name)
                schedule.append({"stage": stage.name, "depends_on": dependencies[stage.name], "position": len(schedule) + 1})
                ledger.save(state)
        state.artifacts[self.artifact_key] = {"dependencies": dependencies, "schedule": schedule, "completed": list(completed), "signature": fingerprint(schedule)}
        state.emit(self.artifact_key, "dependency_dispatch_completed", tasks=len(schedule))
