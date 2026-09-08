"""
Planner module for JARVIS 2.0 Cognitive Core.

Provides:
- Goal decomposition into Plan
- PlanStep with preconditions, effects, dependencies
- Plan validation
- Fallback handling
- Re-planning
- Integration with GoalManager, Reasoner, Memory, ToolSelector, AutonomyManager, PermissionGuard
"""

import time
import uuid
from typing import Any, Dict, List, Optional


class Precondition:
    def __init__(self, description: str, check: str = "", required: bool = True):
        self.description = description
        self.check = check
        self.required = required

    def to_dict(self):
        return {"description": self.description, "check": self.check, "required": self.required}


class Effect:
    def __init__(self, description: str, type_: str = "state_change"):
        self.description = description
        self.type = type_

    def to_dict(self):
        return {"description": self.description, "type": self.type}


class PlanStep:
    def __init__(self, description: str, action: Optional[Any] = None,
                 preconditions: Optional[List[Precondition]] = None,
                 effects: Optional[List[Effect]] = None,
                 dependencies: Optional[List[str]] = None,
                 fallback: Optional[Any] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.action = action
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.dependencies = dependencies or []
        self.fallback = fallback
        self.status = "pending"
        self.result = None
        self.updated_at = time.time()

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action.__name__ if callable(self.action) else str(self.action),
            "preconditions": [p.to_dict() for p in self.preconditions],
            "effects": [e.to_dict() for e in self.effects],
            "dependencies": list(self.dependencies),
            "fallback": self.fallback.__name__ if callable(self.fallback) else str(self.fallback),
            "status": self.status,
            "result": self.result,
            "updated_at": self.updated_at,
        }


class Plan:
    def __init__(self, goal: str, steps: Optional[List[PlanStep]] = None):
        self.id = str(uuid.uuid4())
        self.goal = goal
        self.steps = steps or []
        self.status = "pending"
        self.created_at = time.time()
        self.updated_at = time.time()

    def add_step(self, step: PlanStep):
        self.steps.append(step)
        self.updated_at = time.time()

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Planner:
    """Pianifica azioni per raggiungere un goal."""

    def __init__(self, reasoner=None, memory=None, tool_selector=None,
                 autonomy_manager=None, permission_guard=None):
        self.reasoner = reasoner
        self.memory = memory
        self.tool_selector = tool_selector
        self.autonomy_manager = autonomy_manager
        self.permission_guard = permission_guard

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Plan:
        if not goal or not str(goal).strip():
            raise ValueError("Il goal non può essere vuoto")

        context = context or {}
        plan = Plan(goal=goal)

        if self.reasoner is not None:
            try:
                reasoning = self.reasoner.reason({"facts": context.get("facts", []), "rules": context.get("rules", []), "goal": goal})
                if reasoning.get("proved"):
                    plan.add_step(PlanStep(
                        description=f"Ragionamento: {goal} è dimostrato",
                        effects=[Effect(description=goal, type_="knowledge")],
                    ))
                    plan.status = "completed"
                    return plan
            except Exception:
                pass

        plan.add_step(PlanStep(
            description=f"Analizza il goal: {goal}",
            preconditions=[Precondition(description="Input disponibile", check="input is not None")],
            effects=[Effect(description="Goal analizzato", type_="state_change")],
        ))

        if self.memory is not None:
            plan.add_step(PlanStep(
                description="Recupera conoscenze precedenti",
                preconditions=[Precondition(description="Memory accessibile", check="memory is not None")],
                effects=[Effect(description="Contesto recuperato", type_="information")],
                dependencies=[plan.steps[0].id],
            ))

        if self.tool_selector is not None:
            plan.add_step(PlanStep(
                description="Seleziona strumento appropriato",
                preconditions=[
                    Precondition(description="Tool selector disponibile", check="tool_selector is not None"),
                    Precondition(description="Permessi verificati", check="permission_guard is not None"),
                ],
                effects=[Effect(description="Tool selezionato", type_="selection")],
                dependencies=[plan.steps[0].id],
            ))

        if self.autonomy_manager is not None and self.permission_guard is not None:
            plan.add_step(PlanStep(
                description="Esegui azione con autorizzazione",
                preconditions=[
                    Precondition(description="Autonomia disponibile", check="autonomy_manager is not None"),
                    Precondition(description="Permessi concessi", check="permission_guard.require_user_approval(True)", required=True),
                ],
                effects=[Effect(description="Azione eseguita", type_="state_change")],
                fallback=PlanStep(
                    description="Azione non autorizzata: usa risposta procedurale",
                    effects=[Effect(description="Fallback procedural", type_="fallback")],
                ),
                dependencies=[plan.steps[-1].id],
            ))

        plan.status = "ready"
        return plan

    def validate(self, plan: Plan) -> Dict[str, Any]:
        errors = []
        warnings = []

        step_map = {s.id: s for s in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_map:
                    errors.append(f"Step {step.id} dipende da step inesistente {dep}")

            for precondition in step.preconditions:
                if precondition.required and not precondition.check:
                    warnings.append(f"Precondition '{precondition.description}' senza check per step {step.id}")

        cycle = self._detect_cycle(plan)
        if cycle:
            errors.append(f"Ciclo rilevato: {' -> '.join(cycle)}")

        status = "valid" if not errors else "invalid"
        return {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "plan_id": plan.id,
        }

    def _detect_cycle(self, plan: Plan) -> Optional[List[str]]:
        step_map = {s.id: s for s in plan.steps}
        visited = set()
        rec_stack = set()

        def dfs(step_id, path):
            visited.add(step_id)
            rec_stack.add(step_id)
            path.append(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.dependencies:
                    if dep in rec_stack:
                        idx = path.index(dep)
                        return path[idx:] + [dep]
                    if dep not in visited:
                        result = dfs(dep, path)
                        if result:
                            return result
            rec_stack.remove(step_id)
            path.pop()
            return None

        for step in plan.steps:
            if step.id not in visited:
                result = dfs(step.id, [])
                if result:
                    return result
        return None

    def execute_step(self, step: PlanStep, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        step.status = "in_progress"
        step.updated_at = time.time()

        try:
            if step.action is not None and callable(step.action):
                result = step.action(context)
            elif step.description:
                result = {"status": "simulated", "description": step.description}
            else:
                raise ValueError("Step senza action o description")

            step.status = "completed"
            step.result = result
            step.updated_at = time.time()
            return {"step_id": step.id, "status": "completed", "result": result}
        except Exception as exc:
            step.status = "failed"
            step.result = {"error": str(exc)}
            step.updated_at = time.time()

            if step.fallback is not None:
                if callable(step.fallback):
                    fallback_result = step.fallback(context)
                else:
                    fallback_result = {"status": "fallback", "description": str(step.fallback)}
                return {"step_id": step.id, "status": "fallback", "result": fallback_result, "error": str(exc)}

            return {"step_id": step.id, "status": "failed", "error": str(exc)}

    def execute(self, plan: Plan, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        validation = self.validate(plan)
        if validation["status"] == "invalid":
            return {"plan_id": plan.id, "status": "invalid", "errors": validation["errors"]}

        step_map = {s.id: s for s in plan.steps}
        executed = set()
        results = []

        for step in plan.steps:
            deps_met = all(d in executed for d in step.dependencies)
            if not deps_met:
                results.append({
                    "step_id": step.id,
                    "status": "skipped",
                    "reason": "dependencies not met",
                })
                continue

            result = self.execute_step(step, context)
            results.append(result)
            if result["status"] in ("completed", "fallback"):
                executed.add(step.id)

        plan.status = "completed" if all(r["status"] in ("completed", "fallback", "skipped") for r in results) else "failed"
        plan.updated_at = time.time()
        return {"plan_id": plan.id, "status": plan.status, "results": results}
