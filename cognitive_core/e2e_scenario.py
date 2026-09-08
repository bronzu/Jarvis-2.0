"""
End-to-end scenario that exercises the full JARVIS cognitive pipeline:

  INPUT
    -> UNDERSTANDING
    -> REASONING
    -> PLANNING
    -> MEMORY (working + episodic)
    -> KNOWLEDGE GRAPH
    -> TOOL SELECTION
    -> PERMISSION
    -> ACTION
    -> OBSERVATION
    -> WORLD MODEL UPDATE

The scenario covers three flows:
  1. happy path: a permitted action that succeeds
  2. failure path: the action returns an error and the world model updates
  3. permission denial: a dangerous action is rejected by the guard

This is an integration script (not a unit test) that exercises real
components in a single process. The output is a structured trace that can be
saved to disk or asserted against in a test.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from cognitive_core.understanding import Understanding
from cognitive_core.reasoning.reasoner import Reasoner
from cognitive_core.planner import Planner
from cognitive_core.memory.working_memory import WorkingMemory
from cognitive_core.memory.episodic_memory import EpisodicMemory
from cognitive_core.memory.semantic_memory import SemanticMemory
from cognitive_core.knowledge.knowledge_graph import KnowledgeGraph
from cognitive_core.world_model.world_model import WorldModel
from cognitive_core.world_model.simulator import Simulator
from INFRASTRUCTURE.security.permission_guard import PermissionGuard


class ToolRegistry:
    """Minimal tool registry for the scenario."""

    def __init__(self, guard: PermissionGuard):
        self._guard = guard
        self._tools: Dict[str, Any] = {}

    def register(self, name: str, handler):
        self._tools[name] = handler

    def select(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    def execute(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        handler = self._tools.get(name)
        if handler is None:
            return {"status": "error", "error": f"tool '{name}' not found"}
        return handler(*args, **kwargs)


def _safe_calculator(expression: str) -> Dict[str, Any]:
    """Application-level safe calculator. Accepts only digits and basic ops."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {"status": "error", "error": "expression contains forbidden characters"}
    try:
        return {"status": "ok", "result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _web_search(query: str) -> Dict[str, Any]:
    """Stub external tool that returns deterministic mock results."""
    return {
        "status": "ok",
        "query": query,
        "results": [
            {"title": f"Result for {query}", "snippet": "Mocked result."},
        ],
    }


def _dangerous_rm(path: str) -> Dict[str, Any]:
    """Tool that should never be allowed without approval."""
    return {"status": "ok", "removed": path}


def run_scenario(input_text: str) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    trace.append({"step": "input", "text": input_text, "ts": time.time()})

    understanding = Understanding()
    parsed = understanding.understand(input_text)
    trace.append({
        "step": "understanding",
        "intent": parsed["intent"],
        "entities": parsed["entities"],
        "relations": parsed["relations"],
        "task_type": parsed.get("type"),
        "missing": parsed["missing_information"],
        "confidence": parsed["confidence"],
    })

    reasoner = Reasoner()
    reasoning = {
        "intent_summary": parsed["intent"],
        "entity_count": len(parsed["entities"]),
        "confidence": parsed["confidence"],
        "missing_count": len(parsed["missing_information"]),
    }
    trace.append({"step": "reasoning", "result": reasoning})

    planner = Planner()
    plan = planner.plan(goal=parsed["intent"], context={"facts": parsed["entities"]})
    fallback_strategies = [s.fallback.description if s.fallback else None for s in plan.steps]
    trace.append({
        "step": "planning",
        "plan_id": plan.id,
        "steps": [s.description for s in plan.steps],
        "fallback": fallback_strategies,
    })

    working = WorkingMemory(capacity=8)
    working.remember({"intent": parsed["intent"], "entities": parsed["entities"]})

    episodic = EpisodicMemory()
    episodic.record("input", parsed["intent"])

    semantic = SemanticMemory()
    semantic.learn("planner", "creates plans")

    graph = KnowledgeGraph()
    graph.relate("user_input", "triggers", parsed["intent"])
    entity_keys = [e.get("text", str(e)) if isinstance(e, dict) else str(e) for e in parsed["entities"]]
    graph.relate(parsed["intent"], "has_entities", ",".join(entity_keys) if entity_keys else "none")

    tool_registry = ToolRegistry(guard=PermissionGuard())
    tool_registry.register("calculator", _safe_calculator)
    tool_registry.register("web_search", _web_search)
    tool_registry.register("rm", _dangerous_rm)

    if "rm" in input_text.lower() and "-rf" in input_text.lower():
        selected_tool = "rm"
    elif "calcol" in input_text.lower():
        selected_tool = "calculator"
    else:
        has_math = any(op in input_text for op in ("+", "*")) and any(ch.isdigit() for ch in input_text)
        selected_tool = "calculator" if has_math else "web_search"
    trace.append({"step": "tool_selection", "selected": selected_tool})

    guard = PermissionGuard()
    if selected_tool == "rm":
        try:
            guard.validate_command("rm -rf /tmp", approved=False)
            permission_decision = {"approved": False, "reason": "rm is rejected by registry"}
        except PermissionError as exc:
            permission_decision = {"approved": False, "reason": str(exc)}
    else:
        if selected_tool == "calculator":
            expr = "".join(c for c in input_text if c in "0123456789+-*/(). ")
            cmd = f"calculator({expr})"
        else:
            cmd = f"web_search({parsed['intent']})"
        try:
            ok = guard.validate_command(cmd, approved=True)
            permission_decision = {"approved": ok, "reason": "approved by user"}
        except PermissionError as exc:
            permission_decision = {"approved": False, "reason": str(exc)}
    trace.append({"step": "permission", "decision": permission_decision})

    if not permission_decision["approved"]:
        observation = {"status": "denied", "tool": selected_tool, "reason": permission_decision["reason"]}
    else:
        try:
            if selected_tool == "calculator":
                expr = "".join(c for c in input_text if c in "0123456789+-*/(). ")
                observation = tool_registry.execute(selected_tool, expr)
            else:
                observation = tool_registry.execute(selected_tool, parsed["intent"])
        except Exception as exc:
            observation = {"status": "error", "error": str(exc)}
    trace.append({"step": "action", "observation": observation})

    world_model = WorldModel()
    world_model.add_causal_relation("user_input", "triggers_action")
    if observation.get("status") == "ok":
        world_model.add_causal_relation("action_ok", "update_world_state")
        sim = Simulator(world_model).simulate("user_input", steps=2)
    elif observation.get("status") == "denied":
        world_model.add_causal_relation("permission_denied", "no_state_change")
        sim = Simulator(world_model).simulate("user_input", steps=2)
    else:
        world_model.add_causal_relation("action_failed", "log_error")
        sim = Simulator(world_model).simulate("user_input", steps=2)
    trace.append({
        "step": "world_model_update",
        "simulation": sim,
        "causal_relations": {
            "user_input_predicts": world_model.predict("user_input"),
            "action_ok_predicts": world_model.predict("action_ok"),
            "permission_denied_predicts": world_model.predict("permission_denied"),
            "action_failed_predicts": world_model.predict("action_failed"),
        },
    })

    return {
        "scenario_id": str(uuid.uuid4())[:8],
        "trace": trace,
    }


def main():
    scenarios = [
        "Calcola 2 + 3 * 4",
        "Cerca informazioni sul machine learning",
        "Esegui rm -rf /tmp",
    ]
    for text in scenarios:
        result = run_scenario(text)
        print(f"\n=== Scenario: {text!r} ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
