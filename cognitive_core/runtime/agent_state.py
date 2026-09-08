"""Agent state for the JARVIS Agent Runtime.

The AgentState is the *complete, serializable* picture of an agent at a
single point in time. It is the single object the runtime serializes for
checkpoint/resume/replay.

Why this exists
---------------
The previous JARVIS architecture scattered state across several places
(GoalManager, ObservationFeedback, EpisodicMemory, etc.). The Agent
Runtime needs a single, self-contained, JSON-serializable object that
represents the running task so it can be:

  * saved   — written to disk for crash safety
  * restored — rehydrated after a process restart
  * checkpointed — periodic snapshots
  * resumed  — continued from a saved state
  * audited — read by the AuditSystem
  * replayed — fed back into a deterministic loop for debugging

What it contains (per the master prompt, Rule 11):

  task_id
  goal, subgoals
  current_plan, current_step
  belief_state      (Phase 4 will add a real BeliefEngine; here a dict)
  working_memory, relevant_memory
  world_state
  observations, actions, action_results
  verification_results, failures, hypotheses
  uncertainty       (float in [0, 1])
  permissions, budget, deadline
  agent_identity, capabilities, active_skills
  termination_reason, checkpoint
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional


class StateError(RuntimeError):
    """Raised when agent state cannot be constructed or serialized."""


def _normalize_budget(b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(b or {})
    for key in ("steps", "tokens", "tool_calls"):
        v = out.get(key, 0)
        out[key] = int(v) if v else 0
    unlimited = out.get("_unlimited", [])
    if isinstance(unlimited, set):
        unlimited = sorted(unlimited)
    out["_unlimited"] = list(unlimited)
    if int(out.get("tokens", 0)) == 0 and "tokens" not in out["_unlimited"]:
        out["_unlimited"].append("tokens")
    if int(out.get("tool_calls", 0)) == 0 and "tool_calls" not in out["_unlimited"]:
        out["_unlimited"].append("tool_calls")
    return out


_DEFAULT_CAPABILITIES = (
    "memory.read",
    "memory.write",
    "reasoning.use",
    "tool.select",
)


class AgentState:
    """Serializable, versioned agent state.

    Thread-safety: this class does NOT provide internal locking. The
    AgentRuntime owns synchronization. Treat instances as value objects
    that the runtime snapshots before each step.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        task_id: Optional[str] = None,
        goal: str = "",
        subgoals: Optional[List[str]] = None,
        current_plan: Optional[Dict[str, Any]] = None,
        current_step: Optional[str] = None,
        belief_state: Optional[Dict[str, Any]] = None,
        working_memory: Optional[Dict[str, Any]] = None,
        relevant_memory: Optional[List[Dict[str, Any]]] = None,
        world_state: Optional[Dict[str, Any]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        action_results: Optional[List[Dict[str, Any]]] = None,
        verification_results: Optional[List[Dict[str, Any]]] = None,
        failures: Optional[List[Dict[str, Any]]] = None,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
        uncertainty: float = 1.0,
        permissions: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, Any]] = None,
        deadline: Optional[float] = None,
        agent_identity: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
        active_skills: Optional[List[str]] = None,
        termination_reason: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_id = task_id or str(uuid.uuid4())
        if not isinstance(goal, str):
            raise StateError("goal must be a string")
        self.goal = goal
        self.subgoals: List[str] = list(subgoals or [])
        self.current_plan: Optional[Dict[str, Any]] = current_plan
        self.current_step: Optional[str] = current_step
        self.belief_state: Dict[str, Any] = dict(belief_state or {})
        self.working_memory: Dict[str, Any] = dict(working_memory or {})
        self.relevant_memory: List[Dict[str, Any]] = list(relevant_memory or [])
        self.world_state: Dict[str, Any] = dict(world_state or {})
        self.observations: List[Dict[str, Any]] = list(observations or [])
        self.actions: List[Dict[str, Any]] = list(actions or [])
        self.action_results: List[Dict[str, Any]] = list(action_results or [])
        self.verification_results: List[Dict[str, Any]] = list(verification_results or [])
        self.failures: List[Dict[str, Any]] = list(failures or [])
        self.hypotheses: List[Dict[str, Any]] = list(hypotheses or [])
        self.uncertainty: float = max(0.0, min(1.0, float(uncertainty)))
        self.permissions: Dict[str, Any] = dict(permissions or {})
        # Budget semantics: each key has a value and an "unlimited" flag.
        # A value of 0 means "unlimited" (default) to avoid spuriously
        # exhausted budgets for unused dimensions. A negative value means
        # the budget is set but already exhausted.
        self.budget: Dict[str, Any] = _normalize_budget(
            budget if budget is not None else {"steps": 100, "tokens": 0, "tool_calls": 0}
        )
        self.deadline: Optional[float] = deadline
        self.agent_identity: Dict[str, Any] = dict(agent_identity or {"name": "JARVIS"})
        self.capabilities: List[str] = list(capabilities or list(_DEFAULT_CAPABILITIES))
        self.active_skills: List[str] = list(active_skills or [])
        self.termination_reason: Optional[str] = termination_reason
        self.checkpoint: Optional[Dict[str, Any]] = checkpoint
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at

    # ------------------------------------------------------------------
    # Budget / deadline accounting
    # ------------------------------------------------------------------
    def consume_budget(self, steps: int = 0, tokens: int = 0, tool_calls: int = 0) -> None:
        unlimited = set(self.budget.get("_unlimited", []) or [])
        if "steps" not in unlimited:
            self.budget["steps"] = max(0, int(self.budget.get("steps", 0)) - int(steps))
        if "tokens" not in unlimited:
            self.budget["tokens"] = max(0, int(self.budget.get("tokens", 0)) - int(tokens))
        if "tool_calls" not in unlimited:
            self.budget["tool_calls"] = max(0, int(self.budget.get("tool_calls", 0)) - int(tool_calls))
        self.updated_at = time.time()

    def budget_exhausted(self) -> bool:
        unlimited = set(self.budget.get("_unlimited", []) or [])
        for key in ("steps", "tokens", "tool_calls"):
            if key in unlimited:
                continue
            if int(self.budget.get(key, 0)) <= 0:
                return True
        return False

    def deadline_exceeded(self, now: Optional[float] = None) -> bool:
        if self.deadline is None:
            return False
        return (now or time.time()) >= self.deadline

    # ------------------------------------------------------------------
    # Append-only event streams (for replay/audit)
    # ------------------------------------------------------------------
    def record_observation(self, observation: Dict[str, Any]) -> None:
        if not isinstance(observation, dict):
            raise StateError("observation must be a dict")
        self.observations.append(dict(observation))
        self.updated_at = time.time()

    def record_action(self, action: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(action, dict):
            raise StateError("action must be a dict")
        self.actions.append(dict(action))
        if result is not None:
            if not isinstance(result, dict):
                raise StateError("action result must be a dict")
            self.action_results.append(dict(result))
        self.updated_at = time.time()

    def record_verification(self, verification: Dict[str, Any]) -> None:
        if not isinstance(verification, dict):
            raise StateError("verification must be a dict")
        self.verification_results.append(dict(verification))
        self.updated_at = time.time()

    def record_failure(self, failure: Dict[str, Any]) -> None:
        if not isinstance(failure, dict):
            raise StateError("failure must be a dict")
        self.failures.append(dict(failure))
        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        def _deep(x):
            if isinstance(x, dict):
                return {k: _deep(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_deep(v) for v in x]
            return x
        return {
            "schema_version": self.SCHEMA_VERSION,
            "task_id": self.task_id,
            "goal": self.goal,
            "subgoals": list(self.subgoals),
            "current_plan": _deep(self.current_plan),
            "current_step": self.current_step,
            "belief_state": _deep(self.belief_state),
            "working_memory": _deep(self.working_memory),
            "relevant_memory": _deep(self.relevant_memory),
            "world_state": _deep(self.world_state),
            "observations": _deep(self.observations),
            "actions": _deep(self.actions),
            "action_results": _deep(self.action_results),
            "verification_results": _deep(self.verification_results),
            "failures": _deep(self.failures),
            "hypotheses": _deep(self.hypotheses),
            "uncertainty": self.uncertainty,
            "permissions": _deep(self.permissions),
            "budget": _deep(self.budget),
            "deadline": self.deadline,
            "agent_identity": _deep(self.agent_identity),
            "capabilities": list(self.capabilities),
            "active_skills": list(self.active_skills),
            "termination_reason": self.termination_reason,
            "checkpoint": _deep(self.checkpoint),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        if not isinstance(data, dict):
            raise StateError("AgentState payload must be a dict")
        schema = data.get("schema_version", cls.SCHEMA_VERSION)
        if schema != cls.SCHEMA_VERSION:
            raise StateError(f"unsupported schema_version: {schema!r}")
        st = cls(
            task_id=data.get("task_id"),
            goal=data.get("goal", ""),
            subgoals=data.get("subgoals", []),
            current_plan=data.get("current_plan"),
            current_step=data.get("current_step"),
            belief_state=data.get("belief_state", {}),
            working_memory=data.get("working_memory", {}),
            relevant_memory=data.get("relevant_memory", []),
            world_state=data.get("world_state", {}),
            observations=data.get("observations", []),
            actions=data.get("actions", []),
            action_results=data.get("action_results", []),
            verification_results=data.get("verification_results", []),
            failures=data.get("failures", []),
            hypotheses=data.get("hypotheses", []),
            uncertainty=data.get("uncertainty", 1.0),
            permissions=data.get("permissions", {}),
            budget=_normalize_budget(
                data.get("budget", {"steps": 100, "tokens": 0, "tool_calls": 0})
            ),
            deadline=data.get("deadline"),
            agent_identity=data.get("agent_identity", {"name": "JARVIS"}),
            capabilities=data.get("capabilities", list(_DEFAULT_CAPABILITIES)),
            active_skills=data.get("active_skills", []),
            termination_reason=data.get("termination_reason"),
            checkpoint=data.get("checkpoint"),
        )
        st.created_at = float(data.get("created_at", st.created_at))
        st.updated_at = float(data.get("updated_at", st.updated_at))
        return st

    @classmethod
    def from_json(cls, text: str) -> "AgentState":
        """Load state from JSON. Accepts either:
        - a raw state dict (output of ``to_dict``), or
        - a checkpoint wrapper ``{"saved_at": ..., "schema_version": ..., "snapshot": {...}}``
        """
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("snapshot"), dict) and "goal" not in data and "task_id" not in data:
            # Looks like a checkpoint wrapper — unwrap.
            data = data["snapshot"]
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> Dict[str, Any]:
        snap = self.to_dict()
        self.checkpoint = {
            "saved_at": time.time(),
            "schema_version": self.SCHEMA_VERSION,
            "snapshot": snap,
        }
        return self.checkpoint

    def restore_from_checkpoint(self) -> None:
        if not self.checkpoint:
            raise StateError("no checkpoint to restore from")
        snap = self.checkpoint.get("snapshot")
        if not snap:
            raise StateError("checkpoint is empty")
        restored = AgentState.from_dict(snap)
        restored.checkpoint = self.checkpoint
        self.__dict__.update(restored.__dict__)
