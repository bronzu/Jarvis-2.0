"""Recovery policy for the JARVIS Agent Runtime.

Provides strategies for recovering from execution errors and verification
failures:

  - retry: retry the same action
  - reparam: retry with modified parameters
  - different_tool: signal that a different tool/capability is needed
  - different_plan: signal that a different plan/subgoal is needed
  - user_escalation: escalate to the user (WAITING_USER)

The policy tracks recovery attempts per task and applies strategies in
the order above, capping at ``max_attempts``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class RecoveryPolicy:
    """Policy that decides how to recover from failures.

    The policy is intentionally stateless between tasks except for the
    per-task attempt counter.  It does not assume a specific planner or
    tool system; it signals intent through ``state.working_memory`` and
    optional modified actions.
    """

    def __init__(self, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self._attempts: Dict[str, List[Dict[str, Any]]] = {}

    def can_recover(self, task_id: str, state: Any, action: Dict[str, Any], reason: str) -> bool:
        attempts = self._attempts.get(task_id, [])
        return len(attempts) < self.max_attempts

    def attempt(self, task_id: str, state: Any, action: Dict[str, Any], reason: str) -> Dict[str, Any]:
        attempts = self._attempts.setdefault(task_id, [])

        if len(attempts) >= self.max_attempts:
            return {
                "ok": False,
                "strategy": "none",
                "reason": f"max_recovery_attempts_exceeded ({self.max_attempts})",
                "action": None,
            }

        strategies = ["retry", "reparam", "different_tool", "different_plan", "user_escalation"]
        strategy = strategies[min(len(attempts), len(strategies) - 1)]

        recovery_action = self._apply_strategy(task_id, strategy, action, state)

        attempts.append({
            "strategy": strategy,
            "reason": reason,
            "original_action": action,
            "timestamp": time.time(),
        })

        return {
            "ok": strategy != "user_escalation",
            "strategy": strategy,
            "reason": reason,
            "action": recovery_action,
        }

    def _apply_strategy(self, task_id: str, strategy: str, action: Dict[str, Any], state: Any) -> Optional[Dict[str, Any]]:
        if strategy == "retry":
            return action
        elif strategy == "reparam":
            new_action = dict(action)
            args = dict(new_action.get("args") or {})
            args["_recovery_reparam"] = True
            new_action["args"] = args
            return new_action
        elif strategy == "different_tool":
            if hasattr(state, "working_memory"):
                state.working_memory.setdefault("recovery", {})["hint"] = "different_tool"
            return None
        elif strategy == "different_plan":
            if hasattr(state, "working_memory"):
                state.working_memory.setdefault("recovery", {})["hint"] = "different_plan"
            return None
        elif strategy == "user_escalation":
            if hasattr(state, "working_memory"):
                state.working_memory.setdefault("recovery", {})["hint"] = "user_escalation"
            return None
        return None

    def reset(self, task_id: str) -> None:
        self._attempts.pop(task_id, None)

    def history(self, task_id: str) -> List[Dict[str, Any]]:
        return list(self._attempts.get(task_id, []))
