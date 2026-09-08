"""Agent runtime for JARVIS.

This module provides the AgentRuntime: the *primary* orchestrator of
the JARVIS cognitive loop.

Architectural role (per the JARVIS 2.0 -> 3.0 master prompt, Phase 1+2):

    USER
      ↓
    AgentRuntime               <-- THIS MODULE (sits at the top)
      ↓
    CognitiveCoordinator       <-- existing delegate (kept for back-compat)
      ↓
    memory, reasoning, planning, tools, world model, ...

The runtime implements the conceptual loop:

    OBSERVE
      ↓
    UNDERSTAND
      ↓
    SELECT ACTION (structured)
      ↓
    AUTHORIZE (preconditions + security)
      ↓
    EXECUTE
      ↓
    OBSERVE RESULT (structured observation)
      ↓
    VERIFY EFFECT (verified / failed / unknown / unverified)
      ↓
    EVALUATE
      ↓
    UPDATE MEMORY
      ↓
    REPLAN? (continue iterating)

Phase 2 additions
-----------------
* Structured Action schema with preconditions, risk, provenance.
* ActionAuthorizer wrapping existing PermissionGuard / SecurityLayer.
* Structured Observation with evidence, provenance, confidence.
* Verification layer distinguishing VERIFIED / FAILED / UNKNOWN / UNVERIFIED.
* Production-path wiring through ExecutiveRouter.

Spec compliance
---------------
* Does NOT bypass the existing security / permission guard. Any side
  effect must pass through ``action_executor`` or ``tool_system``, both
  of which call the ``PermissionGuard`` and the ``AuditSystem``.
* Does NOT modify the autonomy split (cognitive=YES, constructive=YES,
  authorizative=NO). The runtime never grants itself a permission.
* Does NOT replace the existing ``CognitiveCoordinator``. It uses it
  as a delegate, so all existing tests and the CLI keep working.
* Does NOT implement training.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .agent_state import AgentState
from .task_lifecycle import (
    LifecycleError,
    TaskLifecycle,
    TaskStatus,
)
from .action import (
    ActionAuthorizer,
    ActionError,
    ActionRisk,
    ActionStatus,
    create_action,
    validate_action,
)
from .verification import (
    DefaultVerifier,
    SideEffectVerifier,
    VerificationResult,
    VerificationStatus,
)
from .recovery import RecoveryPolicy


logger = logging.getLogger("JARVIS.AgentRuntime")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class AgentLoopResult:
    """The outcome of a single ``run()`` invocation of the AgentRuntime."""
    task_id: str
    status: str
    iterations: int = 0
    final_state: Optional[Dict[str, Any]] = None
    response: Any = None
    termination_reason: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "iterations": self.iterations,
            "final_state": self.final_state,
            "response": self.response,
            "termination_reason": self.termination_reason,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A "verifier" is a callable that takes (action, result, state) and returns
# a VerificationResult.
Verifier = Callable[[Dict[str, Any], Dict[str, Any], AgentState], VerificationResult]

# An "action" produced by the planner is a dict like:
#   {"name": "...", "args": {...}, "capability": "..."}
# The runtime calls ``action_executor`` to actually run it.
ActionProducer = Callable[[AgentState, Dict[str, Any]], Optional[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_result_summary(result: Any, max_len: int = 256) -> str:
    """Bounded, safe textual summary of a tool result for audit logs."""
    try:
        text = str(result)
    except Exception:
        return "<non-serializable>"
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class AgentRuntime:
    """Orchestrates the cognitive loop on top of the existing components.

    The runtime is intentionally constructed with explicit dependencies.
    Tests can inject a fake ``cognitive_coordinator``, a fake
    ``action_executor``, a fake ``memory``, a fake ``audit`` and a fake
    ``verifier`` to deterministically exercise the loop.
    """

    def __init__(
        self,
        cognitive_coordinator: Any = None,
        action_executor: Any = None,
        memory: Any = None,
        audit: Any = None,
        verifier: Optional[Verifier] = None,
        max_iterations: int = 8,
        default_budget_steps: int = 100,
        default_deadline_seconds: Optional[float] = None,
        action_authorizer: Optional[ActionAuthorizer] = None,
        observation_feedback: Any = None,
        tool_system: Any = None,
        recovery_policy: Optional[RecoveryPolicy] = None,
    ) -> None:
        self.cognitive_coordinator = cognitive_coordinator
        self.action_executor = action_executor
        self.memory = memory
        self.audit = audit
        self._verifier: Verifier = verifier or _default_verifier
        try:
            mi = int(max_iterations)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_iterations must be an integer, got {max_iterations!r}") from exc
        if mi < 1:
            raise ValueError(f"max_iterations must be >= 1, got {mi}")
        self._max_iterations = mi
        try:
            dbs = int(default_budget_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"default_budget_steps must be an integer, got {default_budget_steps!r}") from exc
        if dbs < 1:
            raise ValueError(f"default_budget_steps must be >= 1, got {dbs}")
        self._default_budget_steps = dbs
        self._default_deadline_seconds = default_deadline_seconds
        self._action_authorizer = action_authorizer or ActionAuthorizer()
        self._observation_feedback = observation_feedback
        self._tool_system = tool_system
        self._recovery_policy = recovery_policy or RecoveryPolicy()
        self._recovered_action: Optional[Dict[str, Any]] = None
        # Tasks tracked by the runtime: task_id -> (lifecycle, state, last_history)
        self._tasks: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Task lifecycle management
    # ------------------------------------------------------------------
    def create_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        subgoals: Optional[List[str]] = None,
        budget_steps: Optional[int] = None,
        deadline_seconds: Optional[float] = None,
        permissions: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not goal or not str(goal).strip():
            raise ValueError("goal cannot be empty")
        if budget_steps is None:
            steps_budget = int(self._default_budget_steps)
        else:
            try:
                steps_budget = int(budget_steps)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"budget_steps must be an integer, got {budget_steps!r}") from exc
            if steps_budget < 0:
                raise ValueError(f"budget_steps must be >= 0, got {steps_budget}")
        state = AgentState(
            goal=str(goal).strip(),
            subgoals=subgoals,
            budget={"steps": steps_budget, "tokens": 0, "tool_calls": 0},
            deadline=(
                time.time() + float(deadline_seconds)
                if deadline_seconds is not None
                else (
                    time.time() + float(self._default_deadline_seconds)
                    if self._default_deadline_seconds is not None
                    else None
                )
            ),
            permissions=dict(permissions or {}),
            working_memory={"context": dict(context or {})},
        )
        lifecycle = TaskLifecycle(state.task_id, initial=TaskStatus.QUEUED)
        self._tasks[state.task_id] = {
            "lifecycle": lifecycle,
            "state": state,
            "_cancel_requested": False,
            "_pause_requested": False,
        }
        self._audit("AGENT_TASK_CREATED", state.task_id, {"goal": state.goal})
        return state.task_id

    def get_state(self, task_id: str) -> AgentState:
        return self._require_task(task_id)["state"]

    def get_status(self, task_id: str) -> TaskStatus:
        return self._require_task(task_id)["lifecycle"].status

    def cancel(self, task_id: str) -> None:
        entry = self._require_task(task_id)
        entry["_cancel_requested"] = True

    def pause(self, task_id: str) -> None:
        entry = self._require_task(task_id)
        entry["_pause_requested"] = True

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------
    def checkpoint(self, task_id: str) -> Dict[str, Any]:
        state = self.get_state(task_id)
        snap = state.save_checkpoint()
        self._audit("AGENT_CHECKPOINT", task_id, {"saved_at": snap.get("saved_at")})
        return snap

    def resume(self, task_id: str) -> AgentState:
        state = self.get_state(task_id)
        if not state.checkpoint:
            raise RuntimeError(f"no checkpoint for task {task_id}")
        state.restore_from_checkpoint()
        self._audit("AGENT_RESUMED", task_id, {})
        return state

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        task_id: str,
        action_producer: Optional[ActionProducer] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentLoopResult:
        """Run the cognitive loop until terminal status.

        ``action_producer`` decides what action to take each iteration. If
        ``None``, the runtime uses the existing CognitiveCoordinator as a
        one-shot delegate (legacy behavior). This keeps backward
        compatibility while allowing new code to inject a real agent loop.
        """
        entry = self._require_task(task_id)
        lifecycle: TaskLifecycle = entry["lifecycle"]
        state: AgentState = entry["state"]
        if context:
            state.working_memory.setdefault("context", {}).update(dict(context))

        history: List[Dict[str, Any]] = []
        iterations = 0

        resuming_from_paused = lifecycle.status == TaskStatus.PAUSED
        pre_cancel = bool(entry["_cancel_requested"]) and not resuming_from_paused
        pre_pause = bool(entry["_pause_requested"]) and not resuming_from_paused
        entry["_cancel_requested"] = False
        entry["_pause_requested"] = False

        try:
            lifecycle.transition(TaskStatus.RUNNING, at_time=time.time())
        except LifecycleError:
            if lifecycle.status != TaskStatus.PAUSED:
                raise
            lifecycle.transition(TaskStatus.RUNNING, at_time=time.time())

        if pre_cancel:
            state.save_checkpoint()
            self._audit("AGENT_CHECKPOINT", state.task_id, {"reason": "pre_run_cancel"})
            lifecycle.transition(TaskStatus.CANCELLED, at_time=time.time())
            state.termination_reason = "cancelled_before_run"
            history.append(self._trace(0, "cancelled_pre_run", state))
            self._audit("AGENT_TASK_FINISHED", state.task_id, {
                "status": lifecycle.status.value,
                "iterations": 0,
                "termination_reason": state.termination_reason,
            })
            return AgentLoopResult(
                task_id=state.task_id,
                status=lifecycle.status.value,
                iterations=0,
                final_state=state.to_dict(),
                response=None,
                termination_reason=state.termination_reason,
                history=history,
            )

        if pre_pause:
            state.save_checkpoint()
            self._audit("AGENT_CHECKPOINT", state.task_id, {"reason": "pre_run_pause"})
            lifecycle.transition(TaskStatus.PAUSED, at_time=time.time())
            state.termination_reason = "paused_before_run"
            history.append(self._trace(0, "paused_pre_run", state))
            self._audit("AGENT_TASK_FINISHED", state.task_id, {
                "status": lifecycle.status.value,
                "iterations": 0,
                "termination_reason": state.termination_reason,
            })
            return AgentLoopResult(
                task_id=state.task_id,
                status=lifecycle.status.value,
                iterations=0,
                final_state=state.to_dict(),
                response=None,
                termination_reason=state.termination_reason,
                history=history,
            )

        # MAIN LOOP
        while True:
            if iterations >= self._max_iterations:
                lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                state.termination_reason = "max_iterations_exceeded"
                history.append(self._trace(iterations, "max_iterations_exceeded", state))
                break
            iterations += 1
            entry["_pause_requested"] = False
            entry["_cancel_requested"] = False

            if state.budget_exhausted():
                self._audit("AGENT_BUDGET_EXHAUSTED", state.task_id, {
                    "budget": dict(state.budget),
                })
                lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                state.termination_reason = "budget_exhausted"
                history.append(self._trace(iterations, "budget_exhausted", state))
                break
            if state.deadline_exceeded():
                self._audit("AGENT_DEADLINE_EXCEEDED", state.task_id, {
                    "deadline": state.deadline,
                })
                lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                state.termination_reason = "deadline_exceeded"
                history.append(self._trace(iterations, "deadline_exceeded", state))
                break

            # Step 1: OBSERVE / UNDERSTAND
            observation = self._observe(state, context)
            understanding = self._understand(state, observation)
            structured_obs = self._build_observation(state, observation, understanding)
            state.record_observation({
                "source": "runtime.observe",
                "raw": observation,
                "interpreted": understanding,
                "ts": time.time(),
            })
            state.consume_budget(steps=1)

            # Step 2: produce a candidate action
            legacy_used = False
            if self._recovered_action is not None:
                raw_action = self._recovered_action
                self._recovered_action = None
            elif action_producer is not None:
                raw_action = action_producer(state, {"observation": observation, "understanding": understanding})
            else:
                raw_action = self._legacy_action(state, observation)
                legacy_used = True

            if raw_action is None:
                if not state.actions and not state.working_memory.get("last_verified_result"):
                    lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                    state.termination_reason = "no_action_produced"
                    state.record_failure({"kind": "no_action", "reason": "no_producer_and_no_coordinator"})
                    history.append(self._trace(iterations, "no_action_produced", state))
                    break
                lifecycle.transition(TaskStatus.COMPLETED, at_time=time.time())
                state.termination_reason = "no_further_action"
                history.append(self._trace(iterations, "completed", state))
                break

            # Normalize to structured action
            action = self._normalize_action(raw_action)

            # Step 3: AUTHORIZE
            authorized = self._authorize_structured(state, action)
            if not authorized["ok"]:
                state.record_failure({"kind": "permission_denied", "action": action, "reason": authorized.get("reason")})
                self._audit("AGENT_PERMISSION_DENIED", state.task_id, {
                    "action": action,
                    "reason": authorized.get("reason"),
                })
                lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                state.termination_reason = "permission_denied"
                history.append(self._trace(iterations, "permission_denied", state, action=action))
                break

            # Step 4: EXECUTE
            self._audit("AGENT_ACTION_PROPOSED", state.task_id, {
                "action": action,
                "iteration": iterations,
            })
            try:
                result = self._execute(action)
            except Exception as exc:
                state.record_failure({"kind": "execution_error", "action": action, "error": str(exc)})
                self._audit("AGENT_EXECUTION_ERROR", state.task_id, {
                    "action": action,
                    "error": str(exc),
                })
                lifecycle.transition(TaskStatus.RECOVERING, at_time=time.time())
                history.append(self._trace(iterations, "execution_error", state, action=action, error=str(exc)))
                recovery = self._attempt_recovery(state, action, str(exc))
                if not recovery["ok"]:
                    if recovery.get("strategy") == "user_escalation":
                        lifecycle.transition(TaskStatus.WAITING_USER, at_time=time.time())
                        state.termination_reason = "user_escalation"
                        history.append(self._trace(iterations, "user_escalation", state, action=action))
                        break
                    lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                    state.termination_reason = recovery.get("reason", "execution_error_unrecoverable")
                    history.append(self._trace(iterations, "execution_error_unrecoverable", state, action=action, error=str(exc)))
                    break
                if recovery.get("action") is not None:
                    self._recovered_action = recovery["action"]
                lifecycle.transition(TaskStatus.RUNNING, at_time=time.time())
                continue

            state.record_action(action, result)
            self._audit("AGENT_ACTION_EXECUTED", state.task_id, {
                "action": action,
                "result_summary": _safe_result_summary(result),
            })

            # Step 5: OBSERVE RESULT + VERIFY
            verification = self._verify(action, result, state)
            verification_dict = verification.to_dict()
            state.record_verification(verification_dict)
            structured_obs["verification"] = verification_dict
            if self._observation_feedback is not None:
                try:
                    self._observation_feedback.record_structured(
                        action=action,
                        observation=structured_obs,
                        verification=verification_dict,
                        session_id=state.task_id,
                    )
                except Exception as exc:
                    logger.warning("ObservationFeedback recording failed: %s", exc)

            if verification.status != VerificationStatus.VERIFIED:
                state.record_failure({"kind": "verification_failed", "action": action, "verification": verification_dict})
                self._audit("AGENT_VERIFICATION_FAILED", state.task_id, {
                    "action": action,
                    "verification": verification_dict,
                })
                lifecycle.transition(TaskStatus.RECOVERING, at_time=time.time())
                history.append(self._trace(iterations, "verification_failed", state, action=action, verification=verification_dict))
                recovery = self._attempt_recovery(state, action, verification.reason)
                if not recovery["ok"]:
                    if recovery.get("strategy") == "user_escalation":
                        lifecycle.transition(TaskStatus.WAITING_USER, at_time=time.time())
                        state.termination_reason = "user_escalation"
                        history.append(self._trace(iterations, "user_escalation", state, action=action))
                        break
                    lifecycle.transition(TaskStatus.FAILED, at_time=time.time())
                    state.termination_reason = recovery.get("reason", "verification_failed_unrecoverable")
                    history.append(self._trace(iterations, "verification_failed_unrecoverable", state, action=action, verification=verification_dict))
                    break
                if recovery.get("action") is not None:
                    self._recovered_action = recovery["action"]
                lifecycle.transition(TaskStatus.RUNNING, at_time=time.time())
                continue

            # Step 6: EVALUATE / update working memory
            state.working_memory["last_verified_action"] = action
            state.working_memory["last_verified_result"] = result
            state.uncertainty = max(0.0, state.uncertainty - 0.1)
            history.append(self._trace(iterations, "verified", state, action=action, result=result))

            if legacy_used:
                lifecycle.transition(TaskStatus.COMPLETED, at_time=time.time())
                state.termination_reason = "legacy_delegate_completed"
                history.append(self._trace(iterations, "legacy_completed", state))
                break

            if entry["_cancel_requested"]:
                state.save_checkpoint()
                self._audit("AGENT_CHECKPOINT", state.task_id, {"reason": "cancel"})
                lifecycle.transition(TaskStatus.CANCELLED, at_time=time.time())
                state.termination_reason = "cancelled_by_caller"
                break
            if entry["_pause_requested"]:
                state.save_checkpoint()
                self._audit("AGENT_CHECKPOINT", state.task_id, {"reason": "pause"})
                lifecycle.transition(TaskStatus.PAUSED, at_time=time.time())
                state.termination_reason = "paused_by_caller"
                break

        self._audit("AGENT_TASK_FINISHED", state.task_id, {
            "status": lifecycle.status.value,
            "iterations": iterations,
            "termination_reason": state.termination_reason,
        })

        return AgentLoopResult(
            task_id=state.task_id,
            status=lifecycle.status.value,
            iterations=iterations,
            final_state=state.to_dict(),
            response=state.working_memory.get("last_verified_result"),
            termination_reason=state.termination_reason,
            history=history,
        )

    # ------------------------------------------------------------------
    # Action / Observation / Verification helpers
    # ------------------------------------------------------------------
    def _normalize_action(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw action dict into a structured action."""
        if not isinstance(raw, dict):
            raise ActionError("action must be a dict")
        if raw.get("id") and raw.get("name") and raw.get("capability"):
            return validate_action(raw)
        name = raw.get("name", "unknown")
        capability = raw.get("capability", "unknown")
        args = dict(raw.get("args") or {})
        result = raw.get("result")
        if isinstance(result, dict):
            args.setdefault("_embedded_result", result)
        action = create_action(
            name=name,
            capability=capability,
            args=args,
            requires_approval=bool(raw.get("requires_approval")),
            approved=bool(raw.get("approved", False)),
            risk=ActionRisk(raw.get("risk", ActionRisk.LOW.value)),
            metadata=raw.get("metadata"),
            provenance=raw.get("provenance"),
        )
        if isinstance(result, dict):
            action["result"] = result
        return action

    def _build_observation(self, state: AgentState, raw: Dict[str, Any], understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Build a structured observation dict."""
        return {
            "source": "runtime.observe",
            "timestamp": time.time(),
            "action_id": None,
            "raw": raw,
            "interpreted": understanding,
            "confidence": 1.0 if raw.get("memory_hits") else 0.5,
            "world_state_delta": {},
            "evidence": {"memory_hits": raw.get("memory_hits", [])[:3]},
            "provenance": {"component": "AgentRuntime", "task_id": state.task_id},
        }

    def _authorize_structured(self, state: AgentState, action: Dict[str, Any]) -> Dict[str, Any]:
        """Authorize a structured action using the ActionAuthorizer."""
        try:
            return self._action_authorizer.authorize(action, session_id=state.task_id)
        except ActionError as exc:
            return {"ok": False, "reason": str(exc), "action": action}

    def _verify(self, action: Dict[str, Any], result: Dict[str, Any], state: AgentState) -> VerificationResult:
        """Verify an action's result using the configured verifier."""
        try:
            if hasattr(self._verifier, "verify"):
                raw = self._verifier.verify(action, result, state)
            elif callable(self._verifier):
                raw = self._verifier(action, result, state)
            else:
                raw = self._verifier
            if isinstance(raw, VerificationResult):
                return raw
            if isinstance(raw, dict):
                status = VerificationStatus.UNVERIFIED
                if raw.get("verified") is True:
                    status = VerificationStatus.VERIFIED
                elif raw.get("verified") is False:
                    status = VerificationStatus.FAILED
                elif "status" in raw:
                    try:
                        status = VerificationStatus(raw["status"])
                    except ValueError:
                        pass
                return VerificationResult(
                    status=status,
                    method=raw.get("method", ""),
                    reason=raw.get("reason", ""),
                    evidence=raw.get("evidence", {}),
                    action_id=action.get("id"),
                )
            logger.warning("Verifier returned unexpected type: %s", type(raw))
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                method="verifier.unexpected_type",
                reason=f"verifier_returned_{type(raw).__name__}",
                action_id=action.get("id"),
            )
        except Exception as exc:
            logger.warning("Verifier raised an exception: %s", exc)
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                method="verifier.error",
                reason=str(exc),
                action_id=action.get("id"),
            )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------
    def _observe(self, state: AgentState, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"ts": time.time(), "context": dict(context or {})}
        if self.memory is not None and hasattr(self.memory, "search_semantic"):
            try:
                hits = self.memory.search_semantic(state.goal)
                obs["memory_hits"] = list(hits)[:3]
                state.relevant_memory = list(hits)[:3]
            except Exception as exc:
                obs["memory_error"] = str(exc)
        return obs

    def _understand(self, state: AgentState, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "goal": state.goal,
            "subgoals": list(state.subgoals),
            "memory_hits_count": len(observation.get("memory_hits", [])),
        }

    def _authorize(self, state: AgentState, action: Dict[str, Any]) -> Dict[str, Any]:
        if action.get("requires_approval") and not action.get("approved", False):
            return {"ok": False, "reason": "requires_user_approval"}
        if action.get("capability") in {"dangerous", "system_override"}:
            return {"ok": False, "reason": "capability_not_self_authorizable"}
        return {"ok": True}

    def _execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.action_executor is None:
            if isinstance(action, dict) and isinstance(action.get("result"), dict):
                return action["result"]
            return {"status": "executed", "note": "no_action_executor_configured", "action": action, "verified": False}
        if hasattr(self.action_executor, "execute"):
            return self.action_executor.execute(action)
        return {"status": "executed", "action": action}

    def _attempt_recovery(self, state: AgentState, action: Dict[str, Any], reason: str) -> Dict[str, Any]:
        result = self._recovery_policy.attempt(state.task_id, state, action, reason)
        if not result["ok"]:
            return {
                "ok": False,
                "strategy": result.get("strategy", "none"),
                "reason": result.get("reason", reason),
            }
        return {
            "ok": True,
            "strategy": result.get("strategy", "retry"),
            "action": result.get("action"),
            "reason": result.get("reason", reason),
        }

    def _legacy_action(self, state: AgentState, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use the existing CognitiveCoordinator as a one-shot delegate.

        The delegate result is wrapped as a read-only action.  Verification
        is delegated to the runtime verifier so success is never assumed
        without evidence.
        """
        if self.cognitive_coordinator is None:
            return None
        try:
            result = self.cognitive_coordinator.process(state.goal, session_id=state.task_id)
        except Exception as exc:
            return {
                "name": "legacy.coordinator_failed",
                "args": {"error": str(exc)},
                "capability": "read",
                "result": {"error": str(exc), "verified": False},
            }
        return {
            "name": "legacy.cognitive_coordinator",
            "args": {"goal": state.goal},
            "capability": "read",
            "result": {
                "status": "delegated",
                "delegate_result": result,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_task(self, task_id: str) -> Dict[str, Any]:
        entry = self._tasks.get(task_id)
        if entry is None:
            raise KeyError(f"unknown task_id: {task_id}")
        return entry

    def _audit(self, event_type: str, task_id: str, metadata: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.log_event(
                event_type=event_type,
                component="AgentRuntime",
                action="orchestrate",
                status="success",
                metadata={"task_id": task_id, **(metadata or {})},
                session_id=task_id,
            )
        except Exception as exc:
            try:
                logger.warning("AgentRuntime audit emission failed for %s: %s", event_type, exc)
            except Exception:
                pass

    def _trace(
        self,
        iteration: int,
        event: str,
        state: AgentState,
        action: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        verification: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "event": event,
            "ts": time.time(),
            "task_id": state.task_id,
            "status": self._tasks[state.task_id]["lifecycle"].status.value,
        }
        if action is not None:
            entry["action"] = action
        if result is not None:
            entry["result"] = result
        if verification is not None:
            entry["verification"] = verification
        if error is not None:
            entry["error"] = error
        return entry


# ---------------------------------------------------------------------------
# Default verifier
# ---------------------------------------------------------------------------
def _default_verifier(
    action: Dict[str, Any],
    result: Dict[str, Any],
    state: AgentState,
) -> Dict[str, Any]:
    """Conservative default verifier.

    A real agent should pass a domain-specific verifier. The default
    treats an action as verified only if the result is explicitly marked
    ``verified=True`` (e.g. by the action_executor after a real
    side-effect check). For read-only actions (``capability="read"``) a
    ``response`` or ``delegate_result`` in the result is accepted as
    evidence because the outcome of a read-only action IS the returned
    data. Otherwise it returns ``verified=False`` so the runtime does
    not silently claim success (Rule 15).
    """
    verified = bool(result.get("verified", False))
    if not verified and action.get("capability") == "read":
        if result.get("response") is not None or result.get("delegate_result") is not None:
            verified = True
    return {
        "verified": verified,
        "method": "default.explicit_flag",
        "reason": "" if verified else "result_not_marked_verified",
    }
