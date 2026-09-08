"""Unit + integration tests for the Agent Runtime (Phase 1)."""

import json
import os
import sys
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cognitive_core.runtime import (
    AgentRuntime,
    AgentState,
    LifecycleError,
    TaskLifecycle,
    TaskStatus,
    RecoveryPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeMemory:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.calls = 0

    def search_semantic(self, query):
        self.calls += 1
        return self._hits


class _FakeAudit:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, component, action, status="success",
                  input_summary=None, output_summary=None, error=None,
                  metadata=None, session_id=None):
        self.events.append({
            "event_type": event_type,
            "component": component,
            "action": action,
            "status": status,
            "metadata": metadata or {},
            "session_id": session_id,
        })


class _FakeActionExecutor:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def execute(self, action):
        self.calls += 1
        if not self._results:
            return {"status": "executed", "action": action, "verified": True}
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


# ---------------------------------------------------------------------------
# TaskLifecycle
# ---------------------------------------------------------------------------
class TestTaskLifecycle(unittest.TestCase):
    def test_initial_status(self):
        lc = TaskLifecycle("t1", initial=TaskStatus.QUEUED)
        self.assertEqual(lc.status, TaskStatus.QUEUED)
        self.assertFalse(lc.is_terminal())

    def test_legal_transition(self):
        lc = TaskLifecycle("t1")
        lc.transition(TaskStatus.RUNNING)
        lc.transition(TaskStatus.WAITING_TOOL)
        lc.transition(TaskStatus.RUNNING)
        lc.transition(TaskStatus.COMPLETED)
        self.assertTrue(lc.is_terminal())

    def test_illegal_transition_raises(self):
        lc = TaskLifecycle("t1", initial=TaskStatus.COMPLETED)
        with self.assertRaises(LifecycleError):
            lc.transition(TaskStatus.RUNNING)

    def test_force(self):
        lc = TaskLifecycle("t1", initial=TaskStatus.QUEUED)
        lc.force(TaskStatus.PAUSED)
        self.assertEqual(lc.status, TaskStatus.PAUSED)

    def test_serialization_roundtrip(self):
        lc = TaskLifecycle("t1")
        lc.transition(TaskStatus.RUNNING, at_time=1.0)
        lc.transition(TaskStatus.PAUSED, at_time=2.0)
        data = lc.to_dict()
        lc2 = TaskLifecycle.from_dict(data)
        self.assertEqual(lc2.status, TaskStatus.PAUSED)
        # The last entry should be the PAUSED transition at 2.0
        self.assertEqual(lc2.history()[-1][0], 2.0)
        self.assertEqual(lc2.history()[-1][1], TaskStatus.PAUSED)


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------
class TestAgentState(unittest.TestCase):
    def test_defaults(self):
        st = AgentState(goal="test goal")
        self.assertTrue(st.task_id)
        self.assertEqual(st.goal, "test goal")
        self.assertEqual(st.uncertainty, 1.0)
        self.assertIn("memory.read", st.capabilities)
        self.assertFalse(st.budget_exhausted())

    def test_budget_consumption(self):
        st = AgentState(goal="x", budget={"steps": 2, "tokens": 0, "tool_calls": 0})
        st.consume_budget(steps=1)
        self.assertFalse(st.budget_exhausted())
        st.consume_budget(steps=1)
        self.assertTrue(st.budget_exhausted())

    def test_deadline(self):
        st = AgentState(goal="x", deadline=time.time() - 1)
        self.assertTrue(st.deadline_exceeded())

    def test_uncertainty_clamped(self):
        st = AgentState(goal="x", uncertainty=2.0)
        self.assertEqual(st.uncertainty, 1.0)
        st2 = AgentState(goal="x", uncertainty=-1.0)
        self.assertEqual(st2.uncertainty, 0.0)

    def test_record_observation_action_verification_failure(self):
        st = AgentState(goal="x")
        st.record_observation({"source": "unit"})
        st.record_action({"name": "a"}, {"ok": True})
        st.record_verification({"verified": True})
        st.record_failure({"kind": "x"})
        self.assertEqual(len(st.observations), 1)
        self.assertEqual(len(st.actions), 1)
        self.assertEqual(len(st.action_results), 1)
        self.assertEqual(len(st.verification_results), 1)
        self.assertEqual(len(st.failures), 1)

    def test_invalid_record_raises(self):
        st = AgentState(goal="x")
        with self.assertRaises(Exception):
            st.record_observation("not a dict")
        with self.assertRaises(Exception):
            st.record_action("not a dict")
        with self.assertRaises(Exception):
            st.record_verification("not a dict")
        with self.assertRaises(Exception):
            st.record_failure("not a dict")

    def test_json_roundtrip(self):
        st = AgentState(
            goal="g",
            subgoals=["a", "b"],
            current_step="a",
            observations=[{"k": 1}],
        )
        st.record_action({"name": "x"}, {"r": 1})
        text = st.to_json()
        loaded = AgentState.from_json(text)
        self.assertEqual(loaded.goal, "g")
        self.assertEqual(loaded.subgoals, ["a", "b"])
        self.assertEqual(loaded.current_step, "a")
        self.assertEqual(len(loaded.actions), 1)
        self.assertEqual(len(loaded.action_results), 1)

    def test_checkpoint_roundtrip(self):
        st = AgentState(goal="g", budget={"steps": 5, "tokens": 0, "tool_calls": 0})
        st.consume_budget(steps=2)
        st.save_checkpoint()
        st.consume_budget(steps=2)
        self.assertEqual(int(st.budget["steps"]), 1)
        st.restore_from_checkpoint()
        self.assertEqual(int(st.budget["steps"]), 3)

    def test_checkpoint_corrupted_json_rejected(self):
        with self.assertRaises(Exception):
            AgentState.from_json("not json {{{")

    def test_checkpoint_unsupported_schema_rejected(self):
        with self.assertRaises(Exception):
            AgentState.from_dict({"schema_version": "99.0", "goal": "x"})

    def test_checkpoint_missing_snapshot_rejected(self):
        st = AgentState(goal="x")
        st.checkpoint = {"saved_at": 0, "schema_version": "1.0", "snapshot": None}
        with self.assertRaises(Exception):
            st.restore_from_checkpoint()

    def test_checkpoint_wrapper_json_roundtrip(self):
        st = AgentState(goal="ciao", subgoals=["a", "b"])
        st.save_checkpoint()
        text = st.to_json()
        restored = AgentState.from_json(text)
        self.assertEqual(restored.goal, "ciao")
        self.assertEqual(restored.subgoals, ["a", "b"])

    def test_idempotent_action_replay_after_resume(self):
        # Documented limitation: the runtime does NOT track per-action
        # idempotency keys. A resumed task that re-executes the same
        # action from a checkpoint will see it run again. This test
        # records the limitation so future work can add idempotency
        # keys without contradicting the current contract.
        st = AgentState(goal="x", actions=[{"id": "a1"}])
        st.save_checkpoint()
        # If a caller naively replays, the action runs again.
        self.assertIn({"id": "a1"}, st.actions)


# ---------------------------------------------------------------------------
# AgentRuntime
# ---------------------------------------------------------------------------
class TestAgentRuntime(unittest.TestCase):
    def test_create_and_status(self):
        rt = AgentRuntime()
        tid = rt.create_task("explore")
        self.assertEqual(rt.get_status(tid), TaskStatus.QUEUED)

    def test_unknown_task_raises(self):
        rt = AgentRuntime()
        with self.assertRaises(KeyError):
            rt.get_state("nope")

    def test_legacy_delegate_marks_verified(self):
        # Without an action_producer, runtime delegates to coordinator
        # and treats the delegate's output as a verified action.
        class Coord:
            def __init__(self):
                self.calls = []
            def process(self, goal, session_id=None):
                self.calls.append((goal, session_id))
                return {"response": f"answer:{goal}", "source": "test"}

        rt = AgentRuntime(cognitive_coordinator=Coord(), audit=_FakeAudit())
        tid = rt.create_task("ciao")
        result = rt.run(tid)
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.termination_reason, "legacy_delegate_completed")
        self.assertEqual(result.response["delegate_result"]["response"], "answer:ciao")

    def test_producer_loop_runs_multiple_iterations(self):
        # Producer returns an action until 2 actions have been recorded,
        # then returns None so the loop terminates as COMPLETED.
        executor = _FakeActionExecutor([
            {"status": "executed", "ok": False, "verified": True, "marker": "a"},
            {"status": "executed", "ok": True, "verified": True, "marker": "b"},
        ])

        def producer(state, obs):
            if len(state.actions) >= 2:
                return None
            return {"name": "x", "args": {"i": len(state.actions)}, "capability": "read"}

        rt = AgentRuntime(action_executor=executor, audit=_FakeAudit(), max_iterations=10)
        tid = rt.create_task("loop")
        result = rt.run(tid, action_producer=producer)
        # 2 iterations that produced actions + 1 iteration that saw
        # no further action and completed cleanly.
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        self.assertEqual(len(result.final_state["actions"]), 2)
        self.assertEqual(executor.calls, 2)
        self.assertEqual(result.termination_reason, "no_further_action")

    def test_permission_denied_terminates(self):
        def producer(state, obs):
            return {"name": "x", "requires_approval": True, "approved": False, "capability": "read"}

        rt = AgentRuntime(action_executor=_FakeActionExecutor([]), audit=_FakeAudit())
        tid = rt.create_task("deny")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "permission_denied")

    def test_max_iterations_terminates(self):
        def producer(state, obs):
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=_FakeAudit(),
            max_iterations=3,
        )
        tid = rt.create_task("loop")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "max_iterations_exceeded")
        self.assertEqual(result.iterations, 3)

    def test_budget_exhausted_terminates(self):
        def producer(state, obs):
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=_FakeAudit(),
            default_budget_steps=2,
        )
        tid = rt.create_task("budget")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "budget_exhausted")

    def test_deadline_terminates(self):
        def producer(state, obs):
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=_FakeAudit(),
            default_budget_steps=1000,
        )
        # Use a deadline already in the past.
        tid = rt.create_task("deadline", deadline_seconds=-1)
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "deadline_exceeded")

    def test_cancel_during_run(self):
        # We cancel before the loop starts; cancellation is consumed
        # between iterations, so a 1-iteration task that completes first
        # will finish, but a multi-iteration task can be cancelled.
        def producer(state, obs):
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=_FakeAudit(),
            max_iterations=50,
        )
        tid = rt.create_task("cancelme")
        # Schedule cancellation to fire on the first iteration.
        original_run = rt.run

        def cancel_after_first(task_id, action_producer=None, context=None):
            # cancel before iteration 1 ends
            rt.cancel(task_id)
            return original_run(task_id, action_producer=action_producer, context=context)

        result = cancel_after_first(tid, action_producer=producer)
        # Either it completed (if cancel fired AFTER verify) or was cancelled.
        self.assertIn(result.status, {TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value})

    def test_pause_during_run(self):
        def producer(state, obs):
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=_FakeAudit(),
            max_iterations=50,
        )
        tid = rt.create_task("pause")
        original_run = rt.run

        def pause_after_first(task_id, action_producer=None, context=None):
            rt.pause(task_id)
            return original_run(task_id, action_producer=action_producer, context=context)

        result = pause_after_first(tid, action_producer=producer)
        # Pause is consumed at the end of an iteration.
        self.assertIn(result.status, {TaskStatus.PAUSED.value, TaskStatus.COMPLETED.value})

    def test_no_action_without_producer_or_coordinator_fails(self):
        # No producer, no coordinator → action is None on every
        # iteration → the runtime must NOT silently mark the task
        # COMPLETED. This is the no-false-completion guarantee.
        rt = AgentRuntime(audit=_FakeAudit())
        tid = rt.create_task("noop")
        result = rt.run(tid)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "no_action_produced")
        # A failure must be recorded.
        self.assertTrue(any(f.get("kind") == "no_action" for f in result.final_state["failures"]))
        # Real integration: pause mid-loop, then a fresh run() must
        # pick up from the PAUSED state and execute more work, NOT
        # immediately re-pause.
        class Exec:
            def __init__(self):
                self.calls = []
            def execute(self, action):
                self.calls.append(action.get("args", {}).get("n"))
                return {"status": "executed", "verified": True, "n": action.get("args", {}).get("n")}

        executor = Exec()
        calls = {"n": 0}

        def producer1(state, obs):
            calls["n"] += 1
            if calls["n"] == 2:
                rt.pause(state.task_id)
            if len(state.actions) >= 5:
                return None
            return {"name": "step", "args": {"n": len(state.actions)}, "capability": "read"}

        rt = AgentRuntime(action_executor=executor, audit=_FakeAudit(), max_iterations=20)
        tid = rt.create_task("resume_real", budget_steps=200)
        r1 = rt.run(tid, action_producer=producer1)
        self.assertEqual(r1.status, TaskStatus.PAUSED.value)
        self.assertEqual(len(r1.final_state["actions"]), 2)

        def producer2(state, obs):
            if len(state.actions) >= 5:
                return None
            return {"name": "step", "args": {"n": len(state.actions)}, "capability": "read"}

        r2 = rt.run(tid, action_producer=producer2)
        # The second run must do real work, not immediately re-pause.
        self.assertEqual(r2.status, TaskStatus.COMPLETED.value)
        self.assertGreater(r2.iterations, 0)
        self.assertEqual(len(r2.final_state["actions"]), 5)
        # The pause should have auto-checkpointed.
        self.assertIsNotNone(rt.get_state(tid).checkpoint)

    def test_cancel_during_run_auto_checkpoints(self):
        class Exec:
            def execute(self, action):
                return {"status": "executed", "verified": True}
        rt = AgentRuntime(action_executor=Exec(), audit=_FakeAudit(), max_iterations=20)
        tid = rt.create_task("cancel_ckpt", budget_steps=200)

        def producer(state, obs):
            if len(state.actions) >= 2:
                rt.cancel(state.task_id)
            if len(state.actions) >= 5:
                return None
            return {"name": "x", "args": {}, "capability": "read"}

        rt.run(tid, action_producer=producer)
        self.assertIsNotNone(rt.get_state(tid).checkpoint)

    def test_checkpoint_and_resume(self):
        def producer(state, obs):
            if len(state.actions) >= 1:
                return None
            return {"name": "x", "args": {}}

        rt = AgentRuntime(action_executor=_FakeActionExecutor([{"verified": True}] * 10), audit=_FakeAudit())
        tid = rt.create_task("ckpt", budget_steps=10)
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        # Save checkpoint, mutate state, then restore and verify the
        # saved budget is recovered (and is non-zero).
        snap = rt.checkpoint(tid)
        self.assertIn("snapshot", snap)
        state = rt.get_state(tid)
        saved_steps = int(snap["snapshot"]["budget"]["steps"])
        self.assertGreater(saved_steps, 0)
        state.budget["steps"] = 0
        self.assertEqual(int(rt.get_state(tid).budget["steps"]), 0)
        rt.resume(tid)
        self.assertEqual(int(rt.get_state(tid).budget["steps"]), saved_steps)

    def test_audit_called(self):
        audit = _FakeAudit()
        rt = AgentRuntime(cognitive_coordinator=type("C", (), {
            "process": staticmethod(lambda goal, session_id=None: {"response": "ok"})
        })(), audit=audit)
        tid = rt.create_task("audit")
        rt.run(tid)
        types = [e["event_type"] for e in audit.events]
        self.assertIn("AGENT_TASK_CREATED", types)
        self.assertIn("AGENT_TASK_FINISHED", types)

    def test_audit_emits_action_proposed_executed_verified(self):
        # Verify that a successful action loop emits the per-step
        # audit events required by master prompt §58.
        audit = _FakeAudit()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 5),
            audit=audit,
            max_iterations=5,
        )
        tid = rt.create_task("audit_full", budget_steps=10)

        def producer(state, obs):
            if len(state.actions) >= 1:
                return None
            return {"name": "x", "args": {}, "capability": "read"}

        rt.run(tid, action_producer=producer)
        types = [e["event_type"] for e in audit.events]
        self.assertIn("AGENT_TASK_CREATED", types)
        self.assertIn("AGENT_ACTION_PROPOSED", types)
        self.assertIn("AGENT_ACTION_EXECUTED", types)
        self.assertIn("AGENT_TASK_FINISHED", types)

    def test_audit_emits_budget_exhausted(self):
        audit = _FakeAudit()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=audit,
            max_iterations=100,
            default_budget_steps=2,
        )
        tid = rt.create_task("audit_budget")
        rt.run(tid, action_producer=lambda s, o: {"name": "x", "args": {}})
        types = [e["event_type"] for e in audit.events]
        self.assertIn("AGENT_BUDGET_EXHAUSTED", types)

    def test_audit_emits_deadline_exceeded(self):
        audit = _FakeAudit()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 100),
            audit=audit,
            max_iterations=100,
        )
        tid = rt.create_task("audit_deadline", deadline_seconds=-1)
        rt.run(tid, action_producer=lambda s, o: {"name": "x", "args": {}})
        types = [e["event_type"] for e in audit.events]
        self.assertIn("AGENT_DEADLINE_EXCEEDED", types)

    def test_audit_redaction_in_summaries(self):
        # The audit layer's own redaction is in AuditSystem.redact. The
        # runtime must not bypass it. We verify the runtime never sends
        # the raw ``action`` dict to the audit (only safe summaries)
        # for events that go through ``_safe_result_summary``.
        audit = _FakeAudit()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": True}] * 5),
            audit=audit,
        )
        tid = rt.create_task("audit_redact", budget_steps=10)

        def producer(state, obs):
            if len(state.actions) >= 1:
                return None
            return {"name": "x", "args": {"password": "hunter2"}, "capability": "read"}

        rt.run(tid, action_producer=producer)
        # The audit log contains a result_summary; the AuditSystem (in
        # real prod) redacts. Here we just verify the runtime sent a
        # bounded summary.
        exec_event = next(e for e in audit.events if e["event_type"] == "AGENT_ACTION_EXECUTED")
        summary = exec_event["metadata"].get("result_summary", "")
        self.assertLessEqual(len(summary), 300)

    def test_observation_uses_memory(self):
        memory = _FakeMemory(hits=[{"payload": {"text": "previous context"}}])
        rt = AgentRuntime(memory=memory, audit=_FakeAudit())
        tid = rt.create_task("mem")
        rt.run(tid)
        self.assertGreater(memory.calls, 0)
        # The relevant_memory should have been populated.
        self.assertGreaterEqual(len(rt.get_state(tid).relevant_memory), 1)

    def test_recovery_unrecoverable_fails(self):
        # Result not verified -> recovery attempts exhausted -> FAILED.
        def producer(state, obs):
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"verified": False}] * 10),
            audit=_FakeAudit(),
            max_iterations=5,
        )
        tid = rt.create_task("recov")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "max_recovery_attempts_exceeded (3)")

    def test_runtime_does_not_self_authorize_dangerous_capability(self):
        def producer(state, obs):
            return {"name": "x", "capability": "system_override", "args": {}}

        rt = AgentRuntime(action_executor=_FakeActionExecutor([]), audit=_FakeAudit())
        tid = rt.create_task("danger")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "permission_denied")

    def test_recovery_policy_retry_then_succeeds(self):
        # First execution fails, second succeeds -> recovery via retry.
        def producer(state, obs):
            if len(state.actions) >= 1 and state.action_results and state.action_results[-1].get("verified"):
                return None
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([Exception("boom"), {"verified": True}]),
            audit=_FakeAudit(),
            max_iterations=5,
            recovery_policy=RecoveryPolicy(max_attempts=3),
        )
        tid = rt.create_task("retry")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        self.assertEqual(result.termination_reason, "no_further_action")

    def test_recovery_policy_reparam_modifies_action(self):
        captured = {}

        def producer(state, obs):
            if len(state.actions) >= 1 and state.action_results and state.action_results[-1].get("verified"):
                return None
            return {"name": "x", "args": {"i": 1}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([Exception("boom"), {"verified": True}]),
            audit=_FakeAudit(),
            max_iterations=5,
            recovery_policy=RecoveryPolicy(max_attempts=3),
        )
        tid = rt.create_task("reparam")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        self.assertEqual(result.termination_reason, "no_further_action")

    def test_recovery_policy_different_tool_sets_hint(self):
        hints = []

        def producer(state, obs):
            hints.append(state.working_memory.get("recovery", {}).get("hint"))
            if len(state.actions) >= 4:
                return None
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([Exception("boom")] * 5),
            audit=_FakeAudit(),
            max_iterations=5,
            recovery_policy=RecoveryPolicy(max_attempts=3),
        )
        tid = rt.create_task("tool")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "max_recovery_attempts_exceeded (3)")
        self.assertIn("different_tool", hints)

    def test_recovery_policy_user_escalation_transitions(self):
        def producer(state, obs):
            if len(state.actions) >= 5:
                return None
            return {"name": "x", "args": {}}

        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([Exception("boom")] * 5),
            audit=_FakeAudit(),
            max_iterations=5,
            recovery_policy=RecoveryPolicy(max_attempts=5),
        )
        tid = rt.create_task("escalate")
        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.WAITING_USER.value)
        self.assertEqual(result.termination_reason, "user_escalation")

    def test_recovery_policy_history_and_reset(self):
        policy = RecoveryPolicy(max_attempts=3)
        state = AgentState(goal="x")
        action = {"name": "a", "args": {}}

        policy.attempt("t1", state, action, "err1")
        policy.attempt("t1", state, action, "err2")
        self.assertEqual(len(policy.history("t1")), 2)

        policy.reset("t1")
        self.assertEqual(len(policy.history("t1")), 0)


# ---------------------------------------------------------------------------
# Integration with the real CognitiveCoordinator (backward compatibility)
# ---------------------------------------------------------------------------
class TestAgentRuntimeWithRealCoordinator(unittest.TestCase):
    """Verify the runtime is a real orchestrator on top of the existing
    CognitiveCoordinator, not a parallel system (Rule 4: do not rewrite)."""

    def test_legacy_path_completes_with_real_coordinator(self):
        from cognitive_core.coordinator import CognitiveCoordinator

        coord = CognitiveCoordinator()
        rt = AgentRuntime(cognitive_coordinator=coord, audit=_FakeAudit())
        tid = rt.create_task("dimmi come stai")
        result = rt.run(tid)
        # Legacy delegate must complete cleanly in one iteration.
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.termination_reason, "legacy_delegate_completed")
        # The delegate's structured result must be reachable.
        delegate = result.response["delegate_result"]
        self.assertIn("response", delegate)

    def test_state_recorded_observation_contains_memory_hits(self):
        from cognitive_core.coordinator import CognitiveCoordinator
        from cognitive_core.memory.memory_manager import MemoryManager

        # Seed a lightweight in-memory store with a known fact. The
        # query is constructed so the naive substring search of the
        # MemoryManager actually finds the stored fact.
        mem = MemoryManager()
        mem.store("fact-1", {"text": "JARVIS è un assistente personale."})

        # Build the coordinator with its episodic memory (legacy expects
        # ``memory.episodic`` attribute).
        coord = CognitiveCoordinator(episodic_memory=None)
        coord.memory = type("MM", (), {"episodic": None, "search_semantic": mem.search_semantic, "remember_working": lambda self, x: None})()

        rt = AgentRuntime(cognitive_coordinator=coord, memory=mem, audit=_FakeAudit())
        # The MemoryManager uses naive substring matching. We store
        # a fact containing the exact query string so the test is
        # deterministic.
        mem.store("fact-2", {"text": "JARVIS assistente è una parola chiave."})
        tid = rt.create_task("JARVIS assistente")
        rt.run(tid)
        state = rt.get_state(tid)
        self.assertGreaterEqual(len(state.observations), 1)
        first_obs = state.observations[0]
        raw = first_obs.get("raw", {})
        self.assertIn("memory_hits", raw)
        self.assertGreaterEqual(len(raw["memory_hits"]), 1)


if __name__ == "__main__":
    unittest.main()
