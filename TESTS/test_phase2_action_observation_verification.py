"""Phase 2 tests: Action → Observation → Verification.

Tests cover:
- Structured Action schema and validation
- Action authorization with existing security infrastructure
- Structured Observation recording (legacy + new)
- Verification layer (verified / failed / unknown / unverified)
- AgentRuntime AOV cycle integration
- ExecutiveRouter AgentRuntime wiring
- Security fixes (no os.popen bypass, no auto-approval)
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cognitive_core.runtime import (
    AgentRuntime,
    AgentState,
    ActionAuthorizer,
    ActionError,
    ActionRisk,
    ActionStatus,
    TaskStatus,
    create_action,
    validate_action,
    DefaultVerifier,
    SideEffectVerifier,
    VerificationResult,
    VerificationStatus,
    RecoveryPolicy,
)
from executive_system.observation_feedback import ObservationFeedback
from executive_system.tools.tool_system import ToolSystem
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.security.security_layer import SecurityLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
        return self._results.pop(0)


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------
class TestStructuredAction(unittest.TestCase):
    def test_create_action_minimal(self):
        action = create_action(name="test", capability="read")
        self.assertEqual(action["name"], "test")
        self.assertEqual(action["capability"], "read")
        self.assertEqual(action["status"], ActionStatus.PENDING.value)
        self.assertEqual(action["risk"], ActionRisk.LOW.value)
        self.assertIn("id", action)
        self.assertIn("created_at", action)

    def test_create_action_requires_approval(self):
        action = create_action(name="danger", capability="system_override", requires_approval=True, approved=False)
        self.assertTrue(action["requires_approval"])
        self.assertFalse(action["approved"])

    def test_create_action_missing_name_raises(self):
        with self.assertRaises(ActionError):
            create_action(name="", capability="read")

    def test_create_action_missing_capability_raises(self):
        with self.assertRaises(ActionError):
            create_action(name="x", capability="")

    def test_validate_action_normalizes(self):
        action = {"id": "a1", "name": "x", "capability": "read", "args": {}, "status": "pending"}
        validated = validate_action(action)
        self.assertEqual(validated["status"], ActionStatus.PENDING.value)
        self.assertEqual(validated["risk"], ActionRisk.LOW.value)
        self.assertEqual(validated["risk"], ActionRisk.LOW.value)

    def test_validate_action_invalid_status_raises(self):
        action = {"id": "a1", "name": "x", "capability": "read", "args": {}, "status": "invalid"}
        with self.assertRaises(ActionError):
            validate_action(action)


# ---------------------------------------------------------------------------
# ActionAuthorizer tests
# ---------------------------------------------------------------------------
class TestActionAuthorizer(unittest.TestCase):
    def test_authorizes_safe_read_action(self):
        guard = PermissionGuard()
        layer = SecurityLayer(permission_guard=guard)
        authorizer = ActionAuthorizer(permission_guard=guard, security_layer=layer)
        action = create_action(name="read", capability="read", approved=True)
        result = authorizer.authorize(action)
        self.assertTrue(result["ok"])

    def test_denies_requires_approval_when_not_approved(self):
        guard = PermissionGuard()
        authorizer = ActionAuthorizer(permission_guard=guard)
        action = create_action(name="write", capability="filesystem", requires_approval=True, approved=False)
        result = authorizer.authorize(action)
        self.assertFalse(result["ok"])
        self.assertIn("requires_explicit_user_approval", result["reason"])

    def test_denies_dangerous_capability_self_authorization(self):
        authorizer = ActionAuthorizer()
        action = create_action(name="override", capability="system_override", approved=True)
        result = authorizer.authorize(action)
        self.assertFalse(result["ok"])
        self.assertIn("capability_not_self_authorizable", result["reason"])

    def test_denies_dangerous_capability_even_with_approval(self):
        authorizer = ActionAuthorizer()
        action = create_action(name="override", capability="dangerous", approved=True)
        result = authorizer.authorize(action)
        self.assertFalse(result["ok"])


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------
class TestStructuredObservation(unittest.TestCase):
    def test_legacy_record_still_works(self):
        fb = ObservationFeedback()
        entry = fb.record("action1", "result1")
        self.assertEqual(entry["action"], "action1")
        self.assertEqual(entry["result"], "result1")

    def test_record_structured_accepts_triplet(self):
        fb = ObservationFeedback()
        action = create_action(name="test", capability="read")
        obs = {
            "source": "runtime",
            "timestamp": time.time(),
            "raw": {"data": 1},
            "evidence": {"source": "test"},
        }
        verif = {"status": "verified", "method": "test"}
        entry = fb.record_structured(action, obs, verif)
        self.assertTrue(entry["structured"])
        self.assertEqual(entry["action"]["name"], "test")
        self.assertEqual(entry["verification"]["status"], "verified")

    def test_record_structured_updates_metrics(self):
        fb = ObservationFeedback()
        action = create_action(name="test", capability="read")
        obs = {"source": "runtime", "timestamp": time.time(), "raw": {}}
        verif_verified = {"status": "verified"}
        verif_failed = {"status": "failed"}
        verif_unknown = {"status": "unknown"}
        fb.record_structured(action, obs, verif_verified)
        fb.record_structured(action, obs, verif_failed)
        fb.record_structured(action, obs, verif_unknown)
        metrics = fb.metrics()
        self.assertEqual(metrics["total_actions"], 3)
        self.assertEqual(metrics["successes"], 1)
        self.assertEqual(metrics["failures"], 1)
        self.assertEqual(metrics["unexpected"], 1)

    def test_record_structured_requires_dicts(self):
        fb = ObservationFeedback()
        with self.assertRaises(TypeError):
            fb.record_structured("not_a_dict", {}, {})
        with self.assertRaises(TypeError):
            fb.record_structured({}, "not_a_dict", {})
        with self.assertRaises(TypeError):
            fb.record_structured({}, {}, "not_a_dict")


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------
class TestVerificationLayer(unittest.TestCase):
    def test_default_verifier_accepts_explicit_verified(self):
        verifier = DefaultVerifier()
        action = create_action(name="x", capability="read")
        result = {"status": "ok", "verified": True}
        vr = verifier.verify(action, result, None)
        self.assertEqual(vr.status, VerificationStatus.VERIFIED)

    def test_default_verifier_rejects_unverified(self):
        verifier = DefaultVerifier()
        action = create_action(name="x", capability="read")
        result = {"status": "ok"}
        vr = verifier.verify(action, result, None)
        self.assertEqual(vr.status, VerificationStatus.UNVERIFIED)

    def test_default_verifier_marks_failed_when_explicit_false(self):
        verifier = DefaultVerifier()
        action = create_action(name="x", capability="read")
        result = {"status": "error", "verified": False}
        vr = verifier.verify(action, result, None)
        self.assertEqual(vr.status, VerificationStatus.FAILED)

    def test_side_effect_verifier_detects_filesystem_success(self):
        verifier = SideEffectVerifier()
        action = create_action(name="write", capability="filesystem")
        result = {"status": "ok", "output": "file creato"}
        vr = verifier.verify(action, result, None)
        self.assertEqual(vr.status, VerificationStatus.VERIFIED)

    def test_side_effect_verifier_detects_filesystem_failure(self):
        verifier = SideEffectVerifier()
        action = create_action(name="write", capability="filesystem")
        result = {"status": "errore", "error": "negato"}
        vr = verifier.verify(action, result, None)
        self.assertEqual(vr.status, VerificationStatus.FAILED)

    def test_side_effect_verifier_unknown_when_no_evidence(self):
        verifier = SideEffectVerifier()
        action = create_action(name="unknown", capability="unknown")
        result = {"status": "done"}
        vr = verifier.verify(action, result, None)
        self.assertEqual(vr.status, VerificationStatus.UNKNOWN)

    def test_verification_result_roundtrip(self):
        vr = VerificationResult(
            status=VerificationStatus.VERIFIED,
            method="test",
            reason="ok",
            evidence={"key": "value"},
            action_id="act-123",
        )
        data = vr.to_dict()
        vr2 = VerificationResult.from_dict(data)
        self.assertEqual(vr2.status, VerificationStatus.VERIFIED)
        self.assertEqual(vr2.action_id, "act-123")
        self.assertEqual(vr2.evidence, {"key": "value"})


# ---------------------------------------------------------------------------
# AgentRuntime AOV integration tests
# ---------------------------------------------------------------------------
class TestAgentRuntimeAOV(unittest.TestCase):
    def test_runtime_uses_structured_action_and_verification(self):
        fb = ObservationFeedback()
        verifier = DefaultVerifier()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"status": "ok", "verified": True, "response": "done"}]),
            audit=_FakeAudit(),
            observation_feedback=fb,
            verifier=verifier,
            max_iterations=5,
        )
        tid = rt.create_task("aov-test")
        executed = {"count": 0}

        def producer(state, obs):
            if executed["count"] >= 1:
                return None
            executed["count"] += 1
            return create_action(name="step", capability="read", approved=True)

        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.COMPLETED.value)
        self.assertEqual(result.termination_reason, "no_further_action")
        entries = fb.all()
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["structured"])
        self.assertEqual(entries[0]["verification"]["status"], "verified")

    def test_runtime_records_failed_verification(self):
        fb = ObservationFeedback()
        verifier = DefaultVerifier()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"status": "error", "verified": False}] * 2),
            audit=_FakeAudit(),
            observation_feedback=fb,
            verifier=verifier,
            max_iterations=5,
            recovery_policy=RecoveryPolicy(max_attempts=1),
        )
        tid = rt.create_task("aov-fail")

        def producer(state, obs):
            if len(state.actions) >= 1:
                return None
            return create_action(name="step", capability="read", approved=True)

        result = rt.run(tid, action_producer=producer)
        self.assertEqual(result.status, TaskStatus.FAILED.value)
        self.assertEqual(result.termination_reason, "max_recovery_attempts_exceeded (1)")
        entries = fb.all()
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["verification"]["status"], "failed")
        self.assertEqual(entries[0]["verification"]["status"], "failed")

    def test_runtime_structured_observation_contains_evidence(self):
        fb = ObservationFeedback()
        verifier = DefaultVerifier()
        rt = AgentRuntime(
            action_executor=_FakeActionExecutor([{"status": "ok", "verified": True}]),
            audit=_FakeAudit(),
            observation_feedback=fb,
            verifier=verifier,
            max_iterations=5,
        )
        tid = rt.create_task("evidence-test")
        memory = _FakeMemory(hits=[{"payload": {"text": "previous context"}}])
        rt.memory = memory
        executed = {"count": 0}

        def producer(state, obs):
            if executed["count"] >= 1:
                return None
            executed["count"] += 1
            return create_action(name="step", capability="read", approved=True)

        rt.run(tid, action_producer=producer)
        entries = fb.all()
        self.assertEqual(len(entries), 1)
        obs = entries[0]["observation"]
        self.assertIn("evidence", obs)
        self.assertIn("provenance", obs)
        self.assertIn("timestamp", obs)


# ---------------------------------------------------------------------------
# ExecutiveRouter + AgentRuntime integration tests
# ---------------------------------------------------------------------------
class TestExecutiveRouterAgentRuntime(unittest.TestCase):
    def test_router_uses_agent_runtime_when_configured(self):
        from executive_system.router import ExecutiveRouter

        class FakeCoordinator:
            def process(self, goal, session_id=None):
                return {"response": f"answer:{goal}", "source": "test"}

        rt = AgentRuntime(
            cognitive_coordinator=FakeCoordinator(),
            audit=_FakeAudit(),
        )
        router = ExecutiveRouter(agent_runtime=rt, coordinator=FakeCoordinator())
        result = router.generate("ciao")
        self.assertIn("agent_runtime", result)
        self.assertEqual(result["agent_runtime"]["status"], "completed")

    def test_router_falls_back_when_agent_runtime_fails(self):
        from executive_system.router import ExecutiveRouter

        class BrokenRuntime:
            def create_task(self, goal, **kwargs):
                raise RuntimeError("broken")

        router = ExecutiveRouter(agent_runtime=BrokenRuntime())
        result = router.generate("test")
        self.assertNotIn("agent_runtime", result)
        self.assertIn("response", result)


# ---------------------------------------------------------------------------
# Security fix tests
# ---------------------------------------------------------------------------
class TestSecurityFixes(unittest.TestCase):
    def test_tool_system_uses_subprocess_not_popen(self):
        ts = ToolSystem(security_layer=SecurityLayer())
        runner = ts._subprocess_runner
        self.assertEqual(runner.__name__, "_subprocess_runner")
        output = runner("echo test")
        self.assertIn("test", output)

    def test_terminal_tool_rejects_popen_runner(self):
        from executive_system.tools.terminal_tool import TerminalTool
        tool = TerminalTool(workdir="/tmp/jarvis_phase2_security_test")
        with self.assertRaises(PermissionError):
            tool.run("echo test", os.popen, approved=True)

    def test_coordinator_respects_auto_approve_false(self):
        from cognitive_core.coordinator import CognitiveCoordinator
        coord = CognitiveCoordinator(auto_approve_tools=False)
        self.assertFalse(coord.auto_approve_tools)

    def test_coordinator_default_auto_approve_is_true_for_backward_compat(self):
        from cognitive_core.coordinator import CognitiveCoordinator
        coord = CognitiveCoordinator()
        self.assertTrue(coord.auto_approve_tools)


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------
class _FakeMemory:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.calls = 0

    def search_semantic(self, query):
        self.calls += 1
        return self._hits


if __name__ == "__main__":
    unittest.main()
