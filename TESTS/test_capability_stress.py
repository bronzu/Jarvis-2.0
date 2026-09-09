"""
Capability stress tests for JARVIS 2.0 architecture.

Verifies four pillars:
1. Tool Factory / autonomous construction
2. Permission Guard / autonomous authorization
3. Sandbox + rollback / controlled self-improvement
4. Multi-device adaptability / context switching
"""
import hashlib
import os
import tempfile

import numpy as np
import pytest

from neural_core.os_sandbox import run_isolated, OSIsolationLevel
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from expansion_system.tool_factory.factory import ToolFactory


# ---------------------------------------------------------------------------
# 1. Tool Factory / Autonomia Costruttiva
# ---------------------------------------------------------------------------

class TestToolFactoryAutonomousConstruction:
    def test_generates_sha256_tool_when_missing(self):
        factory = ToolFactory()
        spec = {
            "name": "sha256_tool",
            "description": "Calcola SHA-256 di una stringa",
            "methods": [
                {
                    "name": "compute",
                    "params": ["text: str"],
                    "implementation": "import hashlib as h\nreturn h.sha256(text.encode()).hexdigest()",
                }
            ],
        }
        code = factory.generate(spec)
        validation = factory.validate_generated(code)
        assert validation["valid"] is True, validation.get("error")

        candidate = factory.create_candidate("sha256_tool", code)
        assert candidate["state"] == "sandbox"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tool_path = f.name

        try:
            with open(tool_path, "r") as f:
                tool_code = f.read()
            result = run_isolated(
                tool_code + "\nprint(Sha256ToolTool().compute('ciao'))",
                level=OSIsolationLevel.STRICT,
                timeout_seconds=5.0,
            )
            assert result["status"] == "ok", result
            assert hashlib.sha256(b"ciao").hexdigest() in result["output"]
        finally:
            os.unlink(tool_path)

    def test_generated_tool_cannot_bypass_security_layer(self):
        factory = ToolFactory()
        malicious_spec = {
            "name": "evil_tool",
            "description": "Tool malevolo",
            "methods": [
                {
                    "name": "run",
                    "params": [],
                    "implementation": "import os\nos.system('echo pwned')",
                }
            ],
        }
        code = factory.generate(malicious_spec)
        validation = factory.validate_generated(code)
        assert validation["valid"] is False
        assert "Pattern pericoloso" in validation["error"]


# ---------------------------------------------------------------------------
# 2. Permission Guard / Autonomia Autorizzativa
# ---------------------------------------------------------------------------

class TestPermissionGuardAutonomousAuthorization:
    def test_blocks_high_risk_command_without_approval(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("rm -rf /", approved=False)

    def test_blocks_self_permission_grant(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.grant_self_permission()

    def test_requires_explicit_approval(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.require_user_approval(False)
        assert guard.require_user_approval(True) is True

    def test_blocks_destructive_filesystem_operations(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("os.remove('/etc/passwd')", approved=False)
        with pytest.raises(PermissionError):
            guard.validate_command("shutil.rmtree('/important')", approved=False)


# ---------------------------------------------------------------------------
# 3. Sandbox + Rollback / Self-improvement Controllato
# ---------------------------------------------------------------------------

class TestSandboxRollbackControlledSelfImprovement:
    def test_candidate_failure_is_isolated_and_stable_version_preserved(self):
        stable_code = "def compute(x):\n    return x * 2\n"
        candidate_code = "def compute(x):\n    return x / 0\n"

        stable_result = run_isolated(
            stable_code + "print(compute(4))",
            level=OSIsolationLevel.STRICT,
            timeout_seconds=5.0,
        )
        assert stable_result["status"] == "ok", stable_result
        assert "8" in stable_result["output"]

        candidate_result = run_isolated(
            candidate_code + "print(compute(4))",
            level=OSIsolationLevel.STRICT,
            timeout_seconds=5.0,
        )
        assert candidate_result["status"] in ("blocked", "error"), candidate_result
        assert candidate_result["returncode"] != 0 or "__SANDBOX_ERROR__" in candidate_result["output"]

        stable_result_after = run_isolated(
            stable_code + "print(compute(4))",
            level=OSIsolationLevel.STRICT,
            timeout_seconds=5.0,
        )
        assert stable_result_after["status"] == "ok", stable_result_after
        assert "8" in stable_result_after["output"]


# ---------------------------------------------------------------------------
# 4. Multi-device Adaptability / Context Switching
# ---------------------------------------------------------------------------

class TestMultiDeviceAdaptability:
    def test_desktop_to_wearable_preserves_semantic_information(self):
        desktop_payload = {
            "task": "analisi_sentiment",
            "state": "completed",
            "results": [
                {"text": "Prodotto eccellente", "score": 0.95, "label": "positive"},
                {"text": "Pessimo servizio", "score": 0.88, "label": "negative"},
            ],
            "metadata": {"model_version": "2.1", "timestamp": "2026-09-08T20:00:00Z"},
            "tool_status": ["tokenizer_ok", "model_ok", "inference_ok"],
            "timestamp": "2026-09-08T20:00:00Z",
            "secondary_info": {"debug_log": "...", "raw_probabilities": [0.1, 0.9]},
        }

        def adapt_for_wearable(payload):
            return {
                "task": payload["task"],
                "state": payload["state"],
                "summary": f"{len(payload['results'])} risultati, "
                           f"{sum(1 for r in payload['results'] if r['label']=='positive')} positivi",
                "critical": payload["results"][0] if payload["results"] else None,
                "ts": payload["timestamp"],
            }

        wearable = adapt_for_wearable(desktop_payload)

        assert wearable["task"] == desktop_payload["task"]
        assert wearable["state"] == desktop_payload["state"]
        assert "risultati" in wearable["summary"]
        assert wearable["critical"]["label"] == "positive"
        assert wearable["ts"] == desktop_payload["timestamp"]
        assert "debug_log" not in wearable
        assert "raw_probabilities" not in wearable
