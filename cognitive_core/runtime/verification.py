"""Verification layer for the JARVIS Agent Runtime.

This module implements the verification step of the
Action → Observation → Verification cycle.  It distinguishes between
verified, failed, unknown, and unverified outcomes and never declares
success without evidence (master prompt §15, §16, §45).

Architectural role
------------------
The runtime must not treat:

    tool returned success
        ≡
    action succeeded

and must not treat:

    action executed
        ≡
    desired world state achieved

The correct sequence is:

    ACTION
      ↓
    EXECUTION
      ↓
    TOOL RESULT
      ↓
    OBSERVATION
      ↓
    WORLD STATE UPDATE
      ↓
    VERIFICATION
      ↓
    EFFECT CONFIRMED

Only after ``EFFECT CONFIRMED`` may the runtime mark the step as
successful.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class VerificationStatus(str, Enum):
    """Possible outcomes of a verification step."""
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"
    UNVERIFIED = "unverified"


class VerificationError(RuntimeError):
    """Raised when verification input is invalid."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """The outcome of verifying an action's effect on the world.

    Attributes:
        status: One of the ``VerificationStatus`` values.
        method: How verification was performed (e.g. ``"default.explicit_flag"``,
            ``"filesystem.exists"``, ``"side_effect_check"``).
        reason: Human-readable explanation, especially for ``FAILED`` and
            ``UNKNOWN``.
        evidence: Optional structured evidence supporting the verdict.
        action_id: The action being verified.
        verified_at: Timestamp of verification.
    """
    status: VerificationStatus
    method: str = ""
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_id: Optional[str] = None
    verified_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "method": self.method,
            "reason": self.reason,
            "evidence": self.evidence,
            "action_id": self.action_id,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        if not isinstance(data, dict):
            raise VerificationError("VerificationResult payload must be a dict")
        return cls(
            status=VerificationStatus(data.get("status", VerificationStatus.UNVERIFIED.value)),
            method=data.get("method", ""),
            reason=data.get("reason", ""),
            evidence=data.get("evidence", {}),
            action_id=data.get("action_id"),
            verified_at=data.get("verified_at", time.time()),
        )


# ---------------------------------------------------------------------------
# Verifier interface
# ---------------------------------------------------------------------------

Verifier = Callable[[Dict[str, Any], Dict[str, Any], Any], VerificationResult]


# ---------------------------------------------------------------------------
# Default verifier
# ---------------------------------------------------------------------------

class DefaultVerifier:
    """Conservative default verifier.

    A real agent should pass a domain-specific verifier.  The default
    treats an action as verified **only** if the result is explicitly
    marked ``verified=True`` (e.g. by the action_executor after a real
    side-effect check).  Otherwise it returns ``UNVERIFIED`` so the
    runtime does not silently claim success.

    The default verifier never returns ``VERIFIED`` without an explicit
    positive signal from the execution backend.
    """

    def __init__(self, action_executor: Any = None) -> None:
        self.action_executor = action_executor

    def verify(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        state: Any = None,
    ) -> VerificationResult:
        if not isinstance(action, dict):
            raise VerificationError("action must be a dict")
        if not isinstance(result, dict):
            raise VerificationError("result must be a dict")

        action_id = action.get("id")

        # Explicit verified flag from the execution backend.
        if result.get("verified") is True:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                method="default.explicit_flag",
                reason="result_marked_verified",
                action_id=action_id,
                evidence={"result": _safe_evidence(result)},
            )

        # Explicit failure flag.
        if result.get("verified") is False:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                method="default.explicit_flag",
                reason=result.get("reason", "result_marked_unverified"),
                action_id=action_id,
                evidence={"result": _safe_evidence(result)},
            )

        # No explicit flag: unverified by default.
        return VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            method="default.missing_flag",
            reason="result_does_not_contain_verified_flag",
            action_id=action_id,
            evidence={"result": _safe_evidence(result)},
        )


# ---------------------------------------------------------------------------
# Side-effect verifier
# ---------------------------------------------------------------------------

class SideEffectVerifier:
    """Verifier that checks real side-effects when possible.

    This verifier inspects the action result for evidence of actual
    world modification (file creation, command output, etc.).  It is
    intentionally conservative: absence of evidence is ``UNKNOWN``, not
    ``FAILED``.
    """

    def verify(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        state: Any = None,
    ) -> VerificationResult:
        if not isinstance(action, dict):
            raise VerificationError("action must be a dict")
        if not isinstance(result, dict):
            raise VerificationError("result must be a dict")

        action_id = action.get("id")
        capability = action.get("capability", "")
        evidence: Dict[str, Any] = {"result": _safe_evidence(result)}

        # If the backend already marked verified/failed, respect it.
        if result.get("verified") is True:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                method="side_effect.explicit_flag",
                reason="backend_confirmed_side_effect",
                action_id=action_id,
                evidence=evidence,
            )
        if result.get("verified") is False:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                method="side_effect.explicit_flag",
                reason=result.get("reason", "backend_denied_side_effect"),
                action_id=action_id,
                evidence=evidence,
            )

        # Heuristic checks for common side-effect evidence.
        status = self._heuristic_status(capability, result, evidence)
        return VerificationResult(
            status=status,
            method="side_effect.heuristic",
            reason=self._reason_for(status),
            action_id=action_id,
            evidence=evidence,
        )

    @staticmethod
    def _heuristic_status(capability: str, result: Dict[str, Any], evidence: Dict[str, Any]) -> VerificationStatus:
        text = str(result)
        lower = text.lower()
        if capability == "filesystem":
            if any(k in lower for k in ["errore", "error", "negato", "denied", "failure"]):
                return VerificationStatus.FAILED
            if any(k in lower for k in ["creato", "written", "saved", "ok", "success"]):
                return VerificationStatus.VERIFIED
            return VerificationStatus.UNKNOWN
        if capability == "terminal":
            if any(k in lower for k in ["permission denied", "not allowed", "dangerous"]):
                return VerificationStatus.FAILED
            if result.get("returncode") == 0 or "exit 0" in lower or "ok" in lower:
                return VerificationStatus.VERIFIED
            if result.get("returncode") not in (None, 0):
                return VerificationStatus.FAILED
            return VerificationStatus.UNKNOWN
        if capability == "python":
            if any(k in lower for k in ["error", "traceback", "exception"]):
                return VerificationStatus.FAILED
            if "output" in result or "result" in result:
                return VerificationStatus.VERIFIED
            return VerificationStatus.UNKNOWN
        # Unknown capability: cannot heuristically verify.
        return VerificationStatus.UNKNOWN

    @staticmethod
    def _reason_for(status: VerificationStatus) -> str:
        return {
            VerificationStatus.VERIFIED: "side_effect_evidence_present",
            VerificationStatus.FAILED: "side_effect_evidence_absent_or_error",
            VerificationStatus.UNKNOWN: "side_effect_evidence_inconclusive",
            VerificationStatus.UNVERIFIED: "no_verification_performed",
        }.get(status, "unknown")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_evidence(result: Dict[str, Any], max_len: int = 1024) -> Dict[str, Any]:
    """Bounded, safe extraction of evidence from a result dict."""
    try:
        text = str(result)
    except Exception:
        return {"error": "non_serializable_result"}
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return {"summary": text}
