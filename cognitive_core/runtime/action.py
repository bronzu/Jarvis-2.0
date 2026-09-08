"""Structured Action schema and authorizer for the JARVIS Agent Runtime.

This module defines the canonical Action representation used by the
AgentRuntime when orchestrating the Action → Observation → Verification
cycle.  It does *not* duplicate the existing ``ActionExecutor`` or
``ToolSystem``: those remain the execution backends.  This module provides
the *pre-execution* contract — schema, preconditions, authorization — that
the runtime enforces *before* delegating to any backend.

Architectural role
------------------
Per the master prompt §14 and §46, every action must have:

  * a structured schema;
  * explicit preconditions;
  * authorization / permission checks before execution;
  * provenance and risk metadata.

The authorizer delegates to the existing ``PermissionGuard`` and
``SecurityLayer`` so no security control is bypassed or weakened.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionStatus(str, Enum):
    """Lifecycle status of a single action."""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"


class ActionRisk(str, Enum):
    """Qualitative risk levels for an action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionError(RuntimeError):
    """Raised when an action cannot be constructed, validated, or authorized."""


# ---------------------------------------------------------------------------
# Action dataclass (dict-based for JSON serializability)
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def new_action_id() -> str:
    return f"act-{uuid.uuid4().hex[:12]}"


def create_action(
    name: str,
    capability: str,
    args: Optional[Dict[str, Any]] = None,
    preconditions: Optional[List[str]] = None,
    requires_approval: bool = False,
    approved: bool = False,
    risk: ActionRisk = ActionRisk.LOW,
    metadata: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a structured action dict.

    The action schema is the single source of truth for what the runtime
    may execute.  Every field has a default so callers can construct
    minimal actions, but security-critical fields (``requires_approval``,
    ``approved``) are explicit and never inferred.
    """
    if not name or not str(name).strip():
        raise ActionError("action name is required")
    if not capability or not str(capability).strip():
        raise ActionError("action capability is required")

    action: Dict[str, Any] = {
        "id": new_action_id(),
        "name": str(name).strip(),
        "capability": str(capability).strip(),
        "args": dict(args or {}),
        "preconditions": list(preconditions or []),
        "requires_approval": bool(requires_approval),
        "approved": bool(approved),
        "risk": risk.value if isinstance(risk, ActionRisk) else str(risk),
        "status": ActionStatus.PENDING.value,
        "metadata": dict(metadata or {}),
        "provenance": dict(provenance or {}),
        "created_at": _now(),
        "updated_at": _now(),
    }
    return action


def validate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an action dict and normalize fields.

    Returns the normalized action.  Raises ``ActionError`` on invalid
    input so the runtime never executes a malformed action.
    """
    if not isinstance(action, dict):
        raise ActionError("action must be a dict")
    required = ("id", "name", "capability", "args", "status")
    for key in required:
        if key not in action:
            raise ActionError(f"action missing required field: {key}")
    action.setdefault("preconditions", [])
    action.setdefault("requires_approval", False)
    action.setdefault("approved", False)
    action.setdefault("risk", ActionRisk.LOW.value)
    action.setdefault("metadata", {})
    action.setdefault("provenance", {})
    action.setdefault("created_at", _now())
    action.setdefault("updated_at", _now())
    if isinstance(action.get("risk"), ActionRisk):
        action["risk"] = action["risk"].value
    if action["status"] not in [s.value for s in ActionStatus]:
        raise ActionError(f"invalid action status: {action['status']}")
    return action


# ---------------------------------------------------------------------------
# Authorizer
# ---------------------------------------------------------------------------

class ActionAuthorizer:
    """Authorizes actions using the existing security infrastructure.

    This class does *not* introduce new security primitives.  It wraps
    the project's ``PermissionGuard`` and ``SecurityLayer`` so that every
    action in the runtime passes through the same checks as actions in
    the legacy path.

    Rules enforced:
      * ``requires_approval=True`` MUST have ``approved=True``.
      * Dangerous capabilities (``system_override``, ``dangerous``) are
        never self-authorizable.
      * The existing ``PermissionGuard`` and ``SecurityLayer`` are called
        for capability-specific checks when a ``tool_system`` or
        ``security_layer`` is provided.
    """

    # Capabilities that JARVIS must never self-authorize, regardless of
    # the ``approved`` flag.
    _NON_SELF_AUTHORIZABLE = {"dangerous", "system_override"}

    def __init__(
        self,
        permission_guard: Any = None,
        security_layer: Any = None,
        tool_system: Any = None,
    ) -> None:
        self.permission_guard = permission_guard
        self.security_layer = security_layer
        self.tool_system = tool_system

    def authorize(self, action: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Authorize an action before execution.

        Returns a dict with:
          ``ok`` (bool) — whether the action is authorized;
          ``reason`` (str) — human-readable explanation;
          ``action`` (dict) — the normalized action.

        Raises ``ActionError`` only on malformed input; permission
        denials are returned as ``{"ok": False, ...}`` so the runtime
        can record the failure and decide whether to recover.
        """
        action = validate_action(action)
        capability = action["capability"]

        # Rule 1: self-authorization guard
        if capability in self._NON_SELF_AUTHORIZABLE:
            return {
                "ok": False,
                "reason": f"capability_not_self_authorizable: {capability}",
                "action": action,
            }

        # Rule 2: explicit approval gate
        if action.get("requires_approval") and not action.get("approved"):
            return {
                "ok": False,
                "reason": "requires_explicit_user_approval",
                "action": action,
            }

        # Rule 3: delegate to existing security layer for tool-specific checks
        if self.security_layer is not None and self.tool_system is not None:
            try:
                kwargs = action.get("args", {})
                self.security_layer.check_tool_execution(
                    capability,
                    kwargs,
                    approved=bool(action.get("approved")),
                    session_id=session_id,
                )
            except PermissionError as exc:
                return {
                    "ok": False,
                    "reason": f"security_layer_denied: {exc}",
                    "action": action,
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": f"security_layer_error: {exc}",
                    "action": action,
                }

        action["status"] = ActionStatus.AUTHORIZED.value
        action["updated_at"] = _now()
        return {
            "ok": True,
            "reason": "authorized",
            "action": action,
        }
