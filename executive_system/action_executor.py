from typing import Any, Dict, Optional

from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.audit.audit_system import AuditSystem


class ActionExecutor:
    """Esegue azioni approvate con integrazione security e audit."""

    def __init__(self, permission_guard=None, audit_system=None):
        self.permission_guard = permission_guard or PermissionGuard()
        self.audit_system = audit_system or AuditSystem(enabled=True)

    def execute(self, action, approved=False, session_id=None, metadata=None):
        self.permission_guard.require_user_approval(approved)
        if not callable(action):
            raise TypeError("L'azione deve essere richiamabile")
        try:
            self.audit_system.log_event(
                "TOOL_EXECUTION_START",
                component="ActionExecutor",
                action="execute",
                status="started",
                metadata=metadata or {},
                session_id=session_id,
            )
            result = action()
            self.audit_system.log_event(
                "TOOL_EXECUTION_END",
                component="ActionExecutor",
                action="execute",
                status="success",
                output_summary=self._safe_summary(result),
                session_id=session_id,
            )
            return result
        except Exception as e:
            self.audit_system.log_event(
                "TOOL_EXECUTION_ERROR",
                component="ActionExecutor",
                action="execute",
                status="error",
                error=str(e),
                metadata=metadata or {},
                session_id=session_id,
            )
            raise

    def execute_tool(self, tool_system, capability, approved=False, session_id=None, *args, **kwargs):
        self.permission_guard.require_user_approval(approved)
        if tool_system is None:
            raise ValueError("tool_system è obbligatorio")
        metadata = {"capability": capability, "args": str(args)[:256], "kwargs": str(kwargs)[:256]}
        return self.execute(
            lambda: tool_system.execute(capability, *args, approved=approved, session_id=session_id, **kwargs),
            approved=approved,
            session_id=session_id,
            metadata=metadata,
        )

    def _safe_summary(self, result):
        try:
            text = str(result)
            return text[:1024]
        except Exception:
            return "<non-serializable>"
