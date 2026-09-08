import re
from INFRASTRUCTURE.audit.audit_system import AuditSystem
from INFRASTRUCTURE.security.permission_guard import PermissionGuard


class SecurityLayer:
    def __init__(self, permission_guard=None, audit_system=None):
        self.permission_guard = permission_guard or PermissionGuard()
        self.audit_system = audit_system or AuditSystem(enabled=True)

    def check_tool_execution(self, tool_name, parameters, approved=False, session_id=None):
        self.permission_guard.require_user_approval(approved)
        self.audit_system.log_event(
            "TOOL_REQUEST",
            component="SecurityLayer",
            action="check_tool_execution",
            input_summary=f"tool={tool_name}",
            metadata={"parameters": parameters},
            session_id=session_id,
        )
        return True

    def check_filesystem_access(self, path, operation, approved=False, session_id=None):
        self.permission_guard.require_user_approval(approved)
        self.audit_system.log_event(
            f"FILESYSTEM_{operation.upper()}",
            component="SecurityLayer",
            action=f"check_filesystem_access",
            input_summary=f"path={path}",
            session_id=session_id,
        )
        return True

    def check_code_execution(self, code, approved=False, session_id=None):
        self.permission_guard.require_user_approval(approved)
        for pattern in self.permission_guard.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise PermissionError("Codice pericoloso bloccato dal security layer.")
        self.audit_system.log_event(
            "PYTHON_EXECUTION",
            component="SecurityLayer",
            action="check_code_execution",
            input_summary=f"code_length={len(code)}",
            session_id=session_id,
        )
        return True

    def check_network_access(self, url, approved=False, session_id=None):
        self.permission_guard.require_user_approval(approved)
        self.audit_system.log_event(
            "INTERNET_REQUEST",
            component="SecurityLayer",
            action="check_network_access",
            input_summary=f"url={url}",
            session_id=session_id,
        )
        return True

    def log_permission_event(self, event_type, component, action, status, session_id=None, **kwargs):
        self.audit_system.log_event(
            event_type,
            component=component,
            action=action,
            status=status,
            metadata=kwargs,
            session_id=session_id,
        )

    def get_audit_log(self):
        return self.audit_system.get_events()
