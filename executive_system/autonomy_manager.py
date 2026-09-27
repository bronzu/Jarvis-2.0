import inspect
import re

from INFRASTRUCTURE.security.permission_guard import PermissionGuard


_DANGEROUS_PATTERNS = [
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\bsubprocess\.(run|call|Popen|check_output|check_call)\b"),
    re.compile(r"\b__import__\b"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bcompile\s*\("),
    re.compile(r"\bpickle\.(load|loads|dumps)\b"),
    re.compile(r"\bmarshal\b"),
    re.compile(r"\bos\.chmod\b"),
    re.compile(r"\bos\.chown\b"),
    re.compile(r"\bos\.unlink\b"),
    re.compile(r"\bos\.remove\b"),
    re.compile(r"\bshutil\.(rmtree|rm|move|copy)\b"),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\bgetattr\b"),
    re.compile(r"\bsetattr\b"),
    re.compile(r"\bdelattr\b"),
    re.compile(r"\b__reduce__\b"),
    re.compile(r"\b__class__\b"),
    re.compile(r"\b__bases__\b"),
    re.compile(r"\b__subclasses__\b"),
    re.compile(r"\b__globals__\b"),
    re.compile(r"\b__builtins__\b"),
    re.compile(r"\bctypes\b"),
    re.compile(r"\bresource\b"),
]


def _validate_action_source(func: callable) -> None:
    """Reject callables whose source code contains dangerous patterns."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        raise PermissionError(
            "Cannot inspect source of action — only source-available callables are permitted"
        )
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(source):
            raise PermissionError(
                f"Azione autonoma bloccata: pattern pericoloso rilevato ({pattern.pattern})"
            )


class AutonomyManager:
    def __init__(self, permission_guard=None):
        self.permission_guard = permission_guard or PermissionGuard()

    def act(self, action, approved=False):
        self.permission_guard.require_user_approval(approved)
        if not callable(action):
            raise TypeError("L'azione deve essere richiamabile")
        _validate_action_source(action)
        return action()
