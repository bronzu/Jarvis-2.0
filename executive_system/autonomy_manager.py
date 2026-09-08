from INFRASTRUCTURE.security.permission_guard import PermissionGuard


class AutonomyManager:
    def __init__(self, permission_guard=None): self.permission_guard = permission_guard or PermissionGuard()
    def act(self, action, approved=False):
        self.permission_guard.require_user_approval(approved)
        if not callable(action): raise TypeError("L'azione deve essere richiamabile")
        return action()
