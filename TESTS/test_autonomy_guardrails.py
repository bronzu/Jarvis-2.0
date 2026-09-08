import pytest
from executive_system.autonomy_manager import AutonomyManager
from INFRASTRUCTURE.security.permission_guard import PermissionGuard


def test_autonomy_cannot_self_authorize_or_act_without_approval():
    guard = PermissionGuard()
    with pytest.raises(PermissionError):
        guard.grant_self_permission()
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(lambda: "done")
    assert AutonomyManager(guard).act(lambda: "done", approved=True) == "done"


def test_autonomy_rejects_non_callable_action():
    guard = PermissionGuard()
    with pytest.raises((TypeError, PermissionError, ValueError)):
        AutonomyManager(guard).act("not_callable", approved=True)
