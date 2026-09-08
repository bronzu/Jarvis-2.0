"""
Device Control must remain DISABLED BY DESIGN.

These tests verify that:
1. Vision/system control is disabled in config
2. Screenshot capability is disabled
3. No autonomous device interaction is possible
"""

import pytest
from pathlib import Path


def test_vision_config_is_disabled():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "vision:" in content
    assert "enabled: false" in content, "Vision/device control must be disabled in config"


def test_screenshot_config_is_disabled():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()
    assert "screenshot_enabled: false" in config_content, "Screenshot must be disabled in config"


def test_screenshot_raises_on_non_macos():
    from cognitive_core.vision.screenshot import capture_screen
    import sys
    if sys.platform != "darwin":
        with pytest.raises((NotImplementedError, RuntimeError)):
            capture_screen()


def test_vision_system_respects_disabled_config():
    from cognitive_core.vision.vision_system import VisionSystem
    from INFRASTRUCTURE.security.permission_guard import PermissionGuard

    security = PermissionGuard()
    system = VisionSystem(config={"enabled": False})

    result = system.analyze_image("/nonexistent.png")
    assert result["success"] is False
    assert result["available"] is False


def test_screenshot_blocked_when_disabled():
    from cognitive_core.vision.vision_system import VisionSystem
    from INFRASTRUCTURE.security.permission_guard import PermissionGuard

    security = PermissionGuard()
    system = VisionSystem(config={"enabled": True, "screenshot_enabled": False})

    with pytest.raises(PermissionError):
        system.capture_screen()
