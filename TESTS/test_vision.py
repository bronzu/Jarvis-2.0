import os
import tempfile
from pathlib import Path

import pytest

from cognitive_core.vision.vision_system import VisionSystem
from cognitive_core.vision.interfaces import load_image, MockVisionBackend
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from executive_system.tools.tool_system import ToolSystem


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
    b"\x1c $.\x27 ',#\x1c\x1c(7),01444\x1f\x1f9=55=[]B\x00\x08\x07\x08\x07"
    b"\n\x08\n\t\x0b\t\n\x0c\r\x0c\x0c\x0c\x0c\x0c\x0c\x0c\x0c\x0c\x0c\x0c"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    b"\xff\xd9"
)

WEBP_BYTES = (
    b"RIFF\x00\x00\x00\x00WEBP"
    b"\x00\x00\x00\x00VP8L"
    b"\x00\x00\x00\x00\x00\x00\x00\x00"
)


def _write_temp(content, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path


def test_load_png_image():
    path = _write_temp(PNG_BYTES, ".png")
    try:
        image = load_image(path)
        assert image.format == "PNG"
        assert image.width == 1
        assert image.height == 1
        assert image.size_bytes == len(PNG_BYTES)
    finally:
        os.unlink(path)


def test_load_jpeg_image():
    path = _write_temp(JPEG_BYTES, ".jpg")
    try:
        image = load_image(path)
        assert image.format == "JPEG"
        assert image.path == path
    finally:
        os.unlink(path)


def test_load_webp_image():
    path = _write_temp(WEBP_BYTES, ".webp")
    try:
        image = load_image(path)
        assert image.format == "WEBP"
    finally:
        os.unlink(path)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_image("/missing/path/image.png")


def test_load_unsupported_format():
    path = _write_temp(b"this is not an image", ".txt")
    try:
        with pytest.raises(ValueError):
            load_image(path)
    finally:
        os.unlink(path)


def test_load_corrupted_png():
    path = _write_temp(b"this is not a valid png image at all", ".png")
    try:
        with pytest.raises(ValueError):
            load_image(path)
    finally:
        os.unlink(path)


def test_vision_system_disabled():
    system = VisionSystem(config={"enabled": False})
    result = system.analyze_image("/missing.png")
    assert result["success"] is False
    assert result["available"] is False


def test_vision_system_backend_mock():
    system = VisionSystem(config={"enabled": True})
    path = _write_temp(PNG_BYTES, ".png")
    try:
        result = system.analyze_image(path)
        assert result["success"] is True
        assert result["available"] is False
        assert result["image_metadata"]["format"] == "PNG"
    finally:
        os.unlink(path)


def test_vision_screenshot_interface_disabled():
    system = VisionSystem(config={"enabled": True, "screenshot_enabled": False})
    with pytest.raises(PermissionError):
        system.capture_screen()


def test_vision_screenshot_interface_enabled(monkeypatch):
    system = VisionSystem(config={"enabled": True, "screenshot_enabled": True})
    called = []

    def fake_capture(output_path=None):
        called.append(output_path)
        return load_image(_write_temp(PNG_BYTES, ".png"))

    monkeypatch.setattr("cognitive_core.vision.vision_system.capture_screen", fake_capture)
    try:
        image = system.capture_screen()
        assert image is not None
        assert len(called) == 1
    finally:
        for p in called:
            if p is not None:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def test_ocr_interface_disabled():
    system = VisionSystem(config={"enabled": True, "ocr_enabled": False})
    result = system.extract_text("/missing.png")
    assert result["available"] is False


def test_vision_tool_integration():
    security = SecurityLayer(PermissionGuard())
    vision = VisionSystem(config={"enabled": True})
    tool_system = ToolSystem(security_layer=security, vision_system=vision)
    path = _write_temp(PNG_BYTES, ".png")
    try:
        result = tool_system.execute("vision", operation="analyze_image", path=path, approved=True)
        assert result is not None
        assert result["available"] is False
    finally:
        os.unlink(path)


def test_cognitive_core_vision_input():
    vision = VisionSystem(config={"enabled": True})
    coordinator = type("Coordinator", (), {"vision_system": vision, "_log_vision": lambda *a, **k: None})()
    path = _write_temp(PNG_BYTES, ".png")
    try:
        result = vision.analyze_image(path)
        assert "image_metadata" in result
    finally:
        os.unlink(path)


def test_vision_respects_security_layer():
    security = SecurityLayer(PermissionGuard())
    vision = VisionSystem(config={"enabled": True})
    tool_system = ToolSystem(security_layer=security, vision_system=vision)
    with pytest.raises(PermissionError):
        tool_system.execute("vision", operation="analyze_image", path="/etc/passwd", approved=False)
