import json
import os
import tempfile

import pytest

from INFRASTRUCTURE.audit.audit_system import AuditSystem


def test_audit_system_creates_session():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    assert audit.get_session_id() is not None
    events = audit.get_events()
    assert any(e["event_type"] == "SESSION_START" for e in events)
    audit.close()


def test_audit_log_event_has_required_fields():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    audit.log_event(
        "USER_INPUT",
        component="Test",
        action="input",
        input_summary="hello world",
    )
    events = audit.get_events("USER_INPUT")
    assert len(events) == 1
    event = events[0]
    assert "timestamp" in event
    assert "session_id" in event
    assert "event_id" in event
    assert event["event_type"] == "USER_INPUT"
    assert event["component"] == "Test"
    assert event["action"] == "input"
    assert event["status"] == "success"
    assert event["input_summary"] == "hello world"
    audit.close()


def test_audit_redacts_password():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    audit.log_event(
        "USER_INPUT",
        component="Test",
        action="input",
        input_summary="password=secret123",
    )
    events = audit.get_events("USER_INPUT")
    assert "secret123" not in events[-1]["input_summary"]
    audit.close()


def test_audit_redacts_api_key():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    audit.log_event(
        "USER_INPUT",
        component="Test",
        action="input",
        input_summary="api_key=ABCDEF",
    )
    events = audit.get_events("USER_INPUT")
    assert "ABCDEF" not in events[-1]["input_summary"]
    audit.close()


def test_audit_redacts_token():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    audit.log_event(
        "USER_INPUT",
        component="Test",
        action="input",
        input_summary="token=xyz",
    )
    events = audit.get_events("USER_INPUT")
    assert "xyz" not in events[-1]["input_summary"]
    audit.close()


def test_audit_persists_jsonl():
    log_dir = tempfile.mkdtemp()
    audit = AuditSystem(enabled=True, log_directory=log_dir)
    audit.log_event("TOOL_EXECUTION_END", component="FilesystemTool", action="read", status="success")
    audit.close()
    files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    assert len(files) >= 1
    path = os.path.join(log_dir, files[0])
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 1
    event = json.loads(lines[0])
    assert event["event_type"] == "SESSION_START"
    event = json.loads(lines[-1])
    assert event["event_type"] == "SESSION_END"


def test_audit_fault_tolerant():
    audit = AuditSystem(enabled=True, log_directory="/nonexistent/dir/that/cannot/be/created/12345")
    audit.log_event("ERROR", component="Test", action="test", status="error")
    audit.close()
    assert audit.session_id is not None
    assert len(audit.session_id) > 0


def test_audit_rotation():
    log_dir = tempfile.mkdtemp()
    audit = AuditSystem(enabled=True, log_directory=log_dir, max_file_size=100, backup_count=3)
    for _ in range(200):
        audit.log_event("WARNING", component="Test", action="test", input_summary="x" * 50)
    audit.close()
    files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    assert len(files) <= 4


def test_audit_event_types():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    audit.log_event("TOOL_REQUEST", component="ToolSystem", action="execute")
    audit.log_event("TOOL_EXECUTION_START", component="FilesystemTool", action="read")
    audit.log_event("TOOL_EXECUTION_END", component="FilesystemTool", action="read", status="success")
    audit.log_event("PERMISSION_DENIED", component="SecurityLayer", action="check", status="denied")
    audit.log_event("VISION_ANALYSIS_START", component="VisionSystem", action="analyze")
    audit.log_event("VISION_ERROR", component="VisionSystem", action="analyze", status="error", error="fail")
    events = audit.get_events()
    types = {e["event_type"] for e in events}
    assert "TOOL_REQUEST" in types
    assert "TOOL_EXECUTION_START" in types
    assert "TOOL_EXECUTION_END" in types
    assert "PERMISSION_DENIED" in types
    assert "VISION_ANALYSIS_START" in types
    assert "VISION_ERROR" in types
    audit.close()


def test_audit_ignores_unknown_event_type():
    audit = AuditSystem(enabled=True, log_directory=tempfile.gettempdir())
    audit.log_event("UNKNOWN_EVENT", component="Test", action="test")
    events = audit.get_events("UNKNOWN_EVENT")
    assert len(events) == 0
    audit.close()
