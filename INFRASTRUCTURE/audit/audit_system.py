import datetime
import sys
import json
import os
import re
import threading
import uuid
from pathlib import Path


class AuditSystem:
    SENSITIVE_PATTERNS = [
        (re.compile(r"(?i)(password|passwd|pass)\s*[:=]\s*\S+"), r"\1=***"),
        (re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+"), r"\1=***"),
        (re.compile(r"(?i)(token|secret)\s*[:=]\s*\S+"), r"\1=***"),
        (re.compile(r"(?i)(authorization|bearer)\s+\S+"), r"\1=***"),
        (re.compile(r"(?i)(cookie)\s*[:=]\s*\S+"), r"\1=***"),
        (re.compile(r"(?i)(credential)\s*[:=]\s*\S+"), r"\1=***"),
    ]

    EVENT_TYPES = {
        "SESSION_START",
        "SESSION_END",
        "USER_INPUT",
        "MODEL_RESPONSE",
        "TOOL_REQUEST",
        "TOOL_EXECUTION_START",
        "TOOL_EXECUTION_END",
        "TOOL_EXECUTION_ERROR",
        "PERMISSION_REQUEST",
        "PERMISSION_GRANTED",
        "PERMISSION_DENIED",
        "FILESYSTEM_READ",
        "FILESYSTEM_WRITE",
        "TERMINAL_COMMAND",
        "PYTHON_EXECUTION",
        "WEB_REQUEST",
        "INTERNET_REQUEST",
        "MEMORY_READ",
        "MEMORY_WRITE",
        "REASONING_START",
        "REASONING_END",
        "VISION_INPUT",
        "VISION_ANALYSIS_START",
        "VISION_ANALYSIS_END",
        "VISION_ERROR",
        "ERROR",
        "WARNING",
        "SELF_IMPROVEMENT_CYCLE_START",
        "SELF_IMPROVEMENT_CYCLE_END",
        "SELF_IMPROVEMENT_CYCLE_ERROR",
        "SELF_IMPROVEMENT_TRAINING_START",
        "SELF_IMPROVEMENT_TRAINING_END",
        "SELF_IMPROVEMENT_TRAINING_ERROR",
        "SELF_IMPROVEMENT_ROLLBACK",
        "SELF_IMPROVEMENT_DEPLOYMENT",
        # Agent Runtime events (Phase 1)
        "AGENT_TASK_CREATED",
        "AGENT_TASK_FINISHED",
        "AGENT_CHECKPOINT",
        "AGENT_RESUMED",
        "AGENT_PERMISSION_DENIED",
        "AGENT_ACTION_PROPOSED",
        "AGENT_ACTION_EXECUTED",
        "AGENT_EXECUTION_ERROR",
        "AGENT_VERIFICATION_FAILED",
        "AGENT_BUDGET_EXHAUSTED",
        "AGENT_DEADLINE_EXCEEDED",
    }

    def __init__(
        self,
        enabled=True,
        log_directory="./audit_logs",
        log_level="INFO",
        max_file_size=10 * 1024 * 1024,
        backup_count=5,
    ):
        self.enabled = enabled
        self.log_directory = Path(log_directory)
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.session_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._current_file = None
        self._current_size = 0
        self._event_counter = 0
        self._events = []

        if self.enabled:
            try:
                self.log_directory.mkdir(parents=True, exist_ok=True)
                self._open_new_file()
                self.log_event(
                    "SESSION_START",
                    component="AuditSystem",
                    action="initialize",
                    status="success",
                    metadata={"session_id": self.session_id},
                )
            except Exception:
                self.enabled = False

    def _open_new_file(self):
        if self._current_file is not None:
            try:
                self._current_file.close()
            except Exception:
                pass
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file_path = self.log_directory / f"audit_{timestamp}_{self.session_id[:8]}.jsonl"
        self._current_file = open(self._current_file_path, "a", encoding="utf-8")
        self._current_size = 0

    def _rotate_if_needed(self):
        if self._current_file is None:
            return
        try:
            self._current_file.flush()
            self._current_size = self._current_file_path.stat().st_size
        except Exception:
            return
        if self._current_size >= self.max_file_size:
            self._rotate()

    def _rotate(self):
        if self._current_file is not None:
            try:
                self._current_file.close()
            except Exception:
                pass
        base = self._current_file_path
        for i in range(self.backup_count - 1, 0, -1):
            src = base.with_name(base.name + f".{i}")
            dst = base.with_name(base.name + f".{i + 1}")
            if src.exists():
                try:
                    src.rename(dst)
                except Exception:
                    pass
        if base.exists():
            try:
                base.rename(base.with_name(base.name + ".1"))
            except Exception:
                pass
        self._open_new_file()

    def redact(self, data):
        if data is None:
            return None
        text = str(data)
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def log_event(
        self,
        event_type,
        component,
        action,
        status="success",
        input_summary=None,
        output_summary=None,
        error=None,
        metadata=None,
        session_id=None,
    ):
        if not self.enabled:
            return
        if event_type not in self.EVENT_TYPES:
            return
        try:
            with self._lock:
                self._event_counter += 1
                event = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "session_id": session_id or self.session_id,
                    "event_id": str(uuid.uuid4()),
                    "event_type": event_type,
                    "component": component,
                    "action": action,
                    "status": status,
                    "input_summary": self.redact(input_summary),
                    "output_summary": self.redact(output_summary),
                    "error": self.redact(error),
                    "metadata": self.redact(metadata) if isinstance(metadata, dict) else metadata,
                }
                self._events.append(event)
                line = json.dumps(event, ensure_ascii=False, default=str)
                if self._current_file is not None:
                    self._current_file.write(line + "\n")
                    self._current_file.flush()
                    self._rotate_if_needed()
        except Exception as exc:
            sys.stderr.write(f"[AUDIT] log_event failed: {exc}\n")

    def get_events(self, event_type=None, limit=None):
        events = list(self._events)
        if event_type is not None:
            events = [e for e in events if e.get("event_type") == event_type]
        if limit is not None:
            events = events[-limit:]
        return events

    def get_session_id(self):
        return self.session_id

    def close(self):
        try:
            self.log_event(
                "SESSION_END",
                component="AuditSystem",
                action="shutdown",
                status="success",
                metadata={"session_id": self.session_id},
            )
        except Exception:
            pass
        try:
            if self._current_file is not None:
                self._current_file.close()
        except Exception:
            pass
        self._current_file = None
