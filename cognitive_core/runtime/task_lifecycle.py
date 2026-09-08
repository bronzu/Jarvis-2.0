"""Task lifecycle for the JARVIS Agent Runtime.

This module defines the explicit task states a long-running agent task
can be in, the legal transitions between them, and a small helper that
enforces those transitions.

Why this exists
---------------
The previous JARVIS architecture represented task status with simple
booleans (e.g. ``goal["status"] in {"pending","in_progress",...}`` in
``executive_system/goal_manager.py``) or with single-state flags. That
was not sufficient for an agentic runtime that must distinguish:

  * queued       — accepted, not yet started
  * running      — actively iterating the loop
  * paused       — suspended, can be resumed
  * waiting_user — needs explicit human input/approval
  * waiting_tool — blocked on an external tool call
  * waiting_external — blocked on an external agent / system
  * recovering   — handling a known error class
  * failed       — terminated due to unrecoverable error
  * completed    — verified success
  * cancelled    — explicitly cancelled by user/system
  * aborted      — terminated by safety / invariant violation

The legal-transition matrix below makes impossible states unrepresentable
at runtime (Rule 130: classify errors; Rule 133: no magic strings).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Iterable


class LifecycleError(RuntimeError):
    """Raised when an illegal task lifecycle transition is requested."""


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    WAITING_TOOL = "waiting_tool"
    WAITING_EXTERNAL = "waiting_external"
    RECOVERING = "recovering"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"

    @classmethod
    def terminal(cls) -> FrozenSet["TaskStatus"]:
        return frozenset({cls.COMPLETED, cls.FAILED, cls.CANCELLED, cls.ABORTED})


# Explicit transition matrix. Key = current state. Value = set of allowed
# next states. Anything not listed is forbidden and raises LifecycleError.
_TRANSITIONS: Dict[TaskStatus, FrozenSet[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.RUNNING: frozenset({
        TaskStatus.PAUSED,
        TaskStatus.WAITING_USER,
        TaskStatus.WAITING_TOOL,
        TaskStatus.WAITING_EXTERNAL,
        TaskStatus.RECOVERING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.PAUSED: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.WAITING_USER: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.WAITING_TOOL: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.RECOVERING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.WAITING_EXTERNAL: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.RECOVERING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.RECOVERING: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.WAITING_USER,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.ABORTED,
    }),
    TaskStatus.FAILED: frozenset({}),
    TaskStatus.COMPLETED: frozenset({}),
    TaskStatus.CANCELLED: frozenset({}),
    TaskStatus.ABORTED: frozenset({}),
}


def allowed_transitions(current: TaskStatus) -> FrozenSet[TaskStatus]:
    return _TRANSITIONS[current]


def is_legal_transition(current: TaskStatus, target: TaskStatus) -> bool:
    return target in _TRANSITIONS[current]


class TaskLifecycle:
    """Tracks the lifecycle of a single task.

    The class is intentionally small. The AgentRuntime composes one
    TaskLifecycle per task. It is *not* a God Object: it does not know
    about memory, planning, tools, or the cognitive loop. Those are
    the runtime's responsibility.
    """

    def __init__(self, task_id: str, initial: TaskStatus = TaskStatus.QUEUED):
        if not task_id:
            raise ValueError("task_id is required")
        self.task_id = task_id
        self._status: TaskStatus = initial
        self._history: list = [(0.0, initial)]

    @property
    def status(self) -> TaskStatus:
        return self._status

    def is_terminal(self) -> bool:
        return self._status in TaskStatus.terminal()

    def transition(self, target: TaskStatus, at_time: float = 0.0) -> None:
        if not isinstance(target, TaskStatus):
            raise LifecycleError(f"target must be a TaskStatus, got {type(target).__name__}")
        if not is_legal_transition(self._status, target):
            raise LifecycleError(
                f"Illegal transition for task {self.task_id}: "
                f"{self._status.value} -> {target.value}"
            )
        self._status = target
        self._history.append((at_time, target))

    def force(self, status: TaskStatus) -> None:
        """Set status without enforcing transitions (use only on construction / recovery)."""
        if not isinstance(status, TaskStatus):
            raise LifecycleError(f"status must be a TaskStatus, got {type(status).__name__}")
        self._status = status
        self._history.append((0.0, status))

    def history(self) -> Iterable:
        return tuple(self._history)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self._status.value,
            "history": [(t, s.value) for t, s in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskLifecycle":
        initial = TaskStatus(data.get("status", "queued"))
        lc = cls(data["task_id"], initial=initial)
        history = data.get("history", [])
        # If the first history entry matches the initial state, skip it
        # to avoid duplicating the entry already added in __init__.
        for i, (t, s) in enumerate(history):
            if i == 0 and TaskStatus(s) == initial:
                continue
            lc._history.append((float(t), TaskStatus(s)))
        return lc
