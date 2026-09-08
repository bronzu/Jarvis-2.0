import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional


class GoalManager:
    """Pianificatore e gestore degli obiettivi di JARVIS.

    Ogni goal viene scomposto in step tracciabili. Lo stato di ogni step
    evolve attraverso: pending -> in_progress -> completed | failed.
    """

    STATUS_PENDING = "pending"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    def __init__(self):
        self._goals: Dict[str, Dict[str, Any]] = {}
        self._by_text: Dict[str, str] = {}
        self._lock = Lock()

    def add(self, goal: str, steps: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        if not goal or not str(goal).strip():
            raise ValueError("Un obiettivo è obbligatorio")
        goal_id = str(uuid.uuid4())
        with self._lock:
            text_key = str(goal).strip().lower()
            existing_id = self._by_text.get(text_key)
            if existing_id and existing_id in self._goals:
                return existing_id
            self._goals[goal_id] = {
                "id": goal_id,
                "goal": str(goal).strip(),
                "steps": [
                    {
                        "id": str(uuid.uuid4()),
                        "description": str(step).strip(),
                        "status": self.STATUS_PENDING,
                        "result": None,
                        "updated_at": time.time(),
                    }
                    for step in (steps or [])
                ],
                "status": self.STATUS_PENDING,
                "metadata": dict(metadata or {}),
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self._by_text[text_key] = goal_id
        return goal_id

    def get(self, goal_id: str) -> Dict[str, Any]:
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None and goal_id in self._by_text:
                goal = self._goals.get(self._by_text[goal_id])
            if goal is None:
                # Retrocompatibilità: se non esiste, restituisci struttura vuota con status pending
                return {"id": goal_id, "goal": goal_id, "status": self.STATUS_PENDING, "steps": []}
            return dict(goal)

    def get_by_goal_text(self, goal_text: str) -> Optional[Dict[str, Any]]:
        lowered = str(goal_text).strip().lower()
        with self._lock:
            goal_id = self._by_text.get(lowered)
            if goal_id and goal_id in self._goals:
                return dict(self._goals[goal_id])
        return None

    def start(self, goal_id: str) -> Dict[str, Any]:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                # Retrocompatibilità: cerca per testo
                goal_id = self._by_text.get(str(goal_id).strip().lower())
                if not goal_id or goal_id not in self._goals:
                    raise KeyError(f"Goal non trovato: {goal_id}")
                goal = self._goals[goal_id]
            goal["status"] = self.STATUS_IN_PROGRESS
            goal["updated_at"] = time.time()
            return dict(goal)

    def complete_step(self, goal_id: str, step_id: str, result: Any = None) -> Dict[str, Any]:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                raise KeyError(f"Goal non trovato: {goal_id}")
            step = next((s for s in goal["steps"] if s["id"] == step_id), None)
            if not step:
                raise KeyError(f"Step non trovato: {step_id}")
            step["status"] = self.STATUS_COMPLETED
            step["result"] = result
            step["updated_at"] = time.time()
            if all(s["status"] == self.STATUS_COMPLETED for s in goal["steps"]):
                goal["status"] = self.STATUS_COMPLETED
            goal["updated_at"] = time.time()
            return dict(goal)

    def fail_step(self, goal_id: str, step_id: str, error: str = "") -> Dict[str, Any]:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                raise KeyError(f"Goal non trovato: {goal_id}")
            step = next((s for s in goal["steps"] if s["id"] == step_id), None)
            if not step:
                raise KeyError(f"Step non trovato: {step_id}")
            step["status"] = self.STATUS_FAILED
            step["result"] = error
            step["updated_at"] = time.time()
            goal["status"] = self.STATUS_FAILED
            goal["updated_at"] = time.time()
            return dict(goal)

    def complete(self, goal_id: str) -> Dict[str, Any]:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                raise KeyError(f"Goal non trovato: {goal_id}")
            goal["status"] = self.STATUS_COMPLETED
            for step in goal["steps"]:
                if step["status"] == self.STATUS_IN_PROGRESS:
                    step["status"] = self.STATUS_COMPLETED
                    step["updated_at"] = time.time()
            goal["updated_at"] = time.time()
            return dict(goal)

    def next_pending_step(self, goal_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            for step in goal["steps"]:
                if step["status"] == self.STATUS_PENDING:
                    step["status"] = self.STATUS_IN_PROGRESS
                    step["updated_at"] = time.time()
                    goal["updated_at"] = time.time()
                    return dict(step)
        return None

    def all_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._goals.values()]

    def pending_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._goals.values() if item["status"] == self.STATUS_PENDING]

    def in_progress_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._goals.values() if item["status"] == self.STATUS_IN_PROGRESS]

    def failed_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._goals.values() if item["status"] == self.STATUS_FAILED]

    def status_summary(self) -> Dict[str, int]:
        with self._lock:
            summary = {status: 0 for status in (
                self.STATUS_PENDING, self.STATUS_IN_PROGRESS, self.STATUS_COMPLETED, self.STATUS_FAILED
            )}
            for item in self._goals.values():
                summary[item["status"]] = summary.get(item["status"], 0) + 1
            return summary
