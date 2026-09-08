import time
import uuid
from typing import Any, Dict, Optional


class CandidateBuilder:
    VALID_STATES = {"proposed", "sandbox", "testing", "approved", "rejected", "deployed", "rolled_back"}

    def __init__(self):
        self._candidates: Dict[str, Dict[str, Any]] = {}

    def build(self, change, parent_version: Optional[str] = None, reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if not change:
            raise ValueError("Una modifica è obbligatoria")
        candidate_id = str(uuid.uuid4())
        candidate = {
            "id": candidate_id,
            "change": change,
            "state": "sandbox",
            "parent_version": parent_version,
            "reason": reason or "candidatura_generata",
            "metadata": dict(metadata or {}),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._candidates[candidate_id] = candidate
        return candidate

    def create_candidate_version(self, parent_version: str, change: Dict[str, Any], reason: str, benchmark_before: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not parent_version:
            raise ValueError("parent_version è obbligatorio")
        if not change:
            raise ValueError("change è obbligatorio")
        candidate_id = str(uuid.uuid4())
        candidate_version = f"{parent_version}-candidate-{candidate_id[:8]}"
        candidate = {
            "id": candidate_id,
            "candidate_version": candidate_version,
            "parent_version": parent_version,
            "change": change,
            "state": "sandbox",
            "reason": reason,
            "benchmark_before": benchmark_before or {},
            "benchmark_after": None,
            "decision": None,
            "metadata": dict(metadata or {}),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._candidates[candidate_id] = candidate
        return candidate

    def get(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        candidate = self._candidates.get(candidate_id)
        return dict(candidate) if candidate else None

    def update_state(self, candidate_id: str, new_state: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if new_state not in self.VALID_STATES:
            raise ValueError(f"Stato non valido: {new_state}")
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        candidate["state"] = new_state
        candidate["updated_at"] = time.time()
        if metadata:
            candidate.setdefault("metadata", {}).update(metadata)
        if new_state == "deployed":
            candidate["deployed_at"] = time.time()
        if new_state == "rolled_back":
            candidate["rolled_back_at"] = time.time()
        return dict(candidate)

    def record_benchmark_after(self, candidate_id: str, benchmark_after: Dict[str, Any]) -> Dict[str, Any]:
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        candidate["benchmark_after"] = benchmark_after
        candidate["updated_at"] = time.time()
        return dict(candidate)

    def record_decision(self, candidate_id: str, decision: str, reason: str) -> Dict[str, Any]:
        if decision not in {"approve", "reject", "rollback"}:
            raise ValueError("Decisione non valida")
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        candidate["decision"] = decision
        candidate["decision_reason"] = reason
        candidate["decided_at"] = time.time()
        candidate["updated_at"] = time.time()
        return dict(candidate)

    def record_checkpoint(self, candidate_id: str, checkpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        candidate.setdefault("checkpoint", {})
        candidate["checkpoint"].update(checkpoint_info)
        candidate["updated_at"] = time.time()
        return dict(candidate)

    def record_rollback(self, candidate_id: str, rollback_info: Dict[str, Any]) -> Dict[str, Any]:
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        candidate.setdefault("rollback", {})
        candidate["rollback"].update(rollback_info)
        candidate["updated_at"] = time.time()
        return dict(candidate)

    def all_candidates(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._candidates.values()]

    def candidates_by_state(self, state: str) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._candidates.values() if item.get("state") == state]
