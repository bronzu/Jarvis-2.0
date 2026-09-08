import time
from typing import Any, Dict, List, Optional


class ObservationFeedback:
    """Registra osservazioni e outcome delle azioni per il loop di valutazione."""

    OUTCOME_SUCCESS = "success"
    OUTCOME_FAILURE = "failure"
    OUTCOME_UNEXPECTED = "unexpected"

    def __init__(self):
        self._observations: List[Dict[str, Any]] = []
        self._metrics = {
            "total_actions": 0,
            "successes": 0,
            "failures": 0,
            "unexpected": 0,
        }

    def record(self, action: Any, result: Any, outcome: Optional[str] = None, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if outcome is None and session_id is None and not metadata:
            entry = {"action": action, "result": result}
            self._observations.append(entry)
            return entry

        if outcome is None:
            outcome = self._classify_outcome(result)

        entry = {"action": action, "result": result, "outcome": outcome, "session_id": session_id, "timestamp": time.time()}
        entry["suggestion"] = self._suggest_correction(action, result, outcome) if outcome else None
        if metadata:
            entry["metadata"] = metadata

        self._observations.append(entry)
        self._metrics["total_actions"] += 1
        if outcome:
            key = f"{outcome}s" if not outcome.endswith("s") else f"{outcome}es"
            self._metrics[key] = self._metrics.get(key, 0) + 1
        return entry

    def record_structured(
        self,
        action: Dict[str, Any],
        observation: Dict[str, Any],
        verification: Dict[str, Any],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a structured observation for the agentic loop.

        Phase 2 extension. Stores the full Action → Observation →
        Verification triplet with evidence and provenance, while keeping
        the legacy ``record`` API intact.

        Args:
            action: Structured action dict (from ``create_action``).
            observation: Structured observation dict containing at least
                ``source``, ``timestamp``, ``raw``, and optionally
                ``interpreted``, ``confidence``, ``evidence``,
                ``provenance``, ``world_state_delta``.
            verification: Structured verification dict (from
                ``VerificationResult.to_dict``).
            session_id: Optional session/task identifier.
            metadata: Optional extra metadata.

        Returns:
            The stored observation entry.
        """
        if not isinstance(action, dict):
            raise TypeError("action must be a dict")
        if not isinstance(observation, dict):
            raise TypeError("observation must be a dict")
        if not isinstance(verification, dict):
            raise TypeError("verification must be a dict")

        entry = {
            "action": action,
            "observation": observation,
            "verification": verification,
            "session_id": session_id,
            "timestamp": time.time(),
            "structured": True,
        }
        if metadata:
            entry["metadata"] = metadata

        vstatus = verification.get("status", "")
        self._metrics["total_actions"] += 1
        if vstatus == "verified":
            self._metrics["successes"] = self._metrics.get("successes", 0) + 1
        elif vstatus in ("failed", "unverified"):
            self._metrics["failures"] = self._metrics.get("failures", 0) + 1
        elif vstatus == "unknown":
            self._metrics["unexpected"] = self._metrics.get("unexpected", 0) + 1

        self._observations.append(entry)
        return entry

    def all(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._observations]

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._observations[-limit:]]

    def success_rate(self) -> float:
        total = self._metrics["total_actions"]
        if total == 0:
            return 0.0
        return self._metrics["successes"] / total

    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def clear(self):
        self._observations.clear()
        self._metrics = {
            "total_actions": 0,
            "successes": 0,
            "failures": 0,
            "unexpected": 0,
        }

    def _classify_outcome(self, result: Any) -> str:
        text = str(result).lower()
        if any(k in text for k in ["errore", "error", "negato", "denied", "permission", "failure", "fallito"]):
            return self.OUTCOME_FAILURE
        if any(k in text for k in ["inaspettato", "unexpected", "imprevisto", "strano"]):
            return self.OUTCOME_UNEXPECTED
        return self.OUTCOME_SUCCESS

    def _suggest_correction(self, action: Any, result: Any, outcome: str) -> Optional[str]:
        if outcome == self.OUTCOME_FAILURE:
            if "permission" in str(result).lower() or "negato" in str(result).lower():
                return "richiedi_approvazione_utente"
            if "tool" in str(action).lower() and "non disponibile" in str(result):
                return "verifica_registry_tool"
            if "comando" in str(action).lower() or "terminal" in str(action).lower():
                return "controlla_sintassi_comando"
            return "analizzare_causa_fallimento"
        if outcome == self.OUTCOME_UNEXPECTED:
            return "rieseguire_esperimento_controllato"
        return None
