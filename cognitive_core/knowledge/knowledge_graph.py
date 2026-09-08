import time
from typing import Any, Dict, List, Optional


class KnowledgeGraph:
    def __init__(self):
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._relations: Dict[str, List[Dict[str, Any]]] = {}
        self._contradictions: List[Dict[str, Any]] = []

    def assert_fact(self, fact: str, confidence: float = 1.0, source: Optional[str] = None):
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("Un fatto deve essere una stringa non vuota")
        fact = fact.strip()
        existing = self._facts.get(fact)
        if existing is not None:
            if confidence > existing.get("confidence", 0.0):
                existing["confidence"] = confidence
                existing["source"] = source or existing.get("source")
                existing["updated_at"] = time.time()
                existing["version"] = existing.get("version", 1) + 1
            return
        self._facts[fact] = {
            "confidence": max(0.0, min(1.0, confidence)),
            "source": source,
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1,
        }

    def contains(self, fact: str) -> bool:
        return fact in self._facts

    def get_fact(self, fact: str) -> Optional[Dict[str, Any]]:
        return self._facts.get(fact)

    def confidence(self, fact: str) -> float:
        entry = self._facts.get(fact)
        return entry.get("confidence", 0.0) if entry else 0.0

    def relate(self, source: str, relation: str, target: str, confidence: float = 1.0, source_ref: Optional[str] = None):
        self.assert_fact(source, confidence=confidence, source=source_ref)
        self.assert_fact(target, confidence=confidence, source=source_ref)
        entry = {"relation": relation, "target": target, "confidence": confidence, "source": source_ref, "timestamp": time.time()}
        self._relations.setdefault(source, [])
        if entry not in self._relations[source]:
            self._relations[source].append(entry)

    def related(self, source: str) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._relations.get(source, [])]

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """Rileva fatti contraddittori basati su confidence e sorgenti."""
        contradictions = []
        facts = list(self._facts.keys())
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                if self._are_contradictory(f1, f2):
                    contradictions.append({
                        "fact_a": f1,
                        "fact_b": f2,
                        "confidence_a": self._facts[f1]["confidence"],
                        "confidence_b": self._facts[f2]["confidence"],
                        "source_a": self._facts[f1].get("source"),
                        "source_b": self._facts[f2].get("source"),
                    })
        self._contradictions = contradictions
        return contradictions

    def _are_contradictory(self, f1: str, f2: str) -> bool:
        """Euristica semplice: frasi con parole di negazione opposte."""
        negation = {"non", "mai", "nessuno", "niente", "nulla", "falso", "no"}
        words1 = set(f1.lower().split())
        words2 = set(f2.lower().split())
        neg1 = bool(words1 & negation)
        neg2 = bool(words2 & negation)
        if neg1 != neg2:
            shared = words1 & words2
            if len(shared) >= 2:
                return True
        return False

    def verify(self, fact: str, evidence: Optional[str] = None) -> Dict[str, Any]:
        """Verifica un fatto contro evidenze e altre conoscenze."""
        entry = self._facts.get(fact)
        if entry is None:
            return {"status": "unknown", "confidence": 0.0, "reason": "fact_not_found"}
        contradictions = [c for c in self.detect_contradictions() if c["fact_a"] == fact or c["fact_b"] == fact]
        if contradictions:
            return {
                "status": "contradicted",
                "confidence": entry["confidence"],
                "reason": "contradiction_detected",
                "contradictions": contradictions,
            }
        return {
            "status": "verified",
            "confidence": entry["confidence"],
            "source": entry.get("source"),
            "version": entry.get("version"),
        }
