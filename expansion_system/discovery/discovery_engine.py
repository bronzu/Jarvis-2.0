import time
from typing import Any, Dict, List, Optional


class DiscoveryEngine:
    """Gestisce il ciclo di scoperta per problemi sconosciuti."""

    def investigate(self, question: str, hypotheses: List[str]) -> Dict[str, Any]:
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("Domanda valida è obbligatoria")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError("Una lista di ipotesi è obbligatoria")

        investigation_id = str(time.time())
        return {
            "id": investigation_id,
            "question": str(question).strip(),
            "state": "unknown",
            "hypotheses": [str(h) for h in hypotheses],
            "predictions": [],
            "experiments": [],
            "observations": [],
            "conclusion": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    def predict(self, investigation: Dict[str, Any], hypothesis: str) -> Dict[str, Any]:
        if hypothesis not in investigation["hypotheses"]:
            raise ValueError("Ipotesi non proposta nell'investigazione")
        prediction = {
            "hypothesis": hypothesis,
            "expected": hypothesis,
            "confidence": 0.0,
            "timestamp": time.time(),
        }
        investigation["predictions"].append(prediction)
        investigation["updated_at"] = time.time()
        return prediction

    def run_experiment(self, investigation: Dict[str, Any], experiment: Any) -> Dict[str, Any]:
        if not callable(experiment):
            raise TypeError("Esperimento non eseguibile")
        result = experiment()
        outcome = "success" if not isinstance(result, dict) or result.get("error") is None else "error"
        experiment_entry = {
            "result": result,
            "outcome": outcome,
            "timestamp": time.time(),
        }
        investigation["experiments"].append(experiment_entry)
        investigation["updated_at"] = time.time()
        return experiment_entry

    def observe(self, investigation: Dict[str, Any], observation: Dict[str, Any]):
        if not isinstance(observation, dict):
            raise TypeError("observation deve essere un dict")
        observation["timestamp"] = time.time()
        investigation["observations"].append(observation)
        investigation["updated_at"] = time.time()

    def evaluate(self, investigation: Dict[str, Any], evaluator=None) -> Dict[str, Any]:
        if evaluator is not None and callable(evaluator):
            score = evaluator(investigation)
        else:
            score = self._default_evaluation(investigation)

        supported = None
        if score > 0.5 and investigation["predictions"]:
            supported = investigation["predictions"][-1]["hypothesis"]

        investigation["state"] = "known" if supported else "unknown"
        investigation["conclusion"] = supported
        investigation["updated_at"] = time.time()
        return {
            "state": investigation["state"],
            "supported_hypothesis": supported,
            "score": score,
        }

    def conclude(self, investigation: Dict[str, Any], supported_hypothesis: str) -> Dict[str, Any]:
        if supported_hypothesis not in investigation["hypotheses"]:
            raise ValueError("Ipotesi non proposta")
        investigation["state"] = "known"
        investigation["conclusion"] = supported_hypothesis
        investigation["updated_at"] = time.time()
        return {
            "question": investigation["question"],
            "knowledge": supported_hypothesis,
            "state": investigation["state"],
        }

    def _default_evaluation(self, investigation: Dict[str, Any]) -> float:
        if not investigation["observations"]:
            return 0.0
        positive = sum(1 for obs in investigation["observations"] if obs.get("matches_prediction"))
        return positive / max(1, len(investigation["observations"]))

    def generate_hypotheses(self, question: str, facts: Optional[List[str]] = None, failures: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Genera ipotesi automatiche basate sulla domanda, sui fatti noti e sui fallimenti."""
        facts = facts or []
        failures = failures or []
        hypotheses = []
        question_lower = question.lower()

        if "perché" in question_lower or "perche" in question_lower:
            hypotheses.append({"text": "causa_diretta", "confidence": 0.5, "reasoning": "domanda esplicita una causa"})
            hypotheses.append({"text": "correlazione", "confidence": 0.3, "reasoning": "possibile correlazione"})

        if "come" in question_lower:
            hypotheses.append({"text": "meccanismo_noto", "confidence": 0.4, "reasoning": "meccanismo già documentato"})
            hypotheses.append({"text": "meccanismo_nuovo", "confidence": 0.2, "reasoning": "meccanismo non documentato"})

        if "cosa" in question_lower or "che" in question_lower:
            hypotheses.append({"text": "fattore_comune", "confidence": 0.3, "reasoning": "fattore comune tra casi"})

        for fact in facts:
            hypotheses.append({"text": f"conferma_da_{fact[:20]}", "confidence": 0.2, "reasoning": f"supportato da fatto: {fact}"})

        failure_types = [f.get("failure_type") or f.get("type") for f in failures if isinstance(f, dict)]
        for failure_type in failure_types:
            hypotheses.append({
                "text": f"risolve_{failure_type}",
                "confidence": 0.6,
                "reasoning": f"ipotesi generata da fallimento: {failure_type}",
                "source": "failure_data",
            })
        failure_categories = [f.get("category") for f in failures if isinstance(f, dict) and f.get("category")]
        for category in failure_categories:
            hypotheses.append({
                "text": f"migliora_{category}",
                "confidence": 0.4,
                "reasoning": f"ipotesi da categoria debole: {category}",
                "source": "failure_category",
            })

        if not hypotheses:
            hypotheses.append({"text": "ipotesi_generica", "confidence": 0.1, "reasoning": "nessuna ipotesi specifica generata"})

        return hypotheses

    def evaluate_with_metrics(self, investigation: Dict[str, Any]) -> Dict[str, Any]:
        """Valutazione strutturata con metriche multiple."""
        predictions = investigation.get("predictions", [])
        experiments = investigation.get("experiments", [])
        observations = investigation.get("observations", [])

        prediction_coverage = len(predictions) / max(1, len(investigation.get("hypotheses", [])))
        experiment_success = sum(1 for e in experiments if e.get("outcome") == "success") / max(1, len(experiments))
        observation_match = sum(1 for o in observations if o.get("matches_prediction")) / max(1, len(observations))

        score = (prediction_coverage * 0.3 + experiment_success * 0.4 + observation_match * 0.3)
        score = max(0.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "prediction_coverage": round(prediction_coverage, 4),
            "experiment_success": round(experiment_success, 4),
            "observation_match": round(observation_match, 4),
            "method": "structured",
        }
