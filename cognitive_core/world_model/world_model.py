import time
from typing import Any, Dict, List, Optional


class WorldModel:
    """Rappresenta il mondo con entità, proprietà, relazioni e transizioni causali."""

    def __init__(self):
        self._causes: Dict[str, List[str]] = {}
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: List[Dict[str, str]] = []
        self._history: List[Dict[str, Any]] = []

    def add_causal_relation(self, cause, effect):
        if not cause or not effect:
            raise ValueError("Causa ed effetto sono obbligatori")
        self._causes.setdefault(cause, [])
        if effect not in self._causes[cause]:
            self._causes[cause].append(effect)

    def predict(self, state):
        return list(self._causes.get(state, []))

    def register_entity(self, name: str, properties: Optional[Dict[str, Any]] = None):
        if not name or not str(name).strip():
            raise ValueError("Nome entità obbligatorio")
        self._entities[str(name).strip()] = {
            "properties": dict(properties or {}),
            "updated_at": time.time(),
        }

    def set_property(self, entity: str, property_name: str, value: Any):
        if entity not in self._entities:
            self.register_entity(entity)
        self._entities[entity]["properties"][property_name] = value
        self._entities[entity]["updated_at"] = time.time()
        self._history.append({
            "type": "property_update",
            "entity": entity,
            "property": property_name,
            "value": value,
            "timestamp": time.time(),
        })

    def get_property(self, entity: str, property_name: str, default=None):
        return self._entities.get(entity, {}).get("properties", {}).get(property_name, default)

    def relate(self, source: str, relation: str, target: str):
        if not source or not relation or not target:
            raise ValueError("source, relation e target sono obbligatori")
        entry = {"source": source, "relation": relation, "target": target}
        if entry not in self._relations:
            self._relations.append(entry)

    def relations(self, source: Optional[str] = None, relation: Optional[str] = None, target: Optional[str] = None) -> List[Dict[str, str]]:
        result = []
        for entry in self._relations:
            if source is not None and entry["source"] != source:
                continue
            if relation is not None and entry["relation"] != relation:
                continue
            if target is not None and entry["target"] != target:
                continue
            result.append(dict(entry))
        return result

    def apply_change(self, state_change: Dict[str, Any]):
        """Applica un cambiamento di stato e registra conseguenze previste.

        state_change deve contenere 'entity', 'property', 'value'.
        """
        entity = state_change.get("entity")
        prop = state_change.get("property")
        value = state_change.get("value")
        if entity is None or prop is None:
            raise ValueError("state_change richiede 'entity' e 'property'")

        old_value = self.get_property(entity, prop)
        self.set_property(entity, prop, value)

        consequences = []
        exact_key = f"{entity}.{prop}"
        for cause, effects in self._causes.items():
            if cause == exact_key or cause == prop or cause == str(value):
                consequences.extend(effects)
        self._history.append({
            "type": "state_change",
            "entity": entity,
            "property": prop,
            "old_value": old_value,
            "new_value": value,
            "consequences": consequences,
            "timestamp": time.time(),
        })
        return {
            "old_value": old_value,
            "new_value": value,
            "consequences": consequences,
        }

    def current_state(self) -> Dict[str, Any]:
        state = {}
        for entity, data in self._entities.items():
            state[entity] = dict(data.get("properties", {}))
        return state

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        data = list(self._history)
        if limit is not None:
            data = data[-limit:]
        return data

    def predict_with_confidence(self, state: str) -> Dict[str, Any]:
        """Predizione con confidence basata su frequenza storica."""
        predictions = self._causes.get(state, [])
        if not predictions:
            return {"predicted": [], "confidence": 0.0, "method": "none"}

        confidence = min(1.0, len(predictions) / max(1, len(self._history)))
        return {
            "predicted": predictions,
            "confidence": round(confidence, 4),
            "method": "frequency",
            "history_count": len(self._history),
        }

    def simulate(self, initial_state: Dict[str, Any], action: Dict[str, Any], steps: int = 1) -> Dict[str, Any]:
        """Simula una sequenza di transizioni di stato."""
        current = dict(initial_state)
        trace = [dict(current)]
        for _ in range(max(1, steps)):
            state_key = action.get("target_state") or action.get("entity", "")
            predicted = self.predict(state_key)
            if predicted:
                current[state_key] = predicted[0]
            trace.append(dict(current))
        return {
            "initial": trace[0],
            "final": trace[-1],
            "trace": trace,
            "steps": len(trace) - 1,
        }

    def compare_prediction(self, predicted_state: Dict[str, Any], observed_state: Dict[str, Any]) -> Dict[str, Any]:
        """Confronta stato predetto vs osservato e calcola errore."""
        all_keys = set(predicted_state) | set(observed_state)
        errors = []
        matches = 0
        for key in all_keys:
            p = predicted_state.get(key)
            o = observed_state.get(key)
            if p == o:
                matches += 1
            else:
                errors.append({"key": key, "predicted": p, "observed": o})
        total = len(all_keys)
        accuracy = matches / total if total > 0 else 1.0
        return {
            "accuracy": accuracy,
            "error_rate": 1.0 - accuracy,
            "matches": matches,
            "errors": len(errors),
            "details": errors,
        }

    def update_from_feedback(self, predicted_state: Dict[str, Any], observed_state: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        """Aggiorna il world model basandosi su errore di predizione."""
        comparison = self.compare_prediction(predicted_state, observed_state)
        if comparison["accuracy"] < threshold:
            for detail in comparison["details"]:
                entity = detail["key"]
                observed = detail["observed"]
                if entity and observed is not None:
                    self.set_property(entity, "last_observed", observed)
                    self._history.append({
                        "type": "model_update",
                        "entity": entity,
                        "predicted": detail["predicted"],
                        "observed": observed,
                        "accuracy": comparison["accuracy"],
                        "timestamp": time.time(),
                    })
        return comparison
