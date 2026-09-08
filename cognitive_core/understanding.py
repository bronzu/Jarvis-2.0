"""
Understanding module for JARVIS 2.0 Cognitive Core.

Responsabilities:
- intent extraction
- entity extraction
- relation extraction
- task classification
- input normalization
- constraint identification
- goal identification
- missing information identification
- structured representation
"""

import re
from typing import Any, Dict, List, Optional


class Understanding:
    """Trasforma input grezzo in rappresentazione strutturata."""

    INTENT_QUESTION = "question"
    INTENT_COMMAND = "command"
    INTENT_INFORMATION = "information"
    INTENT_MULTISTEP = "multistep"
    INTENT_UNKNOWN = "unknown"

    TASK_CLASSIFIER_PATTERNS = {
        INTENT_QUESTION: [
            r"^(chi|cosa|dove|quando|perché|come|quale|quanti|quante)\b",
            r"\?$",
            r"\b(spiega|definisci|descrivi|dimmi|racconta)\b",
        ],
        INTENT_COMMAND: [
            r"^(esegui|avvia|ferma|apri|chiudi|leggi|scrivi|cerca|connetti|analizza|genera|calcola)\b",
            r"^\[.+\]$",
        ],
        INTENT_INFORMATION: [
            r"\b(il fatto è|la verità è|in realtà|effettivamente|dati|statistiche|valore|risultato)\b",
        ],
        INTENT_MULTISTEP: [
            r"\b(poi|quindi|inoltre|successivamente|infine|prima|dopo|step|fase)\b",
            r"\b(e|&)\b",
        ],
    }

    ENTITY_PATTERNS = [
        r"\b[A-Z][a-z]+\b",  # Nomi propri capitalized
        r"\b\d{4}\b",  # Anni
        r"\b\d+(?:\.\d+)?\b",  # Numeri
        r"`[^`]+`",  # Code blocks
        r"'[^']{3,}'",  # Stringhe tra apici
        r'"[^"]{3,}"',  # Stringhe tra virgolette
    ]

    STOPWORDS = {
        "Il", "La", "Lo", "I", "Gli", "Le", "Un", "Una", "Uno", "Dei", "Delle", "Degli",
        "Come", "Cosa", "Dove", "Quando", "Perché", "Come", "Quale", "Quanti", "Quante",
        "Chi", "Che", "Non", "Per", "Con", "Su", "Tra", "Fra", "Da", "Di", "A", "In",
        "Esegui", "Avvia", "Ferma", "Apri", "Chiudi", "Leggi", "Scrivi", "Cerca", "Connetti", "Analizza", "Genera", "Calcola",
    }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        seen = set()
        for pattern in self.ENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                value = match.group().strip("`'\"")
                if value and value not in seen and value not in self.STOPWORDS:
                    seen.add(value)
                    entities.append({
                        "value": value,
                        "type": self._guess_entity_type(value),
                        "span": match.span(),
                    })
        return entities

    def understand(self, raw_input: Any) -> Dict[str, Any]:
        if raw_input is None:
            return self._empty_understanding("input None")

        text = self._normalize(raw_input)
        if not text:
            return self._empty_understanding("input vuoto")

        intent = self._classify_intent(text)
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities)
        constraints = self._extract_constraints(text)
        goal = self._extract_goal(text, intent)
        missing = self._identify_missing(text, intent, entities, relations)

        return {
            "raw": text,
            "intent": intent,
            "entities": entities,
            "relations": relations,
            "constraints": constraints,
            "goal": goal,
            "missing_information": missing,
            "confidence": self._confidence(intent, entities, relations),
            "type": "understanding",
        }

    def _empty_understanding(self, reason: str) -> Dict[str, Any]:
        return {
            "raw": "",
            "intent": self.INTENT_UNKNOWN,
            "entities": [],
            "relations": [],
            "constraints": [],
            "goal": None,
            "missing_information": [reason],
            "confidence": 0.0,
            "type": "understanding",
        }

    def _normalize(self, raw_input: Any) -> str:
        if isinstance(raw_input, dict):
            text = raw_input.get("text") or raw_input.get("query") or raw_input.get("goal") or str(raw_input)
        else:
            text = str(raw_input)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _classify_intent(self, text: str) -> str:
        scores = {}
        for intent, patterns in self.TASK_CLASSIFIER_PATTERNS.items():
            scores[intent] = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return self.INTENT_UNKNOWN
        return best_intent

    def _guess_entity_type(self, value: str) -> str:
        if re.match(r"^\d{4}$", value):
            return "year"
        if re.match(r"^\d+(?:\.\d+)?$", value):
            return "number"
        if re.match(r"^[A-Z][a-z]+$", value):
            return "proper_noun"
        if re.match(r"^[A-Z]{2,}$", value):
            return "acronym"
        if value.startswith("`") or value.startswith("'") or value.startswith('"'):
            return "code"
        return "generic"

    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relations = []
        entity_values = [e["value"] for e in entities]
        relation_patterns = [
            (r"è\s+(?:un|una|il|la)\s+", "is_a"),
            (r"ha\s+", "has"),
            (r"appartiene\s+a", "belongs_to"),
            (r"causa", "causes"),
            (r"deriva\s+da", "derives_from"),
            (r"richiede", "requires"),
            (r">=|<=|>|<|==|!=", "compares"),
        ]

        for pattern, rel_type in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                window = text[start:end]
                nearby = [v for v in entity_values if v.lower() in window.lower()]
                if len(nearby) >= 2:
                    relations.append({
                        "type": rel_type,
                        "source": nearby[0],
                        "target": nearby[1],
                        "span": match.span(),
                    })
        return relations

    def _extract_constraints(self, text: str) -> List[str]:
        constraints = []
        patterns = [
            r"\b(non\s+deve|non\s+può|obbligatorio|vincolo|solo|massimo|minimo|entro|entro\s+la\s+data)\b[^.]*\.",
            r"\b(if|when|while|unless|until)\b[^.]*\.",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                constraints.append(match.group().strip())
        return constraints

    def _extract_goal(self, text: str, intent: str) -> Optional[str]:
        if intent == self.INTENT_QUESTION:
            return text.rstrip("?")
        if intent == self.INTENT_COMMAND:
            return text
        goal_patterns = [
            r"\b(obiettivo|goal|scopo|target|intenzione)\b[:\s]+([^\.]+)",
            r"^([^\.]{10,})$",
        ]
        for pattern in goal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(match.lastindex).strip()
        return text if len(text) > 5 else None

    def _identify_missing(self, text: str, intent: str, entities: List[Dict], relations: List[Dict]) -> List[str]:
        missing = []
        if intent == self.INTENT_QUESTION and not entities:
            missing.append("specificare il soggetto della domanda")
        if intent == self.INTENT_COMMAND and not entities:
            missing.append("specificare il target del comando")
        if intent == self.INTENT_MULTISTEP and len(entities) < 2:
            missing.append("specificare le entità coinvolte")
        if not text:
            missing.append("input mancante")
        return missing

    def _confidence(self, intent: str, entities: List[Dict], relations: List[Dict]) -> float:
        if intent == self.INTENT_UNKNOWN:
            return 0.2
        score = 0.5
        if entities:
            score += 0.2
        if relations:
            score += 0.2
        if len(entities) > 2:
            score += 0.1
        return min(1.0, score)
