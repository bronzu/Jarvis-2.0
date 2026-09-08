"""Generazione e validazione di ipotesi deduttive."""


class HypothesisGenerator:
    """Trasforma regole in ipotesi e ne verifica le condizioni."""

    def generate(self, rules):
        """Genera un'ipotesi per ogni regola disponibile.

        Ogni ipotesi esprime una possibile conclusione e le condizioni che
        devono essere note per poterla accettare.
        """
        if not isinstance(rules, list):
            raise TypeError("rules deve essere una lista.")

        hypotheses = []
        for rule in rules:
            self._validate_rule(rule)
            hypotheses.append({
                "conditions": list(rule["if"]),
                "conclusion": rule["then"],
            })

        return hypotheses

    def validate(self, hypothesis, known_facts):
        """Restituisce True quando tutte le condizioni sono note."""
        self._validate_hypothesis(hypothesis)

        if not isinstance(known_facts, list):
            raise TypeError("known_facts deve essere una lista.")

        return all(
            condition in known_facts
            for condition in hypothesis["conditions"]
        )

    def _validate_rule(self, rule):
        if not isinstance(rule, dict):
            raise TypeError("Ogni regola deve essere un dizionario.")

        if set(rule) != {"if", "then"}:
            raise ValueError(
                "Ogni regola deve contenere solo 'if' e 'then'."
            )

        if not isinstance(rule["if"], list) or not rule["if"]:
            raise ValueError(
                "Ogni regola deve avere almeno una condizione."
            )

        if not all(isinstance(condition, str) for condition in rule["if"]):
            raise TypeError("Le condizioni della regola devono essere stringhe.")

        if not isinstance(rule["then"], str):
            raise TypeError("La conclusione della regola deve essere una stringa.")

    def _validate_hypothesis(self, hypothesis):
        if not isinstance(hypothesis, dict):
            raise TypeError("L'ipotesi deve essere un dizionario.")

        if set(hypothesis) != {"conditions", "conclusion"}:
            raise ValueError(
                "L'ipotesi deve contenere solo 'conditions' e 'conclusion'."
            )

        if not isinstance(hypothesis["conditions"], list):
            raise TypeError("Le condizioni dell'ipotesi devono essere una lista.")

        if not isinstance(hypothesis["conclusion"], str):
            raise TypeError("La conclusione dell'ipotesi deve essere una stringa.")
