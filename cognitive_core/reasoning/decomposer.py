"""Scomposizione di problemi strutturati per il sistema di reasoning."""


class Decomposer:
    """Separa un problema nei suoi fatti, regole e obiettivo.

    Il componente non interpreta linguaggio naturale: questa responsabilità
    appartiene al futuro organo di Understanding. Riceve invece una
    rappresentazione esplicita del problema e ne restituisce una copia
    validata, pronta per gli altri organi del reasoning.
    """

    REQUIRED_FIELDS = ("facts", "rules", "goal")

    def decompose(self, problem):
        """Restituisce i componenti validati di un problema.

        Args:
            problem: dizionario con ``facts`` (lista di affermazioni),
                ``rules`` (lista di regole) e ``goal`` (affermazione da
                verificare).

        Returns:
            dict: una nuova rappresentazione del problema.
        """
        if not isinstance(problem, dict):
            raise TypeError("Il problema deve essere un dizionario.")

        missing_fields = [
            field for field in self.REQUIRED_FIELDS
            if field not in problem
        ]
        if missing_fields:
            raise ValueError(
                "Il problema deve contenere: "
                + ", ".join(missing_fields) + "."
            )

        facts = self._validate_statements(problem["facts"], "facts")
        rules = self._validate_rules(problem["rules"])
        goal = self._validate_statement(problem["goal"], "goal")

        return {
            "facts": facts,
            "rules": rules,
            "goal": goal,
        }

    def _validate_statements(self, statements, field_name):
        if not isinstance(statements, list):
            raise TypeError(f"{field_name} deve essere una lista.")

        return [
            self._validate_statement(statement, field_name)
            for statement in statements
        ]

    def _validate_rules(self, rules):
        if not isinstance(rules, list):
            raise TypeError("rules deve essere una lista.")

        validated_rules = []
        for rule in rules:
            if not isinstance(rule, dict):
                raise TypeError("Ogni regola deve essere un dizionario.")

            if set(rule) != {"if", "then"}:
                raise ValueError(
                    "Ogni regola deve contenere solo 'if' e 'then'."
                )

            conditions = self._validate_statements(rule["if"], "if")
            if not conditions:
                raise ValueError("Una regola deve avere almeno una condizione.")

            validated_rules.append({
                "if": conditions,
                "then": self._validate_statement(rule["then"], "then"),
            })

        return validated_rules

    def _validate_statement(self, statement, field_name):
        if not isinstance(statement, str):
            raise TypeError(f"{field_name} deve contenere solo stringhe.")

        normalized_statement = statement.strip()
        if not normalized_statement:
            raise ValueError(f"{field_name} non può contenere stringhe vuote.")

        return normalized_statement
