class SemanticMemory:
    def __init__(self):
        self._knowledge = {}

    def learn(self, concept, fact):
        if not concept or not fact:
            raise ValueError("Concetto e fatto sono obbligatori")
        self._knowledge.setdefault(concept, [])
        if fact not in self._knowledge[concept]:
            self._knowledge[concept].append(fact)

    def recall(self, concept):
        return list(self._knowledge.get(concept, []))
