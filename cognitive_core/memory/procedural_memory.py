class ProceduralMemory:
    def __init__(self):
        self._procedures = {}

    def register(self, name, steps):
        if not name or not isinstance(steps, list) or not steps:
            raise ValueError("Nome e passaggi non vuoti sono obbligatori")
        self._procedures[name] = list(steps)

    def retrieve(self, name):
        return list(self._procedures.get(name, []))
