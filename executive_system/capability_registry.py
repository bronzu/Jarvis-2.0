class CapabilityRegistry:
    ALLOWED_STATES = {"available", "missing", "restricted"}

    def __init__(self):
        self._capabilities = {}
        self._tool_factories = {}

    def register(self, name, state="available"):
        if state not in self.ALLOWED_STATES:
            raise ValueError("Stato non valido")
        self._capabilities[name] = state

    def register_tool(self, name, factory):
        if not name or not callable(factory):
            raise ValueError("Nome e factory sono obbligatori")
        self._capabilities[name] = "available"
        self._tool_factories[name] = factory

    def get_factory(self, name):
        return self._tool_factories.get(name)

    def state(self, name):
        return self._capabilities.get(name, "missing")
