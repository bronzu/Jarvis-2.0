from executive_system.capability_registry import CapabilityRegistry


class ToolSelector:
    def __init__(self, registry=None):
        self.registry = registry or CapabilityRegistry()

    def select(self, capability, tools=None):
        if tools is not None:
            if not isinstance(tools, dict):
                raise TypeError("tools deve essere un dizionario")
            tool = tools.get(capability)
            if tool is not None:
                return tool

        factory = self.registry.get_factory(capability)
        if factory is None:
            return None
        return factory()
