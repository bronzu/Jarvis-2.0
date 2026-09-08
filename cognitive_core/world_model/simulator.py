from typing import Any, Dict, List, Optional


class Simulator:
    """Simula passi di evoluzione del mondo secondo il world model."""

    def __init__(self, world_model):
        self.world_model = world_model

    def simulate(self, initial_state: str, steps: int = 1) -> List[str]:
        if steps < 0:
            raise ValueError("steps non può essere negativo")
        states, frontier = [initial_state], [initial_state]
        for _ in range(steps):
            next_frontier = []
            for state in frontier:
                next_frontier.extend(self.world_model.predict(state))
            for state in next_frontier:
                if state not in states:
                    states.append(state)
            frontier = next_frontier
        return states

    def simulate_entity_property(self, entity: str, property_name: str, new_value: Any) -> Dict[str, Any]:
        change = {"entity": entity, "property": property_name, "value": new_value}
        return self.world_model.apply_change(change)
