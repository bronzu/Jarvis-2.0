class ContextManager:
    """
    Mantiene il contesto recente come finestra scorrevole (sliding window).
    
    **DESIGN NOTES**:
    - Similar to MemoryManager but with BOUNDED history (max_turns)
    - Keeps only the last N turns for memory efficiency
    - turn_count continues to increment (shows total turns processed)
    - history is truncated to max_turns items (shows recent context only)
    - WARNING: Historical data is LOST when truncated, only recent context preserved
    
    **Asymmetry with MemoryManager**:
    - ContextManager: Sliding window (bounded)
    - MemoryManager: Keeps all items (unbounded with eviction policy)
    - This is BY DESIGN: Context is short-term, Memory is long-term
    
    **Usage Pattern**:
    For long conversations, consider:
    1. ContextManager for immediate context (recent 5-10 turns)
    2. MemoryManager for historical data (all turns with TTL)
    3. Database for persistent storage of important conversations
    """

    def __init__(self, max_turns=5):
        if max_turns < 1:
            raise ValueError("max_turns deve essere almeno 1")
        self.max_turns = max_turns
        self.history = []
        self.turn_count = 0

    def add_turn(self, user_input, metadata=None):
        if user_input is None:
            raise ValueError("L'input non può essere None")

        self.turn_count += 1
        turn = {
            "turn": self.turn_count,
            "input": str(user_input),
            "metadata": dict(metadata or {}),
        }
        self.history.append(turn)
        self.history = self.history[-self.max_turns:]
        return dict(turn)

    def get_context(self):
        return {
            "turn_count": self.turn_count,
            "history": [dict(item) for item in self.history],
        }
