class WorkingMemory:
    def __init__(self, capacity=10):
        if capacity < 1:
            raise ValueError("capacity deve essere positiva")
        self.capacity, self._items = capacity, []

    def remember(self, item):
        if item is None:
            raise ValueError("Un elemento non può essere None")
        self._items.append(item)
        self._items = self._items[-self.capacity:]

    def recall(self):
        return list(self._items)

    def clear(self):
        self._items.clear()
