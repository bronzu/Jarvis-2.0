import math

import numpy as np


class VectorMemory:
    """Memoria vettoriale per recupero semantico basato su similarità coseno."""

    def __init__(self):
        self._items = {}

    def add(self, item_id, vector):
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be a non-empty string")

        vector_array = np.asarray(vector, dtype=float)
        if vector_array.ndim == 0:
            raise ValueError("vector must be a one-dimensional array")

        self._items[item_id] = vector_array.astype(float)
        return item_id

    def search(self, query_vector, top_k=5):
        if isinstance(query_vector, (list, tuple)):
            query_vector = np.asarray(query_vector, dtype=float)
        query = np.asarray(query_vector, dtype=float)

        if query.ndim == 0:
            raise ValueError("query_vector must be a vector")

        if not self._items:
            return []

        results = []
        for item_id, stored_vector in self._items.items():
            if stored_vector.shape != query.shape:
                raise ValueError("query_vector and stored vectors must have the same dimension")

            dot = float(np.dot(query, stored_vector))
            norm_query = float(np.linalg.norm(query))
            norm_stored = float(np.linalg.norm(stored_vector))
            if norm_query == 0.0 or norm_stored == 0.0:
                score = 0.0
            else:
                score = dot / (norm_query * norm_stored)

            results.append({
                "id": item_id,
                "score": float(score),
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:max(1, int(top_k))]

    def get(self, item_id):
        if item_id not in self._items:
            raise KeyError(item_id)
        return self._items[item_id].copy()

    def __len__(self):
        return len(self._items)

    def __contains__(self, item_id):
        return item_id in self._items
