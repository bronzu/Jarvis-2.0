import numpy as np

from cognitive_core.knowledge.vector_memory import VectorMemory


def test_vector_memory_searches_by_cosine_similarity():
    memory = VectorMemory()
    memory.add("casa", np.array([1.0, 0.0, 0.0]))
    memory.add("auto", np.array([0.0, 1.0, 0.0]))
    memory.add("albero", np.array([0.0, 0.0, 1.0]))

    results = memory.search(np.array([1.0, 0.0, 0.0]), top_k=2)

    assert results[0]["id"] == "casa"
    assert results[0]["score"] == 1.0
    assert results[1]["id"] in {"auto", "albero"}
