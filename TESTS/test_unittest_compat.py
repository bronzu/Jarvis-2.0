import unittest

from cognitive_core.knowledge.vector_memory import VectorMemory
from executive_system.router import ExecutiveRouter
from neural_core.model import SelfAttention


class TestCompatibility(unittest.TestCase):
    def test_vector_memory_is_available(self):
        self.assertIsInstance(VectorMemory(), VectorMemory)

    def test_attention_is_available(self):
        self.assertIsNotNone(SelfAttention())

    def test_router_is_available(self):
        router = ExecutiveRouter()
        self.assertEqual(router.route("prompt semplice")["route"], "local")


if __name__ == "__main__":
    unittest.main()
