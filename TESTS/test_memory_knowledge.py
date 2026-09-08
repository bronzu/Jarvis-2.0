from cognitive_core.knowledge.knowledge_graph import KnowledgeGraph
from cognitive_core.memory.episodic_memory import EpisodicMemory
from cognitive_core.memory.procedural_memory import ProceduralMemory
from cognitive_core.memory.semantic_memory import SemanticMemory
from cognitive_core.memory.working_memory import WorkingMemory


def test_working_memory_keeps_active_context_with_capacity():
    memory = WorkingMemory(capacity=2)
    memory.remember("a"); memory.remember("b"); memory.remember("c")
    assert memory.recall() == ["b", "c"]


def test_episodic_memory_records_and_recalls_events():
    memory = EpisodicMemory(); memory.record("test", "passed")
    assert memory.recall("test") == [{"event": "test", "outcome": "passed"}]


def test_semantic_and_procedural_memory_retrieve_information():
    semantic = SemanticMemory(); semantic.learn("acqua", "bagna")
    procedural = ProceduralMemory(); procedural.register("test", ["esegui", "verifica"])
    assert semantic.recall("acqua") == ["bagna"]
    assert procedural.retrieve("test") == ["esegui", "verifica"]


def test_knowledge_graph_stores_facts_and_relations():
    graph = KnowledgeGraph(); graph.relate("pioggia", "causa", "strada bagnata")
    assert graph.contains("pioggia")
    result = graph.related("pioggia")
    assert len(result) == 1
    assert result[0]["relation"] == "causa"
    assert result[0]["target"] == "strada bagnata"
    assert result[0]["confidence"] == 1.0
