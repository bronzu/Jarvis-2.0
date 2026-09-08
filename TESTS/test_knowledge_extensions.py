"""
Tests for knowledge graph extensions (conflicts, provenance, verification).
"""

import pytest

from cognitive_core.knowledge.knowledge_graph import KnowledgeGraph


@pytest.fixture
def kg():
    return KnowledgeGraph()


class TestKnowledgeGraphExtensions:
    def test_assert_fact_with_confidence(self, kg):
        kg.assert_fact("A è B", confidence=0.8, source="source1")
        assert kg.confidence("A è B") == 0.8

    def test_assert_fact_updates_confidence(self, kg):
        kg.assert_fact("X", confidence=0.5)
        kg.assert_fact("X", confidence=0.9)
        assert kg.confidence("X") == 0.9

    def test_get_fact_metadata(self, kg):
        kg.assert_fact("fact1", confidence=0.7, source="manual")
        fact = kg.get_fact("fact1")
        assert fact["confidence"] == 0.7
        assert fact["source"] == "manual"
        assert fact["version"] == 1

    def test_detect_contradictions(self, kg):
        kg.assert_fact("Il cielo è blu")
        kg.assert_fact("Il cielo non è blu")
        contradictions = kg.detect_contradictions()
        assert len(contradictions) > 0

    def test_verify_unknown_fact(self, kg):
        result = kg.verify("inesistente")
        assert result["status"] == "unknown"

    def test_verify_contradicted_fact(self, kg):
        kg.assert_fact("A è vero")
        kg.assert_fact("A non è vero")
        result = kg.verify("A è vero")
        assert result["status"] == "contradicted"

    def test_verify_consistent_fact(self, kg):
        kg.assert_fact("B è confermato", source="experiment_1")
        result = kg.verify("B è confermato")
        assert result["status"] == "verified"

    def test_relate_with_metadata(self, kg):
        kg.relate("A", "causa", "B", confidence=0.8, source_ref="theory_1")
        related = kg.related("A")
        assert len(related) == 1
        assert related[0]["confidence"] == 0.8
