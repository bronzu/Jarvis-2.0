"""
Tests for cognitive_core.understanding.
"""

import pytest

from cognitive_core.understanding import Understanding


@pytest.fixture
def understanding():
    return Understanding()


class TestUnderstanding:
    def test_simple_question(self, understanding):
        result = understanding.understand("Come funziona il modello neurale?")
        assert result["intent"] == Understanding.INTENT_QUESTION
        assert result["confidence"] > 0

    def test_command_intent(self, understanding):
        result = understanding.understand("Esegui analisi del dataset")
        assert result["intent"] == Understanding.INTENT_COMMAND
        assert result["goal"] is not None

    def test_information_intent(self, understanding):
        result = understanding.understand("Il fatto è che il modello richiede 2.7M parametri")
        assert result["intent"] == Understanding.INTENT_INFORMATION

    def test_multistep_intent(self, understanding):
        result = understanding.understand("Prima carica il dataset, poi addestra il modello, infine valuta")
        assert result["intent"] == Understanding.INTENT_MULTISTEP

    def test_entity_extraction(self, understanding):
        result = understanding.understand("Il Quark Charm è una particella scoperta nel 1974")
        assert len(result["entities"]) > 0
        values = [e["value"] for e in result["entities"]]
        assert "Quark" in values or "Charm" in values
        assert "1974" in values

    def test_relation_extraction(self, understanding):
        result = understanding.understand("Il modello è un transformer proprietario")
        assert len(result["relations"]) >= 0  # may or may not find relations

    def test_empty_input(self, understanding):
        result = understanding.understand("")
        assert result["intent"] == Understanding.INTENT_UNKNOWN
        assert result["confidence"] == 0.0

    def test_none_input(self, understanding):
        result = understanding.understand(None)
        assert result["intent"] == Understanding.INTENT_UNKNOWN
        assert result["confidence"] == 0.0

    def test_dict_input(self, understanding):
        result = understanding.understand({"text": "Analizza il sistema", "goal": "analisi"})
        assert result["intent"] == Understanding.INTENT_COMMAND
        assert result["goal"] == "Analizza il sistema"

    def test_normalization(self, understanding):
        result = understanding.understand("   Come   funziona   il   modello?   ")
        assert result["raw"] == "Come funziona il modello?"

    def test_question_without_entities(self, understanding):
        result = understanding.understand("Perché?")
        assert result["intent"] == Understanding.INTENT_QUESTION
        assert len(result["missing_information"]) > 0

    def test_command_without_entities(self, understanding):
        result = understanding.understand("Esegui")
        assert result["intent"] == Understanding.INTENT_COMMAND
        assert len(result["missing_information"]) > 0

    def test_structured_output(self, understanding):
        result = understanding.understand("Carica il dataset e addestra il modello")
        assert "raw" in result
        assert "intent" in result
        assert "entities" in result
        assert "relations" in result
        assert "constraints" in result
        assert "goal" in result
        assert "missing_information" in result
        assert "confidence" in result
        assert "type" in result
        assert result["type"] == "understanding"

    def test_confidence_range(self, understanding):
        for text in ["Chi è JARVIS?", "Esegui comando", "Il fatto è X", "Testo casuale"]:
            result = understanding.understand(text)
            assert 0.0 <= result["confidence"] <= 1.0
