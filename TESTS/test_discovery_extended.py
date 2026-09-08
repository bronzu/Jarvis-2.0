"""
Tests for discovery engine extensions.
"""

import pytest

from expansion_system.discovery.discovery_engine import DiscoveryEngine


@pytest.fixture
def engine():
    return DiscoveryEngine()


class TestDiscoveryEngineExtensions:
    def test_generate_hypotheses_why_question(self, engine):
        hypotheses = engine.generate_hypotheses("Perché il modello perde?")
        assert len(hypotheses) > 0
        texts = [h["text"] for h in hypotheses]
        assert "causa_diretta" in texts

    def test_generate_hypotheses_how_question(self, engine):
        hypotheses = engine.generate_hypotheses("Come funziona l'attenzione?")
        texts = [h["text"] for h in hypotheses]
        assert "meccanismo_noto" in texts

    def test_generate_hypotheses_with_facts(self, engine):
        hypotheses = engine.generate_hypotheses("Perché X?", facts=["fact1", "fact2"])
        assert len(hypotheses) >= 2

    def test_generate_hypotheses_empty_fallback(self, engine):
        hypotheses = engine.generate_hypotheses("Random query senza keyword")
        assert len(hypotheses) >= 1
        assert any(h["text"] == "ipotesi_generica" for h in hypotheses)

    def test_evaluate_with_metrics(self, engine):
        investigation = engine.investigate("test", ["H1", "H2"])
        engine.predict(investigation, "H1")
        engine.run_experiment(investigation, lambda: {"result": "ok"})
        engine.observe(investigation, {"matches_prediction": True})
        result = engine.evaluate_with_metrics(investigation)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_full_discovery_cycle(self, engine):
        investigation = engine.investigate("Perché X?", ["H1", "H2"])
        hypotheses = engine.generate_hypotheses("Perché X?", ["fact1"])
        investigation["hypotheses"] = [h["text"] for h in hypotheses]

        for h in hypotheses:
            engine.predict(investigation, h["text"])

        engine.run_experiment(investigation, lambda: {"matches_prediction": True})
        engine.observe(investigation, {"matches_prediction": True, "hypothesis": hypotheses[0]["text"]})

        result = engine.evaluate_with_metrics(investigation)
        assert result["score"] > 0.0
        conclusion = engine.conclude(investigation, hypotheses[0]["text"])
        assert conclusion["state"] == "known"
