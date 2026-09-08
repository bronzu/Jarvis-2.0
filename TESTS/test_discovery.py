import pytest

from expansion_system.discovery.discovery_engine import DiscoveryEngine
from expansion_system.discovery.experimenter import Experimenter


class TestDiscoveryEngine:
    def test_investigate_creates_investigation(self):
        engine = DiscoveryEngine()
        inv = engine.investigate("perché piove?", ["umidita_alta", "front_freddo"])
        assert inv["state"] == "unknown"
        assert len(inv["hypotheses"]) == 2

    def test_predict_adds_prediction(self):
        engine = DiscoveryEngine()
        inv = engine.investigate("q?", ["h1"])
        pred = engine.predict(inv, "h1")
        assert pred["expected"] == "h1"

    def test_run_experiment_returns_observation(self):
        engine = DiscoveryEngine()
        inv = engine.investigate("q?", ["h1"])
        exp = engine.run_experiment(inv, lambda: {"matches_prediction": True})
        assert exp["outcome"] == "success"

    def test_evaluate_known_when_positive(self):
        engine = DiscoveryEngine()
        inv = engine.investigate("q?", ["h1"])
        engine.predict(inv, "h1")
        engine.observe(inv, {"matches_prediction": True})
        result = engine.evaluate(inv)
        assert result["state"] == "known"
        assert result["supported_hypothesis"] == "h1"

    def test_conclude_returns_knowledge(self):
        engine = DiscoveryEngine()
        inv = engine.investigate("q?", ["h1"])
        result = engine.conclude(inv, "h1")
        assert result["knowledge"] == "h1"
        assert result["state"] == "known"


class TestExperimenter:
    def test_run_success(self):
        exp = Experimenter()
        result = exp.run(lambda: "ok")
        assert result["outcome"] == "success"
        assert result["result"] == "ok"

    def test_run_error(self):
        exp = Experimenter()
        result = exp.run(lambda: 1 / 0)
        assert result["outcome"] == "error"