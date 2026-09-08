"""
Tests for extended world model capabilities.
"""

import pytest

from cognitive_core.world_model.world_model import WorldModel


@pytest.fixture
def world_model():
    return WorldModel()


class TestWorldModelPrediction:
    def test_predict_returns_effects(self, world_model):
        world_model.add_causal_relation("rain", "wet_ground")
        assert "wet_ground" in world_model.predict("rain")

    def test_predict_unknown_state(self, world_model):
        assert world_model.predict("unknown") == []

    def test_predict_with_confidence(self, world_model):
        world_model.add_causal_relation("A", "B")
        result = world_model.predict_with_confidence("A")
        assert "predicted" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0

    def test_simulate_state_transition(self, world_model):
        world_model.add_causal_relation("start", "end")
        result = world_model.simulate({"start": 1}, {"target_state": "start"})
        assert "initial" in result
        assert "final" in result
        assert "trace" in result

    def test_compare_prediction_perfect_match(self, world_model):
        predicted = {"A": 1, "B": 2}
        observed = {"A": 1, "B": 2}
        result = world_model.compare_prediction(predicted, observed)
        assert result["accuracy"] == 1.0
        assert result["error_rate"] == 0.0

    def test_compare_prediction_mismatch(self, world_model):
        predicted = {"A": 1, "B": 2}
        observed = {"A": 1, "B": 3}
        result = world_model.compare_prediction(predicted, observed)
        assert result["accuracy"] < 1.0
        assert result["errors"] == 1

    def test_update_from_feedback(self, world_model):
        predicted = {"temp": 20}
        observed = {"temp": 25}
        result = world_model.update_from_feedback(predicted, observed)
        assert "accuracy" in result
        history = world_model.history()
        assert len(history) > 0
