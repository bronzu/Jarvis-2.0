"""
Tests for extended reasoning capabilities.
"""

import math

import pytest

from cognitive_core.reasoning.reasoner import Reasoner


class TestAnalogicalReasoning:
    def test_high_similarity_transfers_solution(self):
        r = Reasoner()
        source = {"features": ["A", "B", "C", "D"], "solution": "use_method_X", "confidence": 0.9}
        target = {"features": ["A", "B", "C", "E"]}
        result = r.analogical_reason(source, target)
        assert result["similarity"] > 0.5
        assert result["transferred_solution"] == "use_method_X"

    def test_low_similarity_does_not_transfer(self):
        r = Reasoner()
        source = {"features": ["A", "B", "C", "D"], "solution": "use_method_X"}
        target = {"features": ["E", "F", "G"]}
        result = r.analogical_reason(source, target)
        assert result["similarity"] < 0.5
        assert result["transferred_solution"] is None

    def test_empty_features_returns_zero_confidence(self):
        r = Reasoner()
        result = r.analogical_reason({"features": []}, {"features": ["A"]})
        assert result["confidence"] == 0.0


class TestProbabilisticReasoning:
    def test_bayesian_posterior_normalized(self):
        r = Reasoner()
        hypotheses = [
            {"name": "H1", "prior": 0.6},
            {"name": "H2", "prior": 0.4},
        ]
        evidence = [
            {"hypothesis": "H1", "likelihood": 0.8},
            {"hypothesis": "H2", "likelihood": 0.2},
        ]
        result = r.probabilistic_reason(hypotheses, evidence)
        posteriors = {h["name"]: h["posterior"] for h in result["hypotheses"]}
        assert abs(sum(posteriors.values()) - 1.0) < 1e-6
        assert posteriors["H1"] > posteriors["H2"]

    def test_empty_evidence_uses_prior(self):
        r = Reasoner()
        hypotheses = [{"name": "H1", "prior": 0.7}, {"name": "H2", "prior": 0.3}]
        result = r.probabilistic_reason(hypotheses, [])
        posteriors = {h["name"]: h["posterior"] for h in result["hypotheses"]}
        assert abs(sum(posteriors.values()) - 1.0) < 1e-6

    def test_invalid_input_raises(self):
        r = Reasoner()
        with pytest.raises(ValueError):
            r.probabilistic_reason([], [])
        with pytest.raises(ValueError):
            r.probabilistic_reason([{"name": "H1"}], "not_a_list")


class TestBayesianUpdate:
    def test_update_increases_probability_with_high_likelihood(self):
        r = Reasoner()
        result = r.bayesian_update(prior=0.5, likelihood=0.9, evidence_weight=1.0)
        assert result > 0.5

    def test_update_decreases_probability_with_low_likelihood(self):
        r = Reasoner()
        result = r.bayesian_update(prior=0.9, likelihood=0.1, evidence_weight=1.0)
        assert result < 0.9

    def test_extreme_priors(self):
        r = Reasoner()
        assert r.bayesian_update(0.0, 0.9) == 0.0
        assert r.bayesian_update(1.0, 0.1) == 1.0

    def test_clamping(self):
        r = Reasoner()
        result = r.bayesian_update(0.5, 1.5)
        assert 0.0 <= result <= 1.0


class TestConfidenceScore:
    def test_weighted_average(self):
        r = Reasoner()
        components = [
            {"weight": 0.7, "score": 0.9},
            {"weight": 0.3, "score": 0.4},
        ]
        result = r.confidence_score(components)
        expected = 0.7 * 0.9 + 0.3 * 0.4
        assert abs(result - expected) < 1e-6

    def test_empty_components(self):
        r = Reasoner()
        assert r.confidence_score([]) == 0.0

    def test_score_clamping(self):
        r = Reasoner()
        components = [{"weight": 1.0, "score": 1.5}]
        assert 0.0 <= r.confidence_score(components) <= 1.0
