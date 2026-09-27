import pytest

from expansion_system.discovery.discovery_engine import DiscoveryEngine
from expansion_system.discovery.experimenter import Experimenter
from INFRASTRUCTURE.sandbox.subprocess_sandbox import SubprocessSandbox


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
        sandbox = SubprocessSandbox(timeout=5.0)
        result = exp.run(lambda: "ok", sandbox=sandbox)
        assert result["outcome"] == "success"
        assert result["result"] == "ok"
        assert result["isolated_execution"] is True

    def test_run_error(self):
        exp = Experimenter()
        sandbox = SubprocessSandbox(timeout=5.0)
        result = exp.run(lambda: 1 / 0, sandbox=sandbox)
        assert result["outcome"] == "error"

    def test_run_without_sandbox_rejected(self):
        """Callable experiments without a sandbox must be rejected."""
        exp = Experimenter()
        with pytest.raises(PermissionError):
            exp.run(lambda: "ok")

    def test_run_with_hyperparameters_no_sandbox_allowed(self):
        """Hyperparameter dict experiments work without sandbox (documented)."""
        exp = Experimenter()
        result = exp.run({"learning_rate": 0.01, "epochs": 10})
        assert result["outcome"] == "success"
        assert result["hyperparameter_only"] is True
        assert result["isolated_execution"] is False

    def test_run_privileged_action_blocked_without_sandbox(self):
        """Privileged callable without sandbox must be blocked."""
        exp = Experimenter()

        def privileged():
            with open("/tmp/_exp_privileged_test", "w") as f:
                f.write("EXPLOIT")
            return "executed"

        with pytest.raises(PermissionError):
            exp.run(privileged, sandbox=None)