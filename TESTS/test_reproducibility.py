"""
Tests for neural_core.reproducibility.
"""

from neural_core.reproducibility import run_reproducibility_experiment


def test_reproducibility_experiment_is_numerically_stable():
    result = run_reproducibility_experiment(steps=5, tolerance=1e-6)
    assert result["numerically_reproducible"] is True
    assert result["loss_match"] is True
    assert result["params_match"] is True
    assert result["max_loss_diff"] == 0.0
    assert result["run_a"]["seed"] == result["run_b"]["seed"]
    assert result["run_a"]["param_checksum"] == result["run_b"]["param_checksum"]
