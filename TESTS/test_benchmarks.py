"""
Tests for neural_core.benchmarks.

Verifies that benchmark functions return real measurements with the expected structure.
"""

import os
import statistics

import pytest

from neural_core.benchmarks import (
    _benchmark_startup,
    _benchmark_inference,
    _benchmark_training,
    _benchmark_checkpoint_size,
    run_all,
)


def test_startup_benchmark_returns_real_values():
    result = _benchmark_startup(iterations=3)
    assert result["mean"] > 0
    assert result["median"] > 0
    assert result["unit"] == "seconds"
    assert result["iterations"] == 3


def test_inference_benchmark_returns_real_values():
    result = _benchmark_inference(iterations=5)
    assert result["latency_mean"] > 0
    assert result["latency_median"] > 0
    assert result["tokens_per_sec_mean"] > 0
    assert result["unit"] == "seconds"
    assert result["iterations"] == 5


def test_training_benchmark_returns_real_values():
    result = _benchmark_training(iterations=3)
    assert result["step_time_mean"] > 0
    assert result["step_time_median"] > 0
    assert result["steps_per_sec_mean"] > 0
    assert result["unit"] == "seconds"
    assert result["iterations"] == 3


def test_checkpoint_size_benchmark_returns_real_values(tmp_path):
    result = _benchmark_checkpoint_size(tmp_path)
    assert result["checkpoint_size_bytes"] > 0
    assert result["checkpoint_size_mb"] > 0


def test_run_all_produces_complete_report(tmp_path):
    report = run_all(tmp_path)
    assert "startup" in report
    assert "inference" in report
    assert "training" in report
    assert "checkpoint" in report
    assert "hardware" in report
    assert report["hardware"]["cpu"] > 0
