"""Tests for Finding 5: pickle.load RCE vulnerability in trainer.py, router.py, etc.

The code used `pickle.load()` directly for loading model weights, checkpoints,
and tokenizers. Since pickle can execute arbitrary code via __reduce__, a
malicious pickle file could achieve remote code execution.

Fix: Replaced all `pickle.load`/`pickle.loads` calls with `SafeUnpickler`
which only allows numpy types, builtins, and collections — blocking any
__reduce__ payload that references os.system, subprocess, etc.
"""
import os
import pickle
import sys

import numpy as np
import pytest

from neural_core.trainer import (
    safe_pickle_load,
    safe_pickle_loads,
    SafeUnpickler,
    load_model_from_weights,
)


class MaliciousPayload:
    """Payload that attempts arbitrary code execution via pickle __reduce__."""
    def __reduce__(self):
        return (os.system, ('echo PICKLE_RCE > /tmp/finding5_malicious_test.txt',))


class TestSafeUnpickler:
    def test_safe_unpickler_blocks_rce(self):
        """SafeUnpickler must block __reduce__ payloads calling os.system."""
        buf = pickle.dumps(MaliciousPayload())
        with pytest.raises(PermissionError):
            safe_pickle_loads(buf)

    def test_safe_unpickler_allows_numpy_arrays(self):
        """SafeUnpickler must allow numpy array deserialization."""
        arr = np.array([1.0, 2.0, 3.0])
        buf = pickle.dumps(arr)
        loaded = safe_pickle_loads(buf)
        np.testing.assert_array_equal(loaded, arr)

    def test_safe_unpickler_allows_numpy_dict_list(self):
        """SafeUnpickler must allow dicts/lists with numpy arrays."""
        state = {
            "format": "int8",
            "parameters": [np.array([1.0, 2.0]), np.array([3.0])],
            "vocab_size": 100,
        }
        buf = pickle.dumps(state)
        loaded = safe_pickle_loads(buf)
        assert loaded["format"] == "int8"
        assert len(loaded["parameters"]) == 2
        np.testing.assert_array_equal(loaded["parameters"][0], [1.0, 2.0])

    def test_safe_unpickler_blocks_import(self):
        """SafeUnpickler must block __import__ via builtins.eval."""
        class ImportPayload:
            def __reduce__(self):
                return (eval, ("__import__('os')",))
        buf = pickle.dumps(ImportPayload())
        with pytest.raises(PermissionError):
            safe_pickle_loads(buf)


class TestPickleRCEBlocked:
    def test_load_model_from_weights_rejects_malicious_pickle(self, tmp_path):
        """load_model_from_weights must not execute code from malicious pickle."""
        path = tmp_path / "malicious.pkl"
        with open(path, "wb") as f:
            pickle.dump(MaliciousPayload(), f)

        if os.path.exists("/tmp/finding5_malicious_test.txt"):
            os.remove("/tmp/finding5_malicious_test.txt")

        with pytest.raises(Exception):
            load_model_from_weights(str(path))

        assert not os.path.exists("/tmp/finding5_malicious_test.txt"), \
            "RCE payload executed — file was created!"

    def test_no_pickle_load_in_trainer_source(self):
        """Trainer source must not contain bare pickle.load calls."""
        from neural_core import trainer
        src = open(trainer.__file__).read()
        # SafeUnpickler is allowed; bare pickle.load is not
        assert "pickle.load(" not in src, "Found pickle.load() in trainer.py"
        assert "pickle.loads(" not in src, "Found pickle.loads() in trainer.py"

    def test_no_pickle_load_in_router_source(self):
        """Router source must not contain bare pickle.load calls."""
        from executive_system import router
        src = open(router.__file__).read()
        assert "pickle.load(" not in src, "Found pickle.load() in router.py"
        assert "pickle.loads(" not in src, "Found pickle.loads() in router.py"

    def test_no_pickle_load_in_gpu_benchmark_source(self):
        """gpu_benchmark must not contain bare pickle.load calls."""
        from neural_core import gpu_benchmark
        src = open(gpu_benchmark.__file__).read()
        assert "pickle.load(" not in src
        assert "pickle.loads(" not in src

    def test_no_pickle_load_in_phase2b_report_source(self):
        """phase2b_report must not contain bare pickle.loads calls."""
        from neural_core import phase2b_report
        src = open(phase2b_report.__file__).read()
        assert "pickle.load(" not in src
        assert "pickle.loads(" not in src
