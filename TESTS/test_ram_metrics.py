"""
Tests for memory measurement units and semantic classification.

The benchmark must NOT conflate RSS, virtual memory and Python allocations.
"""

import platform

import pytest


def test_rss_mb_uses_correct_unit_per_platform(monkeypatch):
    """`ru_maxrss` units differ: bytes on Darwin, KB on Linux. Verify the
    formula in each platform by simulating the platform."""
    from neural_core import final_benchmark
    import neural_core.final_benchmark as fb

    target_mb = 100.0
    # On Darwin, ru_maxrss is in bytes; on Linux, in KB.
    fake_value = int(target_mb * 1024 * 1024)  # what Darwin would report

    class FakeUsage:
        ru_maxrss = fake_value

    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeUsage())

    # Darwin
    monkeypatch.setattr(fb.platform, "system", lambda: "Darwin")
    mb_darwin = final_benchmark._rss_mb()
    assert abs(mb_darwin - target_mb) < 0.01, f"Darwin expected {target_mb} MB, got {mb_darwin}"

    # Linux: ru_maxrss is in KB. Simulate a 100 MB process by reporting 100*1024 KB.
    monkeypatch.setattr(fb.platform, "system", lambda: "Linux")
    class FakeUsageLinux:
        ru_maxrss = int(target_mb * 1024)  # 100 MB in KB
    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeUsageLinux())
    mb_linux = final_benchmark._rss_mb()
    assert abs(mb_linux - target_mb) < 0.01, f"Linux expected {target_mb} MB, got {mb_linux}"


def test_rss_bytes_correct_per_platform(monkeypatch):
    from neural_core import final_benchmark
    import neural_core.final_benchmark as fb

    # Darwin: raw value is bytes
    monkeypatch.setattr(fb.platform, "system", lambda: "Darwin")
    class FakeDarwin:
        ru_maxrss = 5_242_880  # 5 MB in bytes
    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeDarwin())
    assert final_benchmark._rss_bytes() == 5_242_880

    # Linux: raw value is KB, so bytes = raw * 1024
    monkeypatch.setattr(fb.platform, "system", lambda: "Linux")
    class FakeLinux:
        ru_maxrss = 5_120  # 5 MB in KB
    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeLinux())
    assert final_benchmark._rss_bytes() == 5_242_880


def test_vm_mb_returns_none_on_darwin(monkeypatch):
    """Darwin does not expose ru_maxv; _vm_mb must return None, not RSS."""
    from neural_core import final_benchmark
    import neural_core.final_benchmark as fb
    monkeypatch.setattr(fb.platform, "system", lambda: "Darwin")
    class FakeDarwin:
        ru_maxrss = 5_242_880
        # Note: no ru_maxv on Darwin
    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeDarwin())
    assert final_benchmark._vm_mb() is None


def test_vm_mb_returns_value_on_linux(monkeypatch):
    from neural_core import final_benchmark
    import neural_core.final_benchmark as fb
    monkeypatch.setattr(fb.platform, "system", lambda: "Linux")
    class FakeLinux:
        ru_maxrss = 5_120
        ru_maxv = 2 * 1024 * 1024  # 2 GB in KB
    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeLinux())
    assert final_benchmark._vm_mb() == 2048.0


def test_rss_bytes_darwin(monkeypatch):
    from neural_core import final_benchmark
    import platform
    fake_value = 5_242_880
    class FakeUsage:
        ru_maxrss = fake_value
    import neural_core.final_benchmark as fb
    monkeypatch.setattr(fb.resource, "getrusage", lambda *_: FakeUsage())
    bytes_val = final_benchmark._rss_bytes()
    if platform.system() == "Darwin":
        assert bytes_val == fake_value
    else:
        assert bytes_val == fake_value * 1024


def test_ram_metrics_classified():
    from neural_core.final_benchmark import _ram_metrics
    r = _ram_metrics()
    assert "rss" in r
    assert "virtual_memory_mb" in r
    assert "python_tracemalloc" in r
    assert r["rss"]["description"].lower().startswith("resident")
    assert "darwin" in r["rss"]["unit_note"].lower() or "linux" in r["rss"]["unit_note"].lower()
    assert isinstance(r["rss"]["mb"], (int, float))
    assert r["rss"]["mb"] > 0


def test_python_tracemalloc_available_with_tracing_started():
    import tracemalloc
    from neural_core.final_benchmark import _python_allocations
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        r = _python_allocations()
        assert r["available"] is True
        assert r["peak_bytes"] >= 0
    finally:
        if not was_tracing:
            tracemalloc.stop()


def test_final_benchmark_ram_contains_all_three_categories(tmp_path):
    from neural_core.final_benchmark import run_final_benchmark
    r = run_final_benchmark(tmp_path)
    assert "rss" in r["ram"]
    assert "python_tracemalloc" in r["ram"]
    assert r["ram"]["rss"]["mb"] > 0
    assert r["ram"]["python_tracemalloc"]["peak_bytes"] >= 0
