import threading

import pytest

from cognitive_core.context_manager import ContextManager
from cognitive_core.memory.memory_manager import MemoryManager
from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.sandbox.sandbox_env import SandboxEnvironment


def test_context_manager_keeps_recent_context_and_stats():
    manager = ContextManager(max_turns=5)

    manager.add_turn("turn 1", {"topic": "a"})
    manager.add_turn("turn 2", {"topic": "b"})
    manager.add_turn("turn 3", {"topic": "c"})
    manager.add_turn("turn 4", {"topic": "d"})
    manager.add_turn("turn 5", {"topic": "e"})
    manager.add_turn("turn 6", {"topic": "f"})

    context = manager.get_context()
    assert len(context["history"]) <= 5
    assert context["history"][-1]["input"] == "turn 6"
    assert context["turn_count"] == 6


def test_memory_manager_handles_concurrent_access_and_returns_semantic_results():
    manager = MemoryManager()
    manager.store("alpha", {"name": "alpha", "tags": ["a", "b"]})
    manager.store("beta", {"name": "beta", "tags": ["a", "c"]})

    def worker():
        for idx in range(20):
            manager.store(f"item-{idx}", {"name": f"item-{idx}", "tags": ["x"]})

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    results = manager.search_semantic("alpha")
    assert any(item["id"] == "alpha" for item in results)


def test_permission_guard_blocks_dangerous_system_calls():
    guard = PermissionGuard()

    with pytest.raises(PermissionError):
        guard.validate_command("os.system('rm -rf /')")

    with pytest.raises(PermissionError):
        guard.validate_command("subprocess.run(['ls', '/'])")

    assert guard.validate_command("python script.py", approved=True) is True


def test_benchmark_comparator_builds_readable_report():
    comparator = BenchmarkComparator()
    report = comparator.compare_versions(
        "v1",
        "v2",
        {
            "logical_accuracy": 0.9,
            "execution_time": 1.1,
            "memory_usage": 400,
            "json_adherence": 0.95,
            "tool_success": 0.8,
        },
        {
            "logical_accuracy": 0.95,
            "execution_time": 0.8,
            "memory_usage": 350,
            "json_adherence": 0.98,
            "tool_success": 0.95,
        },
    )

    assert report["winner"] == "v2"
    assert "summary" in report
    assert report["summary"]["logical_accuracy"]["delta"] > 0


def test_sandbox_environment_isolation():
    sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_sandbox_test")
    file_path = sandbox.write_file("demo.txt", "hello")

    assert file_path.exists()
    assert sandbox.read_file("demo.txt") == "hello"
    with pytest.raises(PermissionError):
        sandbox.write_file("../escape.txt", "nope")
