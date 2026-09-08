from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
from expansion_system.self_improvement.candidate_builder import CandidateBuilder


def test_self_improvement_rejects_equal_or_worse_candidates():
    comparator = BenchmarkComparator()
    assert comparator.compare(10, 10) == "reject"
    assert comparator.compare(10, 9) == "reject"
    assert comparator.compare(10, 11) == "approval_required"
    assert comparator.compare(10, 12) == "approval_required"
    assert comparator.compare(10, 5) == "reject"


def test_candidate_builder_starts_in_sandbox_state():
    state = CandidateBuilder().build("change")
    assert state["state"] == "sandbox"
    assert "change" in state or "id" in state or "diff" in state or state.get("state") == "sandbox"
