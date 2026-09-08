import pytest
from executive_system.action_executor import ActionExecutor
from executive_system.capability_registry import CapabilityRegistry
from executive_system.goal_manager import GoalManager
from executive_system.observation_feedback import ObservationFeedback
from executive_system.tool_selector import ToolSelector

def test_executive_system_tracks_goals_and_capability_gaps():
    goals = GoalManager(); goals.add("test", ["run"]); registry = CapabilityRegistry(); registry.register("shell", "restricted")
    goal = goals.get("test")
    assert goal["status"] == "pending" and registry.state("shell") == "restricted"

def test_action_requires_approval():
    with pytest.raises(PermissionError): ActionExecutor().execute(lambda: "done")


def test_tool_selection_and_observation_feedback_are_explicit():
    selected = ToolSelector().select("read", {"read": "filesystem"})
    feedback = ObservationFeedback(); feedback.record(selected, "ok")
    recorded = feedback.all()
    assert len(recorded) == 1
    assert recorded[0]["action"] == "filesystem"
    assert recorded[0]["result"] == "ok"
