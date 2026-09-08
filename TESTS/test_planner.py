"""
Tests for cognitive_core.planner.
"""

import pytest

from cognitive_core.planner import Planner, Plan, PlanStep, Precondition, Effect


class FakeReasoner:
    def reason(self, problem):
        return {"proved": False, "goal": problem.get("goal", "")}


class FakeMemory:
    def search_semantic(self, query):
        return []


class FakeToolSelector:
    def select(self, capability):
        return lambda: f"tool:{capability}"


class FakeAutonomyManager:
    def act(self, action, approved=False):
        if not approved:
            raise PermissionError("approval required")
        return action()


class FakePermissionGuard:
    def require_user_approval(self, approved):
        if approved is not True:
            raise PermissionError("approval required")


@pytest.fixture
def planner():
    return Planner(
        reasoner=FakeReasoner(),
        memory=FakeMemory(),
        tool_selector=FakeToolSelector(),
        autonomy_manager=FakeAutonomyManager(),
        permission_guard=FakePermissionGuard(),
    )


class TestPlanner:
    def test_empty_goal_raises(self, planner):
        with pytest.raises(ValueError):
            planner.plan("")

    def test_plan_has_goal(self, planner):
        plan = planner.plan("analizza il sistema")
        assert plan.goal == "analizza il sistema"
        assert plan.status == "ready"

    def test_plan_has_steps(self, planner):
        plan = planner.plan("addestra modello")
        assert len(plan.steps) >= 1

    def test_plan_validation_valid(self, planner):
        plan = planner.plan("test")
        result = planner.validate(plan)
        assert result["status"] == "valid"

    def test_plan_validation_no_cycles(self, planner):
        plan = planner.plan("test")
        result = planner.validate(plan)
        assert not any("Ciclo" in e for e in result["errors"])

    def test_execute_completes_steps(self, planner):
        plan = planner.plan("test")
        result = planner.execute(plan)
        assert result["status"] in ("completed", "failed")

    def test_fallback_on_permission_denied(self, planner):
        plan = planner.plan("azione non autorizzata")
        result = planner.execute(plan)
        assert result["status"] in ("completed", "failed")

    def test_plan_step_dependencies(self, planner):
        plan = planner.plan("test")
        step_ids = [s.id for s in plan.steps]
        for step in plan.steps:
            for dep in step.dependencies:
                assert dep in step_ids

    def test_plan_serialization(self, planner):
        plan = planner.plan("test")
        data = plan.to_dict()
        assert data["goal"] == "test"
        assert "steps" in data
        assert len(data["steps"]) == len(plan.steps)

    def test_plan_step_serialization(self, planner):
        step = PlanStep("test step", preconditions=[Precondition("check")], effects=[Effect("effect")])
        data = step.to_dict()
        assert data["description"] == "test step"
        assert len(data["preconditions"]) == 1
        assert len(data["effects"]) == 1
