import pytest

from executive_system.goal_manager import GoalManager


class TestGoalManager:
    def test_add_returns_goal_id(self):
        gm = GoalManager()
        goal_id = gm.add("test goal", ["step1", "step2"])
        assert isinstance(goal_id, str)
        assert len(goal_id) > 0

    def test_get_by_text_compatibility(self):
        gm = GoalManager()
        gm.add("test goal", ["step1"])
        goal = gm.get("test goal")
        assert goal["status"] == GoalManager.STATUS_PENDING
        assert len(goal["steps"]) == 1

    def test_start_and_next_pending_step(self):
        gm = GoalManager()
        goal_id = gm.add("goal", ["a", "b", "c"])
        gm.start(goal_id)
        step = gm.next_pending_step(goal_id)
        assert step["description"] == "a"
        assert step["status"] == GoalManager.STATUS_IN_PROGRESS
        goal = gm.get(goal_id)
        assert goal["status"] == GoalManager.STATUS_IN_PROGRESS

    def test_complete_step_and_goal(self):
        gm = GoalManager()
        goal_id = gm.add("goal", ["a", "b"])
        gm.start(goal_id)
        step = gm.next_pending_step(goal_id)
        gm.complete_step(goal_id, step["id"], "ok")
        step = gm.next_pending_step(goal_id)
        gm.complete_step(goal_id, step["id"], "ok")
        goal = gm.complete(goal_id)
        assert goal["status"] == GoalManager.STATUS_COMPLETED

    def test_fail_step_sets_goal_failed(self):
        gm = GoalManager()
        goal_id = gm.add("goal", ["a", "b"])
        gm.start(goal_id)
        step = gm.next_pending_step(goal_id)
        gm.fail_step(goal_id, step["id"], "boom")
        goal = gm.get(goal_id)
        assert goal["status"] == GoalManager.STATUS_FAILED

    def test_status_summary(self):
        gm = GoalManager()
        gm.add("g1")
        gm.add("g2")
        summary = gm.status_summary()
        assert summary[GoalManager.STATUS_PENDING] == 2
        assert summary[GoalManager.STATUS_COMPLETED] == 0