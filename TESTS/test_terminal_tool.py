from unittest.mock import MagicMock, patch

import pytest

from executive_system.tools.internet_tool import InternetTool
from executive_system.tools.terminal_tool import TerminalTool


class TestConnectiCommand:
    def test_detects_connecti_command(self):
        tool = TerminalTool()
        ok, query = tool._is_connecti_command("[CONNETTI: hello world]")
        assert ok is True
        assert query == "hello world"

    def test_does_not_detect_normal_command(self):
        tool = TerminalTool()
        ok, query = tool._is_connecti_command("echo hello")
        assert ok is False
        assert query is None

    def test_routes_connecti_through_chain(self):
        internet_tool = InternetTool()
        internet_tool.fetch = MagicMock(return_value="<html><body>result</body></html>")
        internet_tool.extract_text = MagicMock(return_value="result")

        coordinator = MagicMock()
        coordinator.process = MagicMock(return_value={"default_response": "processed query"})

        router = MagicMock()
        router.route = MagicMock(return_value={"route": "local"})

        context_manager = MagicMock()
        context_manager.add_turn = MagicMock()

        sandbox = MagicMock()
        sandbox.write_file = MagicMock()

        tool = TerminalTool(
            coordinator=coordinator,
            router=router,
            internet_tool=internet_tool,
            sandbox=sandbox,
            context_manager=context_manager,
        )

        result = tool.run("[CONNETTI: test query]", lambda cmd: cmd, approved=True)

        assert result["query"] == "test query"
        assert result["source"] == "internet_tool"
        assert "response" in result
        context_manager.add_turn.assert_called_once()
        sandbox.write_file.assert_called_once()

    def test_connecti_requires_internet_tool(self):
        tool = TerminalTool()
        with pytest.raises(RuntimeError):
            tool.run("[CONNETTI: test]", lambda cmd: cmd, approved=True)

    def test_connecti_updates_context(self):
        internet_tool = InternetTool()
        internet_tool.fetch = MagicMock(return_value="<html>text</html>")
        internet_tool.extract_text = MagicMock(return_value="text")

        context_manager = MagicMock()
        context_manager.add_turn = MagicMock()

        tool = TerminalTool(internet_tool=internet_tool, context_manager=context_manager)
        tool.run("[CONNETTI: context test]", lambda cmd: cmd, approved=True)

        context_manager.add_turn.assert_called_once()
        args = context_manager.add_turn.call_args
        assert "[CONNETTI: context test]" in args[0][0]
        assert args[0][1]["tool"] == "terminal"
