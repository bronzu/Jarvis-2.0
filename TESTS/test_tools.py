import pytest
from executive_system.tools.filesystem_tool import FilesystemTool
from executive_system.tools.python_runner import PythonRunner
from executive_system.tools.terminal_tool import TerminalTool

def test_tools_require_approval_and_keep_files_in_sandbox(tmp_path):
    tool = FilesystemTool(tmp_path)
    with pytest.raises(PermissionError): tool.write("a.txt", "x")
    tool.write("a.txt", "x", approved=True); assert tool.read("a.txt") == "x"
    with pytest.raises(PermissionError): TerminalTool().run("x", lambda command: command)
    assert PythonRunner().run("result = sum([1, 2])", approved=True)["result"] == 3
