import pytest

from expansion_system.tool_factory.factory import ToolFactory
from expansion_system.tool_factory.installer import Installer
from INFRASTRUCTURE.sandbox.sandbox_env import SandboxEnvironment


class TestToolFactory:
    def test_generate_code_from_spec(self):
        factory = ToolFactory()
        spec = {
            "name": "simple_math_tool",
            "description": "Tool matematico",
            "methods": [
                {"name": "add", "params": ["a", "b"], "implementation": "return a + b"}
            ],
        }
        code = factory.generate(spec, sandbox=None)
        assert "def add(" in code
        assert "return a + b" in code

    def test_validate_generated_accepts_safe_code(self):
        factory = ToolFactory()
        assert factory.validate_generated("x = 1\nreturn x")["valid"] is True

    def test_validate_generated_rejects_dangerous_code(self):
        factory = ToolFactory()
        assert factory.validate_generated("import os\nos.system('id')")["valid"] is False

    def test_generate_writes_to_sandbox(self):
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_tool_gen_test_2")
        factory = ToolFactory()
        spec = {
            "name": "sandbox_tool",
            "description": "Test",
            "methods": [
                {"name": "ping", "params": [], "implementation": "return 'pong'"}
            ],
        }
        path = factory.generate(spec, sandbox=sandbox)
        assert path is not None
        assert path.exists()


class TestInstaller:
    def test_install_requires_approval(self):
        installer = Installer()
        with pytest.raises(PermissionError):
            installer.install({"name": "x", "state": "sandbox"})

    def test_install_promotes_candidate(self):
        installer = Installer()
        result = installer.install({"name": "x", "state": "sandbox"}, approved=True)
        assert result["state"] == "installed"