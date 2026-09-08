import os
import subprocess
from pathlib import Path
from executive_system.capability_registry import CapabilityRegistry
from executive_system.tool_selector import ToolSelector
from executive_system.tools.filesystem_tool import FilesystemTool
from executive_system.tools.terminal_tool import TerminalTool
from executive_system.tools.python_runner import PythonRunner
from executive_system.tools.web_tool import WebBrowserTool
from executive_system.tools.internet_tool import InternetTool
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from INFRASTRUCTURE.sandbox.sandbox_env import SandboxEnvironment


class ToolSystem:
    def __init__(self, security_layer=None, base_directory=None, workdir=None, vision_system=None):
        self.security_layer = security_layer or SecurityLayer()
        self.registry = CapabilityRegistry()
        self.selector = ToolSelector(self.registry)
        self.base_directory = Path(base_directory or os.getcwd()).resolve()
        self.workdir = Path(workdir or "/tmp/jarvis-workspace").resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.sandbox = SandboxEnvironment(self.workdir)
        self.vision_system = vision_system
        self._register_default_tools()

    def _register_default_tools(self):
        self.registry.register_tool("filesystem", self._create_filesystem_tool)
        self.registry.register_tool("terminal", self._create_terminal_tool)
        self.registry.register_tool("python", self._create_python_runner)
        self.registry.register_tool("web", self._create_web_tool)
        self.registry.register_tool("internet", self._create_internet_tool)
        if self.vision_system is not None:
            self.registry.register_tool("vision", self._create_vision_tool)

    def _create_filesystem_tool(self):
        return FilesystemTool(self.base_directory)

    def _create_terminal_tool(self):
        return TerminalTool(
            workdir=self.workdir,
            permission_guard=self.security_layer.permission_guard,
        )

    def _create_python_runner(self):
        return PythonRunner()

    def _create_web_tool(self):
        return WebBrowserTool()

    def _create_internet_tool(self):
        return InternetTool()

    def _create_vision_tool(self):
        return self.vision_system

    def select(self, capability):
        factory = self.registry.get_factory(capability)
        if factory is None:
            return None
        return factory()

    def execute(self, capability, *args, approved=False, session_id=None, **kwargs):
        tool = self.select(capability)
        if tool is None:
            raise ValueError(f"Tool '{capability}' non disponibile")
        self.security_layer.check_tool_execution(capability, kwargs, approved=approved, session_id=session_id)
        if capability == "filesystem":
            operation = kwargs.get("operation")
            path = kwargs.get("path")
            if operation in ("write", "delete", "create"):
                self.security_layer.check_filesystem_access(path, operation, approved=approved, session_id=session_id)
            if operation == "read":
                return tool.read(path)
            elif operation == "write":
                return tool.write(path, kwargs.get("content", ""), approved=approved)
            elif operation == "list":
                return os.listdir(tool._path(kwargs.get("path", ".")))
            else:
                raise ValueError(f"Operazione filesystem '{operation}' non supportata")
        elif capability == "terminal":
            command = kwargs.get("command")
            self.security_layer.check_code_execution(command, approved=approved, session_id=session_id)
            return tool.run(command, self._subprocess_runner, approved=approved)
        elif capability == "python":
            code = kwargs.get("code")
            self.security_layer.check_code_execution(code, approved=approved, session_id=session_id)
            if not hasattr(tool, "run_isolated"):
                raise RuntimeError(
                    "PythonRunner is missing run_isolated(): the production "
                    "execution path requires OS-level sandboxing. Refusing "
                    "to fall back to in-process execution."
                )
            return tool.run_isolated(code, approved=approved)
        elif capability == "web":
            query = kwargs.get("query")
            self.security_layer.check_network_access(f"https://duckduckgo.com/html/?q={query}", approved=approved, session_id=session_id)
            return tool.search(query, limit=kwargs.get("limit", 3))
        elif capability == "internet":
            url = kwargs.get("url")
            self.security_layer.check_network_access(url, approved=approved, session_id=session_id)
            if kwargs.get("extract_text"):
                return tool.extract_text(url)
            return tool.fetch(url)
        elif capability == "vision":
            operation = kwargs.get("operation")
            if operation == "analyze_image":
                path = kwargs.get("path")
                if path:
                    self.security_layer.check_filesystem_access(path, "read", approved=approved, session_id=session_id)
                return tool.analyze_image(path)
            elif operation == "capture_screen":
                return tool.capture_screen()
            elif operation == "analyze_screenshot":
                return tool.analyze_screenshot()
            elif operation == "extract_text":
                path = kwargs.get("path")
                if path:
                    self.security_layer.check_filesystem_access(path, "read", approved=approved, session_id=session_id)
                return tool.extract_text(path)
            else:
                raise ValueError(f"Operazione vision '{operation}' non supportata")
        else:
            raise ValueError(f"Esecuzione per '{capability}' non implementata")

    @staticmethod
    def _subprocess_runner(command: str) -> str:
        """Safe subprocess runner for terminal commands.

        Uses ``subprocess.run`` with ``shell=True`` only after the
        command has passed all security checks in ``TerminalTool``.
        Output is captured as text.  The working directory is inherited
        from the calling process; ``TerminalTool`` already restricts
        paths via its own ``_safe_path`` and ``_validate_command_safety``
        gates.
        """
        if not isinstance(command, str):
            raise TypeError("command must be a string")
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return output.strip()

