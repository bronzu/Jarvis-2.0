import ast
import os
import textwrap
from typing import Any, Dict, Optional


class ToolFactory:
    """Factory dei tool: registra, crea candidati e genera codice nella sandbox."""

    _tool_registry = {}

    def __init__(self):
        self._tool_registry = dict(self._tool_registry)

    def register(self, name, factory_callable):
        if not name or not callable(factory_callable):
            raise ValueError("Nome e callable sono obbligatori")
        self._tool_registry[name] = factory_callable
        return factory_callable

    def create_candidate(self, name, source):
        if not name or not isinstance(source, str) or not source.strip():
            raise ValueError("Nome e sorgente sono obbligatori")
        return {"name": name, "source": source, "state": "sandbox"}

    def create_tool(self, name, *args, **kwargs):
        if name not in self._tool_registry:
            raise KeyError(f"Tool non registrata: {name}")
        return self._tool_registry[name](*args, **kwargs)

    @classmethod
    def register_builtin_tools(cls):
        from executive_system.tools.web_tool import WebBrowserTool
        cls._tool_registry["web_browser"] = WebBrowserTool
        return cls._tool_registry

    def generate(self, tool_spec: Dict[str, Any], sandbox: Optional[Any] = None) -> Optional[Any]:
        """Genera codice Python per un nuovo tool e lo scrive nella sandbox se disponibile.

        Args:
            tool_spec: dict con 'name', 'description', 'methods'.
            sandbox: SandboxEnvironment opzionale.

        Returns:
            Path del file generato o None.
        """
        name = tool_spec.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("tool_spec.name è obbligatorio")

        methods = tool_spec.get("methods", [])
        if not methods:
            raise ValueError("tool_spec.methods non può essere vuoto")

        imports = {"os", "re", "typing"}
        for method in methods:
            try:
                tree = ast.parse(method.get("implementation", ""))
            except Exception:
                raise ValueError(f"Implementazione non valida per {method.get('name')}")

        class_name = "".join(part.capitalize() for part in str(name).strip().split("_")) + "Tool"
        lines = [
            "import os",
            "import re",
            "from typing import Any, Dict, Optional",
            "",
            "",
            f"class {class_name}:",
            f'    """{tool_spec.get("description", "Tool generato automaticamente")}"""',
            "",
            "    def __init__(self):",
            "        pass",
            "",
        ]

        for method in methods:
            method_name = method.get("name")
            params = method.get("params", [])
            implementation = method.get("implementation", "pass")
            param_list = ", ".join(["self"] + list(params))
            lines.append(f"    def {method_name}({param_list}):")
            lines.append(f'        """{method_name} generata automaticamente"""')
            lines.append(textwrap.indent(implementation, "        "))
            lines.append("")

        code = "\n".join(lines)

        if sandbox is not None:
            try:
                file_path = sandbox.write_file(f"{name}.py", code)
                return file_path
            except Exception:
                return None

        return code

    def validate_generated(self, code: str) -> Dict[str, Any]:
        """Verifica che il codice generato sia sintatticamente valido e non contenga pattern pericolosi."""
        dangerous = ["__import__", "os.system", "subprocess", "eval(", "exec(", "compile("]
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"valid": False, "error": f"SyntaxError: {e}"}

        for pattern in dangerous:
            if pattern in code:
                return {"valid": False, "error": f"Pattern pericoloso rilevato: {pattern}"}

        return {"valid": True, "error": None}
