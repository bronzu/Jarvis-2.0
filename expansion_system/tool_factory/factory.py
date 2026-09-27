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

        class_name = "".join(part.capitalize() for part in str(name).strip().split("_")) + "Tool"
        lines = [
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
        """Verifica che il codice generato sia sintatticamente valido e non contenga pattern pericolosi.

        Uses AST-based detection to resist string-concatenation obfuscation.
        """
        dangerous = ["__import__", "os.system", "subprocess", "eval(", "exec(", "compile("]
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"valid": False, "error": f"SyntaxError: {e}"}

        # Substring check (first line of defense)
        for pattern in dangerous:
            if pattern in code:
                return {"valid": False, "error": f"Pattern pericoloso rilevato: {pattern}"}

        # AST-based check: detect dangerous calls and imports regardless of
        # string obfuscation (e.g. getattr(__builtins__, '__imp'+'ort__'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in ("os", "subprocess", "sys", "shutil", "ctypes",
                               "marshal", "pickle", "shelve", "commands", "pty"):
                        return {"valid": False, "error": f"Import non consentito: {alias.name}"}
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in ("os", "subprocess", "sys",
                    "shutil", "ctypes", "marshal", "pickle", "shelve", "commands", "pty"):
                    return {"valid": False, "error": f"Import non consentito: {node.module}"}
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id == "os":
                        if func.attr in ("system", "popen", "spawn", "unlink", "remove",
                                        "chmod", "chown", "execv", "execve"):
                            return {"valid": False, "error": f"Chiamata os non consentita: os.{func.attr}"}
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        return {"valid": False, "error": "Chiamata subprocess non consentita"}
                    if func.attr == "system" and isinstance(func.value, ast.Name) and func.value.id == "os":
                        return {"valid": False, "error": "Chiamata os.system non consentita"}
                if isinstance(func, ast.Name) and func.id in ("eval", "exec", "compile", "__import__"):
                    return {"valid": False, "error": f"Funzione pericolosa: {func.id}"}
                # Block getattr to prevent indirect access to dangerous functions
                if isinstance(func, ast.Name) and func.id == "getattr":
                    return {"valid": False, "error": "Chiamata getattr non consentita"}
            elif isinstance(node, ast.Subscript):
                # Block __builtins__[...] access
                if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
                    return {"valid": False, "error": "Accesso a __builtins__ non consentito"}

        return {"valid": True, "error": None}
