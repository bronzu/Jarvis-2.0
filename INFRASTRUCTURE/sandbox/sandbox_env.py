from pathlib import Path


class SandboxEnvironment:
    def __init__(self, base_directory=None):
        base = Path(base_directory) if base_directory else Path("/tmp/jarvis_sandbox")
        self.base_directory = base.resolve()
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative_path):
        target = (self.base_directory / relative_path).resolve()
        if self.base_directory not in target.parents and target != self.base_directory:
            raise PermissionError("Percorso fuori dalla sandbox.")
        return target

    def read_file(self, relative_path):
        return self._safe_path(relative_path).read_text(encoding="utf-8")

    def write_file(self, relative_path, content, approved=True):
        if approved is not True:
            raise PermissionError("Scrittura non autorizzata")
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
