from pathlib import Path


class FilesystemTool:
    def __init__(self, base_directory): self.base_directory = Path(base_directory).resolve()
    def _path(self, relative_path):
        path = (self.base_directory / relative_path).resolve()
        if self.base_directory not in path.parents and path != self.base_directory: raise PermissionError("Percorso fuori sandbox")
        return path
    def read(self, relative_path): return self._path(relative_path).read_text()
    def write(self, relative_path, content, approved=False):
        if not approved: raise PermissionError("Scrittura non autorizzata")
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)
