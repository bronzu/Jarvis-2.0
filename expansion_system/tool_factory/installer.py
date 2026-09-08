import logging
import time
from typing import Any, Dict, Optional


logger = logging.getLogger("JARVIS.Installer")


class Installer:
    """Installa tool candidati nella sandbox e li promuove al registry con approvazione."""

    def __init__(self, registry=None, audit_system=None):
        self.registry = registry
        self.audit_system = audit_system

    def install(self, candidate: Dict[str, Any], approved: bool = False, session_id: Optional[str] = None) -> Dict[str, Any]:
        if not approved:
            raise PermissionError("Installazione non autorizzata")
        if candidate.get("state") != "sandbox":
            raise ValueError("Candidato non verificabile")

        installed = dict(candidate)
        installed["state"] = "installed"
        installed["installed_at"] = time.time()

        if self.registry is not None:
            try:
                self.registry.register_tool(candidate["name"], lambda: installed)
            except Exception as exc:
                logger.error("Registry registration failed for %s: %s", candidate.get("name"), exc)

        if self.audit_system is not None:
            try:
                self.audit_system.log_event(
                    "TOOL_EXECUTION_END",
                    component="Installer",
                    action="install",
                    status="success",
                    metadata={"tool": candidate.get("name")},
                    session_id=session_id,
                )
            except Exception as exc:
                logger.error("Audit logging failed for %s: %s", candidate.get("name"), exc)

        return installed

    def validate_before_install(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue controlli base prima dell'installazione."""
        if not isinstance(candidate, dict):
            raise TypeError("candidate deve essere un dict")
        if not candidate.get("name"):
            return {"ok": False, "reason": "Nome mancante"}
        if not candidate.get("source"):
            return {"ok": False, "reason": "Sorgente mancante"}
        return {"ok": True, "reason": None}
