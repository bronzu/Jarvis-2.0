import os
import re
import shlex
from pathlib import Path

from INFRASTRUCTURE.security.permission_guard import PermissionGuard


_CONNETTI_PATTERN = re.compile(r"^\[CONNETTI:\s*(.+?)\]\s*$")


class TerminalTool:
    """
    Gateway senza shell implicita con isolamento rigoroso.
    
    Garantisce:
    1. Esecuzione solo dentro directory dedicata (chroot-like)
    2. Blocco di accessi a file di sistema interni di JARVIS
    3. Blocco di symlink escapes
    4. Logging di tentate violazioni
    
    Collega la catena operativa:
    NeuralCore → CognitiveCoordinator → router.py → internet_tool.py → Sandbox → ContextManager
    """
    
    # File di sistema critici di JARVIS che non devono essere accessibili
    JARVIS_INTERNAL_PATHS = [
        "/cognitive_core/",
        "/executive_system/",
        "/neural_core/",
        "/expansion_system/",
        "/interface_layer/",
        "/CORE/",
        "/INFRASTRUCTURE/",
        "/INFRASTRUTTURA/",
    ]
    
    # Pattern di comandi pericolosi anche dentro la sandbox
    DANGEROUS_COMMAND_PATTERNS = [
        r"\.\.\/",  # Path traversal
        r"~",  # Home directory escape
        r"symlink|ln\s+-s",  # Symlink creation
        r"mount|umount",  # Filesystem operations
        r"chroot",  # Chroot escape attempts
        r"rm\s+-rf\s+/",  # Destructive root deletion
        r"rm\s+-rf\s+~",  # Destructive home deletion
        r"rm\s+-rf\s+\.",  # Destructive current dir deletion
        r"dd\s+if=",  # Raw disk writes
        r"mkfs",  # Filesystem formatting
        r"shutdown|reboot|poweroff",  # System control
        r":(){ :|:& };:",  # Fork bomb
        r">\s*/",  # Redirection to root paths
        r">\s*~/",  # Redirection to home
        r">\s*\.\./",  # Redirection with traversal
        r"\|",  # Pipe
        r";\s*rm\s",  # Command chaining with rm
        r";\s*shutdown",  # Command chaining with shutdown
        r"&&\s*rm\s",  # Command chaining with rm
        r"\|\|\s*rm\s",  # Command chaining with rm
    ]

    def __init__(self, workdir=None, permission_guard=None,
                 neural_core=None, coordinator=None, router=None,
                 internet_tool=None, sandbox=None, context_manager=None):
        self.workdir = Path(workdir or "/tmp/jarvis-workspace").resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.permission_guard = permission_guard or PermissionGuard()
        self._violation_log = []

        self.neural_core = neural_core
        self.coordinator = coordinator
        self.router = router
        self.internet_tool = internet_tool
        self.sandbox = sandbox
        self.context_manager = context_manager

    def _safe_path(self, target):
        """
        Verifica che il target sia dentro workdir (chroot-like).
        Risolve symlink e path traversal.
        """
        if not isinstance(target, (str, Path)):
            raise TypeError("Target deve essere string o Path")
        
        target_path = Path(target) if isinstance(target, str) else target
        
        # Risolvi il path per eliminare .. e .
        try:
            candidate = (self.workdir / target_path).resolve()
        except (RuntimeError, OSError) as e:
            self._log_violation(f"Path resolution failed: {target}", "path_error")
            raise PermissionError(f"Impossibile risolvere il percorso: {target}")
        
        # Verifica che candidate sia dentro workdir o uguale a workdir
        try:
            candidate.relative_to(self.workdir)
        except ValueError:
            self._log_violation(f"Path escape attempt: {candidate}", "escape_attempt")
            raise PermissionError("Il comando è fuori dalla directory di lavoro autorizzata.")
        
        return candidate

    def _validate_command_safety(self, command):
        """
        Verifica che il comando non contenga pattern pericolosi.

        Usa un'allowlist di comandi ammessi. Qualsiasi comando il cui
        nome programma non compare nell'allowlist viene rifiutato.
        Inoltre vengono bloccati esplicitamente interpreti e tool di rete.
        """
        if not isinstance(command, str):
            raise TypeError("Il comando deve essere una stringa.")

        # Parse the command to extract the program name
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            tokens = command.split()

        if tokens:
            program = tokens[0].lower()
            program_basename = program.split("/")[-1]

            # Block interpreters (python, python3, perl, node, php, etc.)
            for name in ("python", "python2", "python3", "perl", "ruby",
                         "node", "php", "bash", "sh", "zsh", "csh", "tcsh",
                         "dash", "awk", "sed"):
                if program_basename == name:
                    self._log_violation(
                        f"Interpreter invocation blocked: {program_basename}",
                        "interpreter_block",
                    )
                    raise PermissionError(
                        f"Invocazione di interpreti bloccata: {program_basename}"
                    )

            # Block network exfiltration tools (curl, wget, nc, ssh, scp, rsync)
            for name in ("curl", "wget", "nc", "ncat", "netcat", "ssh",
                         "scp", "sftp", "rsync", "telnet", "tftp", "ftp",
                         "socat", "openssl"):
                if program_basename == name:
                    self._log_violation(
                        f"Network tool invocation blocked: {program_basename}",
                        "network_tool_block",
                    )
                    raise PermissionError(
                        f"Strumento di rete bloccato: {program_basename}"
                    )

            # Allowlist enforcement
            allowed = set(self.permission_guard.ALLOWED_COMMANDS)
            if program_basename not in allowed:
                self._log_violation(
                    f"Command not in allowlist: {program_basename}",
                    "allowlist_violation",
                )
                raise PermissionError(
                    f"Comando non autorizzato (non nella allowlist): {program_basename}"
                )

        # Defense-in-depth: still check denylist patterns
        for pattern in self.DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                self._log_violation(
                    f"Dangerous pattern detected: {pattern} in {command}",
                    "dangerous_pattern",
                )
                raise PermissionError(f"Comando contiene pattern pericoloso: {pattern}")

    def _check_jarvis_internal_access(self, command):
        """
        Verifica che il comando non tenti di accedere ai percorsi interni di JARVIS.
        """
        normalized = command.lower()
        for internal_path in self.JARVIS_INTERNAL_PATHS:
            if internal_path.lower() in normalized:
                self._log_violation(f"JARVIS internal access attempt: {internal_path}", "internal_access")
                raise PermissionError(f"Accesso negato ai percorsi interni di JARVIS: {internal_path}")

    def _log_violation(self, message, violation_type):
        """Registra i tentativi di violazione per audit."""
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "type": violation_type,
        }
        self._violation_log.append(log_entry)

    def get_violation_log(self):
        """Ritorna il log delle violazioni tentate."""
        return list(self._violation_log)

    def _is_connecti_command(self, command):
        if not isinstance(command, str):
            return False, None
        match = _CONNETTI_PATTERN.match(command.strip())
        if match:
            return True, match.group(1).strip()
        return False, None

    def _route_connecti(self, query):
        if self.router is not None:
            try:
                routed = self.router.route(f"[CONNETTI: {query}]")
                if isinstance(routed, dict):
                    route = routed.get("route", "local")
                else:
                    route = "local"
            except Exception:
                route = "local"
        else:
            route = "local"

        if self.coordinator is not None:
            try:
                processed = self.coordinator.process(query)
                if isinstance(processed, dict):
                    prompt = processed.get("default_response") or str(processed)
                else:
                    prompt = str(processed)
            except Exception:
                prompt = query
        else:
            prompt = query

        if self.internet_tool is None:
            raise RuntimeError("internet_tool non configurato per il comando [CONNETTI]")

        text = self.internet_tool.extract_text(self.internet_tool.fetch(prompt))

        if self.sandbox is not None:
            try:
                self.sandbox.write_file("connecti_result.txt", text, approved=True)
            except Exception:
                pass

        if self.context_manager is not None:
            try:
                self.context_manager.add_turn(f"[CONNETTI: {query}]", {"tool": "terminal", "route": route})
            except Exception:
                pass

        return {
            "route": route,
            "query": query,
            "response": text,
            "source": "internet_tool",
        }

    def run(self, command, runner, approved=False, target_file=None):
        """
        Esegue il comando con isolazione rigorosa.
        
        Args:
            command: Comando da eseguire
            runner: Callable che esegue il comando
            approved: Richiede approvazione esplicita (deve essere True)
            target_file: File target opzionale per il comando
        
        Returns:
            Risultato dell'esecuzione
        """
        if approved is not True:
            self._log_violation(f"Unapproved command execution attempt: {command}", "unapproved_execution")
            raise PermissionError("Comando non autorizzato")
        
        is_connecti, query = self._is_connecti_command(command)
        if is_connecti:
            return self._route_connecti(query)
        
        if not callable(runner):
            raise TypeError("runner deve essere richiamabile")
        
        self._validate_runner(runner)
        
        # Validazione security del permission guard
        self.permission_guard.validate_command(command, approved=True)
        
        # Validazione ulteriore: comandi pericolosi
        self._validate_command_safety(command)
        
        # Verifica accessi a percorsi interni di JARVIS
        self._check_jarvis_internal_access(command)
        
        # Se specificato, valida il target file
        if target_file is not None:
            safe_path = self._safe_path(target_file)
            # Verifica che il path sia davvero dentro workdir
            assert safe_path.parent == self.workdir or self.workdir in safe_path.parents
        
        # Esegui il comando in ambiente isolato
        try:
            result = runner(command)
            return result
        except Exception as e:
            self._log_violation(f"Command execution error: {command}: {str(e)}", "execution_error")
            raise

    def _validate_runner(self, runner):
        """Verifica che il runner non sia una funzione pericolosa."""
        if runner in (os.system, os.popen):
            raise PermissionError(f"Runner pericoloso non consentito: {runner.__name__}")
