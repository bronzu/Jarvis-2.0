import re


class PermissionGuard:
    DANGEROUS_PATTERNS = [
        r"os\.system\s*\(",
        r"subprocess\s*\.\s*(run|Popen|call|check_output|check_call|check_output)",
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"open\s*\([^,]*,\s*['\"]w",
        r"shutil\.(rmtree|move|copy)",
        r"os\.(remove|unlink|rmdir|removedirs|mkdir|chmod|chown|symlink|link)",
        r"pathlib.*\.(write_text|write_bytes|unlink|rmdir|chmod|symlink_to|hardlink_to)",
        r"\.system\s*\(",
        r"\.popen\s*\(",
        r"\.call\s*\(",
        r">\s*/",  # Redirection to root
        r">\s*~/",  # Redirection to home
        r">\s*\.\./",  # Redirection with traversal
        r"\|",  # Pipe
    ]

    BLOCKED_COMMANDS = [
        "rm -rf",
        "rm -f",
        "dd if=",
        "mkfs",
        "fdisk",
        "kill -9",
        "pkill",
        "killall",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        ":(){ :|:& };:",
        "curl | sh",
        "wget | sh",
        "python -c",
        "python3 -c",
    ]

    def require_user_approval(self, approved):
        if approved is not True:
            raise PermissionError("È richiesta l'approvazione esplicita dell'utente")
        return True

    def grant_self_permission(self):
        raise PermissionError("JARVIS non può concedersi autorizzazioni")

    def validate_command(self, command, approved=False):
        if not isinstance(command, str):
            raise TypeError("Il comando deve essere una stringa.")

        normalized = command.strip()
        if not normalized:
            raise ValueError("Il comando non può essere vuoto.")

        normalized_lower = normalized.lower()

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                raise PermissionError("Comando pericoloso bloccato dal permission guard.")

        for blocked in self.BLOCKED_COMMANDS:
            if blocked.lower() in normalized_lower:
                raise PermissionError(f"Comando bloccato: {blocked}")

        if approved is not True:
            raise PermissionError("È richiesta l'approvazione esplicita dell'utente")
        return True
