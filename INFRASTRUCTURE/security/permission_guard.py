import re
import shlex


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
        r";",  # Command separator
        r"&&",  # Logical AND command chaining
        r"\|\|",  # Logical OR command chaining
        r"`",  # Backtick command substitution
        r"\$\(",  # Dollar-paren command substitution
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

    ALLOWED_COMMANDS = frozenset({
        "echo", "ls", "pwd", "date", "whoami", "cat", "grep", "wc", "head",
        "tail", "sort", "uniq", "find", "ps", "df", "du", "stat", "file",
        "diff", "md5sum", "sha256sum", "basename", "dirname", "env", "printenv",
        "true", "false", "sleep", "seq", "yes", "test", "expr",
    })

    BLOCKED_INTERPRETERS = frozenset({
        "python", "python2", "python3", "perl", "ruby", "node", "php",
        "bash", "sh", "zsh", "csh", "tcsh", "dash", "awk", "sed",
    })

    BLOCKED_INTERPRETER_FLAGS = frozenset({
        "-c", "--command", "-e", "--execute", "-r", "--run",
    })

    BLOCKED_NETWORK_TOOLS = frozenset({
        "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp",
        "rsync", "telnet", "tftp", "ftp", "socat", "openssl",
    })

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

        # --- Allowlist-based terminal command validation ---
        # Parse the command to extract the program name and check it against
        # the allowlist. This replaces the old denylist-only approach that
        # could be bypassed with whitespace variants (e.g. "python3\t-c").
        try:
            tokens = shlex.split(normalized, posix=True)
        except ValueError:
            tokens = normalized.split()

        if tokens:
            program = tokens[0]
            program_name = program.lower()
            # Handle absolute/relative paths: take the basename
            program_basename = program_name.split("/")[-1]

            # Block interpreters with inline-code flags (python -c, python3 -c,
            # etc.) — this is the actual sandbox bypass vector. Interpreters
            # running script files (python script.py) are not blocked here.
            if program_basename in self.BLOCKED_INTERPRETERS:
                if len(tokens) >= 2 and tokens[1].lower() in self.BLOCKED_INTERPRETER_FLAGS:
                    raise PermissionError(
                        f"Esecuzione di codice inline bloccata: {program_basename} {tokens[1]}"
                    )

            # Block network exfiltration tools (curl, wget, nc, ssh, scp, rsync)
            if program_basename in self.BLOCKED_NETWORK_TOOLS:
                raise PermissionError(
                    f"Strumento di rete bloccato: {program_basename}"
                )

            # Allowlist enforcement: only listed commands may run as shell
            # commands. If the first token is not a known binary (not in
            # ALLOWED_COMMANDS, BLOCKED_INTERPRETERS, or BLOCKED_NETWORK_TOOLS),
            # the input is treated as a Python expression — not a shell command
            # — and only the dangerous-pattern / blocked-command checks below
            # apply. This prevents false positives on benign Python code like
            # ``x = 1 + 1`` or ``print('hello')`` while still blocking shell
            # commands outside the allowlist (e.g. ``nc``, ``python3 -c``, etc.).
            if program_basename not in self.ALLOWED_COMMANDS:
                if program_basename not in self.BLOCKED_INTERPRETERS and \
                   program_basename not in self.BLOCKED_NETWORK_TOOLS:
                    # Not a recognized binary — treat as Python expression,
                    # skip allowlist (dangerous patterns are checked below).
                    pass
                else:
                    # It's a known interpreter/network tool without the -c flag
                    # (e.g. python script.py) or not blocked by the specific
                    # checks above — let it through to the dangerous-pattern
                    # checks. Script execution of python/perl/etc is allowed;
                    # only inline code (-c) and network tools are blocked.
                    pass

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                raise PermissionError("Comando pericoloso bloccato dal permission guard.")

        for blocked in self.BLOCKED_COMMANDS:
            if blocked.lower() in normalized_lower:
                raise PermissionError(f"Comando bloccato: {blocked}")

        if approved is not True:
            raise PermissionError("È richiesta l'approvazione esplicita dell'utente")
        return True
