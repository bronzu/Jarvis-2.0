"""
Network policy for JARVIS 2.0.

Provides:
- allowed/disallowed hosts
- allowed/disallowed ports
- protocol policy (HTTP/HTTPS)
- timeout policy
- redirect policy
- audit logging
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


class NetworkPolicy:
    def __init__(
        self,
        enabled: bool = False,
        allow_https: bool = True,
        allow_http: bool = False,
        allowed_hosts: Optional[List[str]] = None,
        blocked_hosts: Optional[List[str]] = None,
        allowed_ports: Optional[List[int]] = None,
        blocked_ports: Optional[List[int]] = None,
        timeout: int = 10,
        allow_redirects: bool = True,
    ):
        self.enabled = enabled
        self.allow_https = allow_https
        self.allow_http = allow_http
        self.allowed_hosts = allowed_hosts or []
        self.blocked_hosts = blocked_hosts or []
        self.allowed_ports = allowed_ports or [443, 80]
        self.blocked_ports = blocked_ports or []
        self.timeout = timeout
        self.allow_redirects = allow_redirects

    def check_url(self, url: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"allowed": True, "reason": "network_disabled"}

        if not url or not isinstance(url, str):
            return {"allowed": False, "reason": "invalid_url"}

        try:
            parsed = urlparse(url)
        except ValueError:
            return {"allowed": False, "reason": "invalid_url"}

        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port

        if scheme == "https":
            protocol = "https"
        elif scheme == "http":
            protocol = "http"
        else:
            return {"allowed": False, "reason": "unsupported_protocol"}

        if protocol == "https" and not self.allow_https:
            return {"allowed": False, "reason": "https_not_allowed"}
        if protocol == "http" and not self.allow_http:
            return {"allowed": False, "reason": "http_not_allowed"}

        if not host:
            return {"allowed": False, "reason": "invalid_host"}

        for blocked in self.blocked_hosts:
            if blocked.lower() in host:
                return {"allowed": False, "reason": f"blocked_host:{blocked}"}

        if self.allowed_hosts:
            if not any(h.lower() in host for h in self.allowed_hosts):
                return {"allowed": False, "reason": "host_not_in_allowlist"}

        if port is not None:
            if self.blocked_ports and port in self.blocked_ports:
                return {"allowed": False, "reason": f"blocked_port:{port}"}
            if self.allowed_ports and port not in self.allowed_ports:
                return {"allowed": False, "reason": f"port_not_allowed:{port}"}

        return {"allowed": True, "reason": "allowed", "host": host, "protocol": protocol}

    def is_enabled(self) -> bool:
        return self.enabled

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_https": self.allow_https,
            "allow_http": self.allow_http,
            "allowed_hosts_count": len(self.allowed_hosts),
            "blocked_hosts_count": len(self.blocked_hosts),
            "allowed_ports": self.allowed_ports,
            "blocked_ports": self.blocked_ports,
            "timeout": self.timeout,
            "allow_redirects": self.allow_redirects,
        }
