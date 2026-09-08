"""
Tests for network policy.
"""

import pytest

from INFRASTRUCTURE.network_policy import NetworkPolicy


@pytest.fixture
def policy():
    return NetworkPolicy(
        enabled=True,
        allow_https=True,
        allow_http=False,
        allowed_hosts=["example.com"],
        blocked_hosts=["evil.com"],
        allowed_ports=[443, 80],
        blocked_ports=[22],
        timeout=5,
        allow_redirects=True,
    )


class TestNetworkPolicy:
    def test_disabled_policy_allows_all(self):
        policy = NetworkPolicy(enabled=False)
        result = policy.check_url("http://any.host/path")
        assert result["allowed"] is True

    def test_https_allowed(self, policy):
        result = policy.check_url("https://example.com/path")
        assert result["allowed"] is True
        assert result["protocol"] == "https"

    def test_http_blocked(self, policy):
        result = policy.check_url("http://example.com/path")
        assert result["allowed"] is False
        assert result["reason"] == "http_not_allowed"

    def test_blocked_host(self, policy):
        result = policy.check_url("https://evil.com/path")
        assert result["allowed"] is False
        assert "blocked_host" in result["reason"]

    def test_host_not_in_allowlist(self, policy):
        result = policy.check_url("https://other.com/path")
        assert result["allowed"] is False
        assert result["reason"] == "host_not_in_allowlist"

    def test_blocked_port(self, policy):
        result = policy.check_url("https://example.com:22/path")
        assert result["allowed"] is False
        assert "blocked_port" in result["reason"]

    def test_invalid_url(self, policy):
        result = policy.check_url("ftp://example.com/path")
        assert result["allowed"] is False
        assert result["reason"] == "unsupported_protocol"

    def test_invalid_input(self, policy):
        result = policy.check_url("")
        assert result["allowed"] is False
        result = policy.check_url(None)
        assert result["allowed"] is False

    def test_summary(self, policy):
        summary = policy.summary()
        assert summary["enabled"] is True
        assert summary["allow_https"] is True
        assert summary["timeout"] == 5
