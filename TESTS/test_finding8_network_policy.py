"""Tests for Finding 8: NetworkPolicy host allowlist bypass via substring matching.

The NetworkPolicy.check_url() used substring matching (`h in host`) for host
allowlist checking. This allowed bypass: a host like `evil-api.example.com`
would be allowed when only `api.example.com` was in the allowlist, because
`api.example.com` is a substring of `evil-api.example.com`.

Fix: Replaced substring matching with domain suffix matching using
`_host_matches()` that only allows exact matches or proper subdomains.
"""
from INFRASTRUCTURE.network_policy import NetworkPolicy


def test_allowlist_exact_match_allowed():
    p = NetworkPolicy(enabled=True, allow_https=True, allowed_hosts=["api.example.com"])
    r = p.check_url("https://api.example.com/data")
    assert r["allowed"] is True


def test_allowlist_subdomain_allowed():
    p = NetworkPolicy(enabled=True, allow_https=True, allowed_hosts=["example.com"])
    r = p.check_url("https://api.example.com/data")
    assert r["allowed"] is True


def test_allowlist_evil_prefix_blocked():
    p = NetworkPolicy(enabled=True, allow_https=True, allowed_hosts=["api.example.com"])
    r = p.check_url("https://evil-api.example.com/data")
    assert r["allowed"] is False
    assert r["reason"] == "host_not_in_allowlist"


def test_allowlist_evil_suffix_blocked():
    p = NetworkPolicy(enabled=True, allow_https=True, allowed_hosts=["api.example.com"])
    r = p.check_url("https://notapi.example.com/data")
    assert r["allowed"] is False
    assert r["reason"] == "host_not_in_allowlist"


def test_blocklist_exact_match_blocked():
    p = NetworkPolicy(enabled=True, allow_https=True, blocked_hosts=["evil.com"])
    r = p.check_url("https://evil.com/data")
    assert r["allowed"] is False
    assert "blocked" in r["reason"]


def test_blocklist_subdomain_blocked():
    p = NetworkPolicy(enabled=True, allow_https=True, blocked_hosts=["evil.com"])
    r = p.check_url("https://api.evil.com/data")
    assert r["allowed"] is False
    assert "blocked" in r["reason"]


def test_blocklist_not_subdomain_allowed():
    p = NetworkPolicy(enabled=True, allow_https=True, blocked_hosts=["evil.com"])
    r = p.check_url("https://notevil.com/data")
    assert r["allowed"] is True


def test_host_matches_helper():
    assert NetworkPolicy._host_matches("example.com", "example.com") is True
    assert NetworkPolicy._host_matches("example.com", "api.example.com") is True
    assert NetworkPolicy._host_matches("example.com", "notexample.com") is False
    assert NetworkPolicy._host_matches("example.com", "example.com.evil.com") is False
    assert NetworkPolicy._host_matches("", "") is False
