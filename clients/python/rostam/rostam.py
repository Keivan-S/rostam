"""The unified Rostam client: one entry point, transport chosen from the target."""
from typing import Optional, Tuple
from urllib.parse import urlsplit

from ._types import TransportError


def _parse_target(target: str) -> Tuple[str, str, int, str]:
    """Return (kind, host, port, base_url). kind is 'http' or 'tcp'.
    http/https scheme -> HTTP; tcp:// or no scheme -> TCP. A bare host with no
    port is an error (the HTTP vs TCP default ports differ, so guessing is unsafe)."""
    if "://" in target:
        parts = urlsplit(target)
        scheme = parts.scheme.lower()
        if scheme in ("http", "https"):
            return "http", parts.hostname or "", parts.port or (443 if scheme == "https" else 80), target
        if scheme == "tcp":
            host, port = parts.hostname or "", parts.port
            if not port:
                raise TransportError("tcp:// target requires an explicit port, e.g. tcp://host:7000")
            return "tcp", host, port, ""
        raise TransportError(f"unknown target scheme {scheme!r}; use http://, https://, or tcp://")
    # bare host:port -> TCP (the native protocol is the default path)
    if ":" not in target:
        raise TransportError(f"target {target!r} needs a port (tcp) or an http:// scheme")
    host, _, port_s = target.rpartition(":")
    try:
        port = int(port_s)
    except ValueError:
        raise TransportError(f"invalid port in target {target!r}")
    return "tcp", host, port, ""


class Rostam:
    """Unified Rostam client. Transport is selected from `target`:
    `http(s)://host:8080` -> REST; `tcp://host:7000` or bare `host:7000` -> the
    binary TCP protocol. Vector ops are flat (`r.search`, `r.hybrid_text`, ...);
    KV lives under `r.kv.*` (TCP only)."""

    def __init__(self, target: str, *, api_key: Optional[str] = None,
                 auth_token: Optional[str] = None, timeout: float = 30.0):
        kind, host, port, base_url = _parse_target(target)
        self._kind = kind
        token = auth_token if auth_token is not None else api_key
        # Backends wired in Tasks 3 & 5; kv wired in Task 4.
        self._t = None  # set by _connect
        self._connect(kind, host, port, base_url, token, timeout)

    def _connect(self, kind, host, port, base_url, token, timeout):
        raise NotImplementedError  # replaced in Tasks 3/5
