"""The unified Rostam client: one entry point, transport chosen from the target."""
from typing import Optional, Tuple
from urllib.parse import urlsplit

from . import _http, _tcp
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
        if kind == "tcp":
            self._t = _tcp.TcpTransport(host, port, token, timeout)
            return
        if kind == "http":
            self._t = _http.HttpTransport(base_url, token, timeout)
            return
        raise TransportError(f"unknown transport kind {kind!r}")

    # ---- flat vector API -----------------------------------------------
    #
    # The union of what both transport backends serve (create_collection,
    # upsert, insert, upsert_batch, delete, get, get_batch, scroll, search,
    # search_docs, search_groups, hybrid_search, hybrid_text, recommend,
    # query, exists) — each method just forwards to the active backend
    # (self._t), which returns the unified rostam._types result objects.
    # Transport-specific extras (health, mv_*, delete_by_filter, bulk_build,
    # kv.*) are wired in later tasks (4 and 6).

    def create_collection(self, *args, **kwargs):
        """See TcpTransport.create_collection / HttpTransport.create_collection."""
        return self._t.create_collection(*args, **kwargs)

    def drop_collection(self, *args, **kwargs):
        """See HttpTransport.drop_collection. (Not yet implemented on
        TcpTransport — see Task 5 report concerns.)"""
        return self._t.drop_collection(*args, **kwargs)

    def upsert(self, *args, **kwargs):
        """See TcpTransport.upsert / HttpTransport.upsert."""
        return self._t.upsert(*args, **kwargs)

    def insert(self, *args, **kwargs):
        """See TcpTransport.insert / HttpTransport.insert."""
        return self._t.insert(*args, **kwargs)

    def upsert_batch(self, *args, **kwargs):
        """See TcpTransport.upsert_batch / HttpTransport.upsert_batch."""
        return self._t.upsert_batch(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """See TcpTransport.delete / HttpTransport.delete."""
        return self._t.delete(*args, **kwargs)

    def get(self, *args, **kwargs):
        """See TcpTransport.get / HttpTransport.get."""
        return self._t.get(*args, **kwargs)

    def get_batch(self, *args, **kwargs):
        """See TcpTransport.get_batch / HttpTransport.get_batch."""
        return self._t.get_batch(*args, **kwargs)

    def scroll(self, *args, **kwargs):
        """See TcpTransport.scroll / HttpTransport.scroll."""
        return self._t.scroll(*args, **kwargs)

    def search(self, *args, **kwargs):
        """See TcpTransport.search / HttpTransport.search."""
        return self._t.search(*args, **kwargs)

    def search_docs(self, *args, **kwargs):
        """See TcpTransport.search_docs / HttpTransport.search_docs."""
        return self._t.search_docs(*args, **kwargs)

    def search_groups(self, *args, **kwargs):
        """See TcpTransport.search_groups / HttpTransport.search_groups."""
        return self._t.search_groups(*args, **kwargs)

    def hybrid_search(self, *args, **kwargs):
        """See TcpTransport.hybrid_search / HttpTransport.hybrid_search."""
        return self._t.hybrid_search(*args, **kwargs)

    def hybrid_text(self, *args, **kwargs):
        """See TcpTransport.hybrid_text / HttpTransport.hybrid_text."""
        return self._t.hybrid_text(*args, **kwargs)

    def recommend(self, *args, **kwargs):
        """See TcpTransport.recommend / HttpTransport.recommend."""
        return self._t.recommend(*args, **kwargs)

    def query(self, *args, **kwargs):
        """See TcpTransport.query / HttpTransport.query."""
        return self._t.query(*args, **kwargs)

    def exists(self, *args, **kwargs):
        """See TcpTransport.exists / HttpTransport.exists."""
        return self._t.exists(*args, **kwargs)

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> "Rostam":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
