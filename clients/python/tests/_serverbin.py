"""Locate the rostam-server binary that the cross-stack tests drive.

Those tests exist to prove the Python client and the Go server agree on a
byte-level wire contract; a mock would only prove the client agrees with itself.
So they need a REAL server, and there are two distinct reasons they might not
get one:

  - **No binary at all.** Nothing to say: skip quietly.
  - **A binary older than the Go sources.** This one used to be silent and cost
    real debugging time. A stale server does not refuse the request — it answers
    a binary body it predates with `invalid json body: invalid character 'r'`,
    which reads exactly like a client-side framing bug. Skip, and say how to fix
    it.

An explicit ROSTAM_SERVER_BIN is trusted as-is: CI builds the binary
immediately before running, and a caller naming a path has already decided. If
that path does not exist it raises rather than skipping — someone asked for a
specific server and did not get it, which is a broken setup, not an absent one.
CI relies on this: it sets the variable so a failed build surfaces as a failure
instead of a lane full of green skips.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# clients/python/tests -> repo root is three levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_CANDIDATE_NAMES = ("rostam-server-test", "rostam-server")

# Directories with no Go sources worth stat-ing. Dot-directories are skipped
# too, which covers .git and any tooling scratch space alongside it.
_SKIP_DIRS = {"node_modules", "site", "_site", "testdata", "clients", "docs"}


def _newest_go_mtime(root: str) -> float:
    """Modification time of the most recently touched .go file under root."""
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".go"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, fn)))
                except OSError:
                    pass  # raced with a build; one file's mtime is not worth failing over
    return newest


def find_server_bin() -> Tuple[Optional[str], str]:
    """Return (path, "") when a usable server binary exists, else (None, reason).

    The reason is written to be shown as a pytest skip message, so it names the
    command that fixes the situation rather than merely reporting it.
    """
    env = os.environ.get("ROSTAM_SERVER_BIN")
    if env:
        if not os.path.exists(env):
            raise RuntimeError(
                f"ROSTAM_SERVER_BIN={env} does not exist. Someone asked for that server "
                f"explicitly, so this is a broken setup rather than a missing one -- "
                f"failing instead of skipping, which would hide it."
            )
        return env, ""

    for name in _CANDIDATE_NAMES:
        cand = os.path.join(_REPO_ROOT, name)
        if not (os.path.exists(cand) and os.access(cand, os.X_OK)):
            continue
        newest_src = _newest_go_mtime(_REPO_ROOT)
        if newest_src and os.path.getmtime(cand) < newest_src:
            return None, (
                f"{name} is older than the Go sources, so it would fail in ways that look "
                f"like client bugs. Rebuild: go build -o {name} ./cmd/rostam-server"
            )
        return cand, ""

    return None, "rostam-server binary not found (set ROSTAM_SERVER_BIN or build rostam-server)"
