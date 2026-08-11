"""Read a request body in either encoding, for the fake servers.

The client sends searches in the binary query framing ("RVQ1") when it can, so a
fake that only calls ``json.loads`` crashes on byte one and the test surfaces as
"Remote end closed connection without response" — a transport error for what is
really an encoding the fake does not speak.

This decodes the framing **from the server's specification**, deliberately
without importing the client's encoder. A decoder built by calling the encoder
in reverse agrees with the encoder by construction and would pass just as
happily if both were wrong; this one fails when they disagree, which is the only
thing it is here to detect. The layout below is copied from httpapi/binary_search.go.
"""

from __future__ import annotations

import json
import struct
import sys
from array import array
from typing import Any, Dict, Optional

RVQ1_MAGIC = b"RVQ1"
RVQ1_HEADER = 28
RVQ1_FLAG_FILTER = 1 << 0


def decode_rvq1(raw: bytes) -> Dict[str, Any]:
    """Decode an RVQ1 body into the dict its JSON twin would have produced."""
    if len(raw) < RVQ1_HEADER:
        raise ValueError(f"RVQ1 body too short: {len(raw)} bytes")
    magic, flags, k, dim, rc, opa, pad, staleness = struct.unpack(">4sIIIBBHQ", raw[:RVQ1_HEADER])
    if magic != RVQ1_MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if pad != 0:
        raise ValueError("reserved bytes are not zero")

    end = RVQ1_HEADER + dim * 4
    if len(raw) < end:
        raise ValueError(f"RVQ1 body truncated: want {end} bytes, have {len(raw)}")
    vec = array("f")
    vec.frombytes(raw[RVQ1_HEADER:end])
    if sys.byteorder == "little":  # the wire is big-endian
        vec.byteswap()

    body: Dict[str, Any] = {"query": list(vec), "k": k}
    if rc:
        body["read_consistency"] = rc
    if opa:
        body["on_partition_unavailable"] = opa
    if staleness:
        body["max_staleness"] = staleness

    if flags & RVQ1_FLAG_FILTER:
        (blen,) = struct.unpack(">I", raw[end:end + 4])
        body["filter"] = json.loads(raw[end + 4:end + 4 + blen])
        end += 4 + blen
    if flags & ~RVQ1_FLAG_FILTER:
        raise ValueError(f"unknown flags {flags:#x}")
    if len(raw) != end:
        raise ValueError(f"trailing bytes after RVQ1 frame: {len(raw) - end}")
    return body


def read_body(headers, rfile) -> Optional[Dict[str, Any]]:
    """Read one request body, JSON or RVQ1, and return it as a dict."""
    n = int(headers.get("Content-Length", 0))
    if not n:
        return None
    raw = rfile.read(n)
    ctype = (headers.get("Content-Type") or "").split(";")[0].strip()
    if ctype == "application/octet-stream" and raw[:4] == RVQ1_MAGIC:
        return decode_rvq1(raw)
    return json.loads(raw)
