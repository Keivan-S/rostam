"""Translation between native Python values and Rostam's tagged metadata wire
form.

Rostam encodes each metadata attribute as a tagged union, e.g. an integer is
``{"kind": "int", "int": 7}`` and a string is ``{"kind": "string", "str": "a"}``.
The SDK hides this: callers pass and receive plain Python values, and these
helpers convert at the boundary. (The Go side omits zero scalars from the JSON,
so decoding defaults a missing payload to the zero value for its kind.)
"""

from __future__ import annotations

from typing import Any, Dict


def encode_value(v: Any) -> Dict[str, Any]:
    """Encode a native Python value as a tagged Rostam Value."""
    # bool must precede int: bool is a subclass of int in Python.
    if isinstance(v, bool):
        return {"kind": "bool", "bool": v}
    if isinstance(v, int):
        return {"kind": "int", "int": v}
    if isinstance(v, float):
        return {"kind": "float", "flt": v}
    if isinstance(v, str):
        return {"kind": "string", "str": v}
    if isinstance(v, (list, tuple)):
        items = list(v)
        if not items:
            raise TypeError("cannot encode an empty list as a metadata value")
        if all(isinstance(x, bool) for x in items):
            raise TypeError("Rostam has no boolean-list metadata kind")
        if all(isinstance(x, int) and not isinstance(x, bool) for x in items):
            return {"kind": "ints", "ints": items}
        if all(isinstance(x, float) for x in items):
            return {"kind": "floats", "flts": items}
        if all(isinstance(x, str) for x in items):
            return {"kind": "strings", "strs": items}
        raise TypeError(f"mixed or unsupported list element types: {items!r}")
    raise TypeError(f"unsupported metadata value type: {type(v).__name__}")


def decode_value(tv: Dict[str, Any]) -> Any:
    """Decode a tagged Rostam Value into a native Python value."""
    kind = tv.get("kind")
    if kind == "string":
        return tv.get("str", "")
    if kind == "int":
        return tv.get("int", 0)
    if kind == "float":
        return tv.get("flt", 0.0)
    if kind == "bool":
        return tv.get("bool", False)
    if kind == "strings":
        return tv.get("strs", [])
    if kind == "ints":
        return tv.get("ints", [])
    if kind == "floats":
        return tv.get("flts", [])
    if kind == "none" or kind is None:
        return None
    raise ValueError(f"unknown Rostam value kind: {kind!r}")


def encode_metadata(meta: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Encode a ``{field: native}`` map into Rostam's tagged metadata map."""
    if not meta:
        return None
    return {k: encode_value(v) for k, v in meta.items()}


def decode_metadata(meta: Dict[str, Any] | None) -> Dict[str, Any]:
    """Decode a tagged metadata map back into ``{field: native}``."""
    if not meta:
        return {}
    return {k: decode_value(v) for k, v in meta.items()}
