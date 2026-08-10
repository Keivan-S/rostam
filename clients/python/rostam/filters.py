"""Metadata filter builders.

These produce the JSON predicate tree Rostam expects, wrapping scalar values in
the tagged form automatically::

    from rostam import filters as f

    f.eq("doc_id", 7)
    f.and_(f.gte("price", 10.0), f.eq("in_stock", True))
    f.in_("tag", ["a", "b"])

Pass the result as the ``filter`` argument to the client's search/delete methods.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._values import encode_value

Filter = Dict[str, Any]


def _cmp(op: str, field: str, value: Any) -> Filter:
    return {"op": op, "field": field, "value": encode_value(value)}


def eq(field: str, value: Any) -> Filter:
    """field == value"""
    return _cmp("eq", field, value)


def ne(field: str, value: Any) -> Filter:
    """field != value"""
    return _cmp("ne", field, value)


def gt(field: str, value: Any) -> Filter:
    """field > value"""
    return _cmp("gt", field, value)


def gte(field: str, value: Any) -> Filter:
    """field >= value"""
    return _cmp("gte", field, value)


def lt(field: str, value: Any) -> Filter:
    """field < value"""
    return _cmp("lt", field, value)


def lte(field: str, value: Any) -> Filter:
    """field <= value"""
    return _cmp("lte", field, value)


def in_(field: str, values: List[Any]) -> Filter:
    """field's scalar is a member of values"""
    return {"op": "in", "field": field, "value": encode_value(list(values))}


def contains(field: str, value: Any) -> Filter:
    """field's array contains the scalar value"""
    return _cmp("contains", field, value)


def and_(*subs: Filter) -> Filter:
    """All subfilters match."""
    return {"op": "and", "and": list(subs)}


def or_(*subs: Filter) -> Filter:
    """Any subfilter matches."""
    return {"op": "or", "or": list(subs)}


def not_(sub: Filter) -> Filter:
    """The subfilter does not match."""
    return {"op": "not", "not": sub}
