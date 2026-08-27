"""A stateful stdlib HTTP fake of Rostam's REST surface, for adapter tests.

It stores upserted points (vector + content + tagged metadata), ranks
search_docs by L2 distance, and evaluates Rostam metadata filters for
scroll/delete_by_filter — enough to exercise the LlamaIndex/Haystack adapters
faithfully without the Go binary.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from _wire import read_body


def _dec(tv):
    k = tv.get("kind")
    return {
        "string": tv.get("str", ""), "int": tv.get("int", 0), "float": tv.get("flt", 0.0),
        "bool": tv.get("bool", False), "strings": tv.get("strs", []),
        "ints": tv.get("ints", []), "floats": tv.get("flts", []),
    }.get(k)


def _match(flt, meta):
    if not flt:
        return True
    op = flt["op"]
    if op == "and":
        return all(_match(s, meta) for s in flt.get("and", []))
    if op == "or":
        return any(_match(s, meta) for s in flt.get("or", []))
    if op == "not":
        return not _match(flt["not"], meta)
    field = flt["field"]
    if field not in meta:
        return False
    got, want = _dec(meta[field]), _dec(flt["value"])
    # Evaluate lazily: a single dict literal would compute every comparison
    # eagerly, and an "in"/"contains" op (list-valued want) then raises
    # TypeError on the unrelated gt/lt branches (str vs list).
    if op == "eq":
        return got == want
    if op == "ne":
        return got != want
    if op == "gt":
        return got > want
    if op == "gte":
        return got >= want
    if op == "lt":
        return got < want
    if op == "lte":
        return got <= want
    if op == "in":
        return got in (want or [])
    if op == "contains":
        return want in (got or [])
    return False


def _l2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


class FakeRostam:
    """Starts a loopback HTTP server emulating Rostam; use .url and .close()."""

    def __init__(self):
        self.docs = {}  # collection -> {id: {"vec","content","metadata"}}
        store = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _body(self):
                return read_body(self.headers, self.rfile)

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _coll(self):
                # /v1/collections/{c}/points/... -> c
                parts = self.path.split("/")
                return parts[3]

            def do_POST(self):
                body = self._body()
                p = self.path
                if p == "/v1/collections":
                    store.docs.setdefault(body["name"], {})
                    return self._send(201, {"name": body["name"]})
                c = self._coll()
                col = store.docs.setdefault(c, {})
                if p.endswith("/points"):
                    col[body["id"]] = {"vec": body["vector"], "content": body.get("content", ""), "metadata": body.get("metadata") or {}}
                    return self._send(200, {"id": body["id"]})
                if p.endswith("/search/docs"):
                    q, k, flt = body["query"], body["k"], body.get("filter")
                    items = [(i, d) for i, d in col.items() if _match(flt, d["metadata"])]
                    items.sort(key=lambda kv: _l2(q, kv[1]["vec"]))
                    docs = [{"id": i, "distance": _l2(q, d["vec"]), "content": d["content"], "metadata": d["metadata"]}
                            for i, d in items[:k]]
                    return self._send(200, {"documents": docs})
                if p.endswith("/points/scroll"):
                    flt, limit = body.get("filter"), body.get("limit", 0)
                    # Deterministic id-ascending order + resume-after-id cursor,
                    # mirroring the server's scroll contract. The fake cursor is
                    # just the last returned id as a string.
                    cursor = body.get("cursor", "")
                    after = int(cursor) if cursor else -1
                    matched = sorted(i for i, d in col.items()
                                     if _match(flt, d["metadata"]) and i > after)
                    page_ids = matched[:limit] if limit else matched
                    docs = [{"id": i, "distance": 0.0, "content": col[i]["content"], "metadata": col[i]["metadata"]}
                            for i in page_ids]
                    next_cursor = str(page_ids[-1]) if (limit and len(matched) > limit) else ""
                    return self._send(200, {"documents": docs, "next_cursor": next_cursor})
                if p.endswith("/points/delete"):
                    flt = body.get("filter")
                    gone = [i for i, d in col.items() if _match(flt, d["metadata"])]
                    for i in gone:
                        del col[i]
                    return self._send(200, {"deleted": len(gone)})
                if p.endswith("/points/batch-get"):
                    ids = body.get("ids", [])
                    with_vec = body.get("with_vector", True)
                    with_pay = body.get("with_payload", True)
                    points, missing = [], []
                    for i in ids:
                        d = col.get(i)
                        if d is None:
                            missing.append(i)
                            continue
                        payload = dict(d["metadata"]) if with_pay else {}
                        if with_pay and d.get("content"):
                            payload["$content"] = {"kind": "string", "str": d["content"]}
                        points.append({
                            "id": i,
                            "vector": list(d["vec"]) if with_vec else [],
                            "payload": payload,
                            "ttl_ms": 0,
                        })
                    return self._send(200, {"points": points, "missing": missing})
                if p.endswith("/points/search/hybrid-text"):
                    q, text, k = body["vector"], body.get("text", ""), body["k"]
                    flt = body.get("filter")
                    terms = set(text.lower().split())
                    scored = []
                    for i, d in col.items():
                        if not _match(flt, d["metadata"]):
                            continue
                        dense = 1.0 / (1.0 + _l2(q, d["vec"]))
                        words = set(d.get("content", "").lower().split())
                        bm25 = len(terms & words)
                        scored.append((i, dense + bm25))   # simple additive fusion
                    scored.sort(key=lambda kv: kv[1], reverse=True)
                    results = [{"id": i, "distance": 0.0, "score": s} for i, s in scored[:k]]
                    return self._send(200, {"results": results})
                if p.endswith("/points/search/hybrid"):
                    q, k = body["dense"], body["k"]
                    flt = body.get("filter")
                    items = [(i, d) for i, d in col.items() if _match(flt, d["metadata"])]
                    items.sort(key=lambda kv: _l2(q, kv[1]["vec"]))
                    results = [{"id": i, "distance": _l2(q, d["vec"]), "score": 1.0 / (1.0 + _l2(q, d["vec"]))}
                               for i, d in items[:k]]
                    return self._send(200, {"results": results})
                self._send(404, {"error": "not found"})

            def do_DELETE(self):
                self._body()
                c = self.path.split("/")[3]
                col = store.docs.setdefault(c, {})
                pid = int(self.path.rsplit("/", 1)[1])
                return self._send(200, {"deleted": col.pop(pid, None) is not None})

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        host, port = self._srv.server_address
        self.url = f"http://{host}:{port}"

    def close(self):
        self._srv.shutdown()
