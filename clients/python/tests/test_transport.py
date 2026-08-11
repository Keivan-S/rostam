"""Transport-level behaviour: connection reuse, and the binary query framing.

These assert on the MECHANISM, not on results. Every test here would pass on the
old connection-per-request JSON client if it only checked that a search returned
the right hits — which is exactly how a silently-disabled optimization survives a
green suite.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rostam import RostamClient, RostamError
from _wire import read_body


class RecordingServer(BaseHTTPRequestHandler):
    """Records how each request arrived. Set class attrs to steer responses."""

    # Without this, BaseHTTPRequestHandler answers HTTP/1.0 and closes after
    # every response — so a keep-alive test would measure the fake, not the
    # client. ClosingServer below keeps that behaviour on purpose.
    protocol_version = "HTTP/1.1"

    content_types = []          # Content-Type of every POSTed search
    connections = 0             # distinct TCP connections accepted
    search_bodies = []
    fail_binary_as_old_server = False   # answer octet-stream like a pre-RVQ1 server
    fail_next_with = None       # (status, error) applied to the next search

    def setup(self):
        super().setup()
        type(self).connections += 1

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        cls = type(self)
        cls.content_types.append(ctype)

        if cls.fail_binary_as_old_server and ctype == "application/octet-stream":
            # Exactly what a server without RVQ1 support does: hands the bytes to
            # its JSON decoder, which fails on the first one.
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            return self._send(400, {"error": "invalid JSON body: invalid character 'R' looking for beginning of value"})

        if cls.fail_next_with is not None:
            status, err = cls.fail_next_with
            cls.fail_next_with = None
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            return self._send(status, {"error": err})

        body = read_body(self.headers, self.rfile)
        cls.search_bodies.append(body)
        k = (body or {}).get("k", 1)
        return self._send(200, {"results": [{"id": i + 1, "distance": 0.5} for i in range(k)],
                                "degraded": False, "missing": []})


class TransportTest(unittest.TestCase):
    def setUp(self):
        RecordingServer.content_types = []
        RecordingServer.search_bodies = []
        RecordingServer.connections = 0
        RecordingServer.fail_binary_as_old_server = False
        RecordingServer.fail_next_with = None
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), RecordingServer)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_search_is_sent_in_the_binary_framing(self):
        c = RostamClient(self.url)
        hits = c.search("docs", [0.25, -0.5, 1.0], k=2)
        c.close()
        self.assertEqual(["application/octet-stream"], RecordingServer.content_types)
        self.assertEqual(2, len(hits))
        # The server decoded the frame independently; the vector must survive it.
        self.assertEqual([0.25, -0.5, 1.0], RecordingServer.search_bodies[0]["query"])
        self.assertEqual(2, RecordingServer.search_bodies[0]["k"])

    def test_filter_survives_the_binary_framing(self):
        c = RostamClient(self.url)
        filt = {"op": "eq", "field": "lang", "value": {"kind": "string", "str": "en"}}
        c.search("docs", [1.0, 2.0], k=1, filter=filt)
        c.close()
        self.assertEqual(filt, RecordingServer.search_bodies[0]["filter"])

    def test_binary_can_be_turned_off(self):
        c = RostamClient(self.url, binary_search=False)
        c.search("docs", [1.0], k=1)
        c.close()
        self.assertEqual(["application/json"], RecordingServer.content_types)

    def test_falls_back_to_json_against_a_server_without_rvq1(self):
        RecordingServer.fail_binary_as_old_server = True
        c = RostamClient(self.url)
        hits = c.search("docs", [1.0, 2.0], k=3)   # binary rejected, retried as JSON
        self.assertEqual(3, len(hits))
        self.assertEqual(["application/octet-stream", "application/json"],
                         RecordingServer.content_types)

        # ...and it must not keep paying for the discovery on every later search.
        c.search("docs", [1.0, 2.0], k=1)
        c.close()
        self.assertEqual(["application/octet-stream", "application/json", "application/json"],
                         RecordingServer.content_types)

    def test_a_real_error_is_not_mistaken_for_a_missing_feature(self):
        """A 400 about the request itself must surface, not trigger a JSON retry."""
        RecordingServer.fail_next_with = (400, "k must be between 1 and 65536")
        c = RostamClient(self.url)
        with self.assertRaises(RostamError) as caught:
            c.search("docs", [1.0], k=0)
        c.close()
        self.assertIn("k must be between", str(caught.exception))
        self.assertEqual(["application/octet-stream"], RecordingServer.content_types)

    def test_connections_are_reused(self):
        c = RostamClient(self.url)
        for _ in range(12):
            c.search("docs", [1.0, 2.0], k=1)
        c.close()
        self.assertEqual(12, len(RecordingServer.content_types))
        self.assertEqual(1, RecordingServer.connections,
                         "12 searches should share one connection, not open one each")

    def test_close_then_reuse_reconnects(self):
        c = RostamClient(self.url)
        c.search("docs", [1.0], k=1)
        c.close()
        c.search("docs", [1.0], k=1)   # the client stays usable after close()
        c.close()
        self.assertEqual(2, RecordingServer.connections)

    def test_concurrent_searches_do_not_share_a_connection(self):
        """The reason there is a pool and not one shared socket.

        Two threads writing into one kept-alive connection interleave and
        desynchronize the response stream; with a pool each thread holds a
        connection for the length of its request.
        """
        c = RostamClient(self.url, pool_maxsize=4)
        errors = []
        counts = []

        def worker():
            try:
                for _ in range(10):
                    counts.append(len(c.search("docs", [1.0, 2.0, 3.0], k=2)))
            except Exception as e:  # pragma: no cover - only on a real failure
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        c.close()

        self.assertEqual([], errors)
        self.assertEqual(80, len(counts))
        self.assertTrue(all(n == 2 for n in counts), "a desynchronized response would differ")


class ClosingServer(RecordingServer):
    """A server that closes the connection after every response (HTTP/1.0)."""

    protocol_version = "HTTP/1.0"


class ClosingServerTest(unittest.TestCase):
    """Pooling must not become a penalty against a server that will not pool.

    The first version of this client put every used connection back regardless,
    so against a closing server each request found a dead socket, failed a write,
    and retried — paying two round trips where the old client paid one.
    """

    def setUp(self):
        ClosingServer.content_types = []
        ClosingServer.search_bodies = []
        ClosingServer.connections = 0
        ClosingServer.fail_binary_as_old_server = False
        ClosingServer.fail_next_with = None
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), ClosingServer)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_works_and_opens_exactly_one_connection_per_request(self):
        c = RostamClient(self.url)
        for _ in range(6):
            self.assertEqual(1, len(c.search("docs", [1.0, 2.0], k=1)))
        c.close()
        # One per request: the client must notice the close and not retry into it.
        self.assertEqual(6, ClosingServer.connections)


if __name__ == "__main__":
    unittest.main()
