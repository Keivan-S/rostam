"""Bounds-checking behaviour of the Phase C response decoders in _vecwire.py.

Those decoders read length prefixes (count/clen/mlen/nnz/dim/...) straight off
the wire and use them to slice the body. A truncated or hostile response must
raise a clear ValueError -- not over-read (silently returning garbage from a
short slice) and not blow up with a lower-level struct.error/IndexError that
callers can't catch uniformly alongside this module's other ValueErrors.
"""

from __future__ import annotations

import struct
import unittest

from rostam import _vecwire as w


class VecwireDecodeTruncatedTest(unittest.TestCase):
    def test_empty_body_raises_valueerror(self):
        decoders = [
            ("_decode_docs_raw", w._decode_docs_raw),
            ("decode_docs_degraded_raw", w.decode_docs_degraded_raw),
            ("decode_scroll_result_raw", w.decode_scroll_result_raw),
            ("decode_groups_degraded_raw", w.decode_groups_degraded_raw),
            ("decode_hybrid_results_degraded", w.decode_hybrid_results_degraded),
            ("decode_query_result_degraded", w.decode_query_result_degraded),
            ("decode_get_batch_result", w.decode_get_batch_result),
            ("decode_search_results_degraded", w.decode_search_results_degraded),
        ]
        for name, fn in decoders:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    fn(b"")

    def test_search_results_count_overclaims_raises(self):
        # count=2 but only one 12-byte [id:u64][distance:f32] row present.
        body = struct.pack(">I", 2) + struct.pack(">Q", 1) + struct.pack(">f", 0.5)
        with self.assertRaises(ValueError):
            w.decode_search_results_degraded(body)

    def test_docs_count_with_no_row_data_raises(self):
        # A well-formed count=1, but no bytes for the row that count promises.
        body = struct.pack(">I", 1)
        with self.assertRaises(ValueError):
            w._decode_docs_raw(body)

    def test_docs_content_length_overclaims_raises(self):
        # count=1, id/distance/score, then a content length that claims far
        # more bytes than the body actually carries. Must raise -- not
        # silently return a short (truncated) `content` string.
        body = (struct.pack(">I", 1) + struct.pack(">Q", 1)
                + struct.pack(">f", 0.0) + struct.pack(">f", 0.0)
                + struct.pack(">I", 1000))  # clen=1000, zero bytes follow
        with self.assertRaises(ValueError):
            w._decode_docs_raw(body)

    def test_get_batch_row_missing_record_body_raises(self):
        # n=1, id, found=1, then nothing: the record itself was never sent.
        body = struct.pack(">I", 1) + struct.pack(">Q", 42) + bytes([1])
        with self.assertRaises(ValueError):
            w.decode_get_batch_result(body)

    def test_get_batch_sparse_nnz_overclaims_raises(self):
        # A found row with dim=0, no ttl truncation, meta absent, sparse
        # present, and nnz claiming far more entries than follow.
        body = (struct.pack(">I", 1) + struct.pack(">Q", 1) + bytes([1])  # n, id, found
                + struct.pack(">I", 0)                                    # dim=0
                + struct.pack(">Q", 0)                                    # ttl_ms=0
                + bytes([0])                                              # meta_present=0
                + bytes([1])                                              # sparse_present=1
                + struct.pack(">I", 999))                                 # nnz=999, no entries follow
        with self.assertRaises(ValueError):
            w.decode_get_batch_result(body)

    def test_group_hits_length_overclaims_raises(self):
        key_json = b'{"kind":"int","int":1}'
        body = (struct.pack(">I", 1)                                   # groups count
                + struct.pack(">I", len(key_json)) + key_json           # key length + tagged key JSON
                + struct.pack(">I", 5000))                              # hits length=5000, no hits data follows
        with self.assertRaises(ValueError):
            w.decode_groups_degraded_raw(body)

    def test_query_result_wrong_mode_raises(self):
        with self.assertRaises(ValueError):
            w.decode_query_result_degraded(bytes([9]))  # not _QUERY_RESULT_MODE_RERANK

    def test_query_result_truncated_after_mode_raises(self):
        # RERANK-tagged, but the fused-results block behind it is missing.
        body = bytes([w._QUERY_RESULT_MODE_RERANK]) + struct.pack(">I", 1)  # count=1, no row
        with self.assertRaises(ValueError):
            w.decode_query_result_degraded(body)

    def test_hybrid_results_row_truncated_raises(self):
        body = struct.pack(">I", 1) + struct.pack(">Q", 1)  # count=1, id, missing distance/score
        with self.assertRaises(ValueError):
            w.decode_hybrid_results_degraded(body)


if __name__ == "__main__":
    unittest.main()
