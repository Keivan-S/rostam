// SPDX-License-Identifier: Apache-2.0

package vtypes

import "time"

// ScanRecord is a complete, live record exported by scanVectors: everything an
// offline resplit needs to re-insert it into a re-hashed generation. The fields
// mirror Collection.Insert's parameters (TTL as a remaining duration, metadata
// as the stored map including the reserved content field, sparse as an owned
// copy) so resplit can round-trip a record straight back through vector_insert.
type ScanRecord struct {
	ID       uint64
	Vec      []float32
	TTL      time.Duration // remaining time-to-live (0 = no expiry)
	Metadata Metadata      // user metadata incl. the reserved content field; nil if none
	Sparse   *SparseVector // owned copy; nil if none
	// Version is the point's per-point CAS version (>= 1 for a live point). Carried
	// through the scan codec and re-applied VERBATIM by the reshard backfill.
	Version uint64
	// KeyExpires is the point's per-key payload TTL map (payload key -> ABSOLUTE
	// unix-millis deadline), an OWNED clone. nil/empty when the point has no per-key
	// TTL.
	KeyExpires map[string]uint64
}

// BM25GlobalStats carries the cross-partition BM25 corpus statistics fed into the
// two-phase (dfs_query_then_fetch) text scorer: N is the summed live document
// count across all partitions, Avgdl is the global average document length, and
// DF is the summed per-query-term document frequency. Built by the ops layer from
// each shard's CorpusStats; consumed by the *Global scoring entry points so every
// shard scores with the SAME IDF.
type BM25GlobalStats struct {
	N     int
	Avgdl float32
	DF    map[uint32]int
}
