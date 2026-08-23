// SPDX-License-Identifier: Apache-2.0

package vtypes

// GroupOpts configures a group-by-document search.
type GroupOpts struct {
	// GroupBy is the metadata field to group on (e.g. "doc_id"). Required; a
	// hit lacking the field, or whose value is a list/none kind (not a scalar),
	// is skipped. An empty GroupBy returns ErrEmptyGroupBy.
	GroupBy string
	// GroupSize is the maximum number of hits returned per group, best-first.
	// Values <= 0 default to 1 (one representative chunk per document).
	GroupSize int
	// Filter is an optional metadata predicate applied during candidate search.
	Filter Filter
	// FetchK is the candidate pool collapsed into groups. 0 (or too small)
	// defaults to max(4*k*GroupSize, 50).
	FetchK int

	// ReadConsistency and OnPartitionUnavailable are cross-shard routing knobs
	// consumed by the clustered fan-out coordinator; the single-node engine
	// ignores them. 0 = AnyReplica / Partial (defaults); 1 = LeaderOnly / Fail.
	ReadConsistency        uint8
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's committed
	// frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64
}

// Group is one group of search hits sharing a GroupBy field value. Hits are
// ordered best-first (ascending distance); Key is the shared field value.
type Group struct {
	Key  Value      `json:"key"`
	Hits []Document `json:"hits"`
}
