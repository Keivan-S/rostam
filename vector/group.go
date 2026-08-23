// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strconv"
	"time"
)

// Group-by-document search: collapse KNN hits that share a metadata field value
// into one group, returning the top-k groups (ranked by their best member) each
// with up to GroupSize best hits. The use case is RAG retrieval over chunked
// documents — you want the k most relevant *documents*, represented by their
// best chunk(s), not k chunks that may all come from one document. It is pure
// post-processing over a candidate pool (no index or graph changes), layered on
// the same search path as SearchDocs.

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
	// defaults to max(4*k*GroupSize, 50). A larger pool finds more groups and
	// fills them more fully at the cost of more distance computations; if the
	// pool is exhausted before k groups fill, fewer groups (or smaller groups)
	// are returned rather than the search widening unboundedly.
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

// GroupCandidates returns the top-(opts.FetchK) candidate documents (ascending
// distance) that SearchGroups groups, WITHOUT grouping. Used by the cross-shard
// coordinator for exact group fan-out. The caller sets opts.FetchK to the
// resolved pool size (SearchGroups does so before calling this). If opts.FetchK
// is <= 0 on entry, a floor of 50 is applied as a safety net.
func (h *hnsw) GroupCandidates(query []float32, opts GroupOpts) ([]Document, error) {
	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	if opts.GroupBy == "" {
		return nil, ErrEmptyGroupBy
	}
	fetchK := opts.FetchK
	if fetchK <= 0 {
		fetchK = 50
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	q := query
	if h.cfg.Metric == Cosine {
		s.qbuf = append(s.qbuf[:0], query...)
		normalize(s.qbuf)
		q = s.qbuf
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	raw := h.searchDenseLocked(s, q, fetchK, pred, nil)
	docs := make([]Document, 0, len(raw))
	for _, r := range raw {
		if d, ok := h.docForLocked(r); ok {
			docs = append(docs, d)
		}
	}
	return docs, nil
}

// GroupDocuments groups candidate documents (ascending distance) by
// opts.GroupBy into the top-k groups (best member first), each with up to
// opts.GroupSize hits. cands must be the candidate pool (top-fetchK by
// distance). Exported for the cross-shard coordinator.
func GroupDocuments(cands []Document, opts GroupOpts, k int) []Group {
	groupSize := opts.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	groups := make([]Group, 0, k)
	index := make(map[string]int, k) // group key -> position in groups
	full := 0                        // groups that have reached groupSize
	for _, d := range cands {
		key, kok := d.Metadata[opts.GroupBy]
		if !kok {
			continue // hit has no value for the group field
		}
		ks, scalar := groupKeyString(key)
		if !scalar {
			continue // can't group on a list/none value
		}

		if gi, exists := index[ks]; exists {
			g := &groups[gi]
			if len(g.Hits) >= groupSize {
				continue
			}
			g.Hits = append(g.Hits, d)
			if len(g.Hits) == groupSize {
				full++
			}
			continue
		}
		// New group: only the first k distinct keys (best-first) are kept.
		if len(groups) >= k {
			continue
		}
		index[ks] = len(groups)
		groups = append(groups, Group{Key: key, Hits: []Document{d}})
		if groupSize == 1 {
			full++
		}
		if len(groups) == k && full == k {
			break // k full groups: nothing later can improve the result
		}
	}
	return groups
}

// SearchGroups runs a (optionally filtered) KNN search and collapses the results
// by the GroupOpts.GroupBy metadata field, returning up to k groups — ranked by
// each group's best (nearest) member — with up to GroupOpts.GroupSize hits each.
//
// Because the candidate pool is processed best-first, the first k distinct group
// keys encountered are exactly the k groups with the best top member; later
// candidates only fill those groups (up to GroupSize) and never displace them.
func (h *hnsw) SearchGroups(query []float32, k int, opts GroupOpts) ([]Group, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	if opts.GroupBy == "" {
		return nil, ErrEmptyGroupBy
	}
	if k <= 0 {
		return nil, nil
	}
	groupSize := opts.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	fetchK := resolveGroupFetchK(k, groupSize, opts.FetchK)

	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}

	s := getLayerScratch()
	defer layerScratchPool.Put(s)
	q := query
	if h.cfg.Metric == Cosine {
		s.qbuf = append(s.qbuf[:0], query...)
		normalize(s.qbuf)
		q = s.qbuf
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	raw := h.searchDenseLocked(s, q, fetchK, pred, nil)
	h.searchOps.Add(1)

	// Group in a single locked pass: read ONLY the group-by field per candidate
	// (no metadata clone) to bucket, and materialize the full Document (the
	// expensive metadata clone in docForLocked) ONLY for candidates accepted as
	// hits — at most k*groupSize of the fetchK pool, vs a Document per candidate in
	// the old GroupDocuments(GroupCandidates(...)) path (the SearchGroups allocation
	// hotspot). This is byte-identical to that path; the cross-shard coordinator
	// still calls GroupCandidates/GroupDocuments unchanged, and the P>1==P1 oracle
	// tests pin the equivalence.
	now := uint64(h.now())
	groups := make([]Group, 0, k)
	index := make(map[string]int, k) // group key -> position in groups
	full := 0                        // groups that have reached groupSize
	for _, r := range raw {
		slot, ok := h.arena.idMap[r.ID]
		if !ok {
			continue
		}
		// docForLocked moves contentField into Document.Content and strips it from
		// Document.Metadata, so a group-by on it never matches there — mirror that.
		if opts.GroupBy == contentField {
			continue
		}
		key, kok := h.liveMeta(slot, now)[opts.GroupBy]
		if !kok {
			continue // hit has no value for the group field
		}
		ks, scalar := groupKeyString(key)
		if !scalar {
			continue // can't group on a list/none value
		}
		if gi, exists := index[ks]; exists {
			g := &groups[gi]
			if len(g.Hits) >= groupSize {
				continue
			}
			d, dok := h.docForLocked(r)
			if !dok {
				continue
			}
			g.Hits = append(g.Hits, d)
			if len(g.Hits) == groupSize {
				full++
			}
			continue
		}
		// New group: only the first k distinct keys (best-first) are kept.
		if len(groups) >= k {
			continue
		}
		d, dok := h.docForLocked(r)
		if !dok {
			continue
		}
		index[ks] = len(groups)
		groups = append(groups, Group{Key: key, Hits: []Document{d}})
		if groupSize == 1 {
			full++
		}
		if len(groups) == k && full == k {
			break // k full groups: nothing later can improve the result
		}
	}
	return groups, nil
}

// resolveGroupFetchK resolves the candidate-pool size SearchGroups (and the
// grouped Query API path) collapses into groups: it keeps an explicit caller
// FetchK that already covers k*groupSize, else widens to 4*(k*groupSize) with a
// floor of 50. The SINGLE source of truth for the pool size so the standalone
// SearchGroups and the Query API grouped path fetch the IDENTICAL wide pool (the
// oracle equivalence depends on this). k and groupSize must already be resolved
// (groupSize defaulted to 1 when <= 0).
func resolveGroupFetchK(k, groupSize, fetchK int) int {
	if want := k * groupSize; fetchK < want {
		if fetchK = 4 * want; fetchK < 50 {
			fetchK = 50
		}
	}
	return fetchK
}

// GroupFetchK is the exported wrapper over resolveGroupFetchK so the cross-package
// coordinator (Query API grouped fan-out) widens the per-partition flat query to
// the IDENTICAL candidate pool the single-node grouped query (queryGrouped) and the
// standalone SearchGroups use — the SINGLE source of truth for the wide pool, on
// which the P>1==P1 + oracle equivalence depends. k and groupSize must already be
// resolved (groupSize defaulted to 1 when <= 0).
func GroupFetchK(k, groupSize, fetchK int) int { return resolveGroupFetchK(k, groupSize, fetchK) }

// groupKeyString renders a scalar Value as a collision-free map key (kind-tagged
// so an int and the string of the same digits never collide). Returns ok=false
// for list and none kinds, which cannot serve as a group key.
func groupKeyString(v Value) (string, bool) {
	switch v.Kind {
	case ValueString:
		return "s" + v.Str, true
	case ValueInt:
		return "i" + strconv.FormatInt(v.Int, 10), true
	case ValueFloat:
		return "f" + strconv.FormatFloat(v.Flt, 'g', -1, 64), true
	case ValueBool:
		if v.Bool {
			return "b1", true
		}
		return "b0", true
	}
	// ValueGeo (and list/none kinds) cannot serve as a group-by key: a lat/lon
	// point is a 2-D value with no meaningful single-string bucket identity, so
	// it declines here (ok=false) and the row is simply not grouped.
	return "", false
}
