// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"sort"
	"time"
)

// RAG store path: store a document (chunk text + metadata) alongside each
// vector, retrieve the content with search results, upsert by id, and delete by
// filter — the ergonomics a retrieval-augmented-generation pipeline needs.
//
// Content is stored as a reserved metadata field (contentField) rather than a
// separate column, so it inherits the metadata persistence already wired through
// snapshot, the instant-restart sidecar, and the WAL — no new persistence code.
// The reserved field is excluded from the payload (filter) index (payloadIndex.
// reindex skips it) so large text is never indexed as an equality key, and it is
// stripped from the Metadata returned to callers.

// contentField is the reserved metadata key holding a record's document content.
// The leading byte keeps it out of normal user field names.
const contentField = "$content"

// Document is a search result enriched with its stored content and (filterable)
// metadata — what a RAG caller actually wants back from a query.
type Document struct {
	ID       uint64   `json:"id"`
	Distance float32  `json:"distance"`
	Score    float32  `json:"score"` // fusion score for hybrid results; 0 for plain KNN
	Content  string   `json:"content"`
	Metadata Metadata `json:"metadata,omitempty"` // user metadata, with the reserved content field removed
}

// RawDocument is Document with its metadata left as the JSON bytes the result
// wire already carries, instead of a decoded map. It exists for ONE reason: a
// response whose only destination is JSON does not need the metadata decoded and
// re-encoded, and that round-trip is the dominant cost of a search_docs response.
//
// IT IS NOT A SECOND RESULT SHAPE. Its JSON rendering is byte-identical to
// Document's for every value the wire can carry, and that identity is what makes
// substituting it safe:
//
//   - The fields, their order, their JSON names and their omitempty flags mirror
//     Document exactly (TestRawDocumentMirrorsDocument reflects over both and
//     fails if either side gains, loses, renames or re-tags a field).
//   - Metadata's bytes were produced by json.Marshal of the very Metadata map
//     Document would hold, so emitting them verbatim emits what marshalling that
//     map would have produced. encoding/json re-escapes a json.RawMessage on the
//     way out (HTML escaping is on by default), but the bytes are already escaped
//     and escaping is idempotent, so the copy is exact. There is exactly ONE
//     escape that does not survive that argument — the one encoding/json writes
//     for invalid UTF-8 — and the decoder that fills this field normalizes it
//     before handing the bytes over (ops.checkRawMetadataJSON explains why).
//
// LIFETIME: Metadata ALIASES the buffer it was decoded from — it is a window into
// the op result bytes, not a copy. A RawDocument must not outlive that buffer.
// Decoders that hand these out (ops.DecodeVectorDocsRaw and friends) say so; a
// caller that needs to retain metadata past the response should decode the typed
// Document instead.
type RawDocument struct {
	ID       uint64          `json:"id"`
	Distance float32         `json:"distance"`
	Score    float32         `json:"score"` // fusion score for hybrid results; 0 for plain KNN
	Content  string          `json:"content"`
	Metadata json.RawMessage `json:"metadata,omitempty"` // verbatim wire bytes; aliases the source buffer
}

// RawGroup is Group with raw-metadata hits — the RawDocument counterpart of
// Group, for the same reason and with the same JSON-identity guarantee. Key is
// kept as raw JSON too: the group wire carries it as the json.Marshal of a
// vector.Value, so re-emitting those bytes is what marshalling the Value would
// produce.
type RawGroup struct {
	Key  json.RawMessage `json:"key"`
	Hits []RawDocument   `json:"hits"`
}

// WithContent returns a copy of meta carrying document content in the reserved
// content field — for callers (e.g. the networked client) that must embed
// content into metadata before encoding an upsert. The reserved field is
// excluded from filtering and stripped from SearchDocs' returned Metadata.
func WithContent(meta Metadata, content string) Metadata { return withContent(meta, content) }

// withContent returns a copy of meta with the content field set (or meta
// unchanged when content is empty). The caller's map is never mutated.
func withContent(meta Metadata, content string) Metadata {
	if content == "" {
		return meta
	}
	m := make(Metadata, len(meta)+1)
	for k, v := range meta {
		m[k] = v
	}
	m[contentField] = NewString(content)
	return m
}

// fetchDocs enriches search results with stored content + user metadata, read
// under the index's read lock. A result whose id was deleted between the search
// and this fetch is skipped (benign race).
func (h *hnsw) fetchDocs(results []Result) []Document {
	h.mu.RLock()
	defer h.mu.RUnlock()
	docs := make([]Document, 0, len(results))
	for _, r := range results {
		if d, ok := h.docForLocked(r); ok {
			docs = append(docs, d)
		}
	}
	return docs
}

// docForLocked builds the Document for result r from arena state: its stored
// content and user metadata (with the reserved content field stripped). Returns
// ok=false if the id is no longer present (deleted between search and fetch).
// The caller must hold h.mu for reading — used by both fetchDocs and the
// group-by-document path, neither of which may re-acquire the read lock.
func (h *hnsw) docForLocked(r Result) (Document, bool) {
	slot, ok := h.arena.idMap[r.ID]
	if !ok {
		return Document{}, false
	}
	d := Document{ID: r.ID, Distance: r.Distance, Score: r.Score}
	if meta := h.liveMeta(slot, uint64(h.now())); len(meta) > 0 {
		if cv, ok := meta[contentField]; ok && cv.Kind == ValueString {
			d.Content = cv.Str
		}
		out := make(Metadata, len(meta))
		for k, v := range meta {
			if k != contentField {
				out[k] = v
			}
		}
		if len(out) > 0 {
			d.Metadata = out
		}
	}
	return d, true
}

// matchingIDs returns the ids of live records whose metadata satisfies pred,
// gathered under the read lock — via the payload index when the filter is
// narrowable, else a scan of live slots. The caller deletes them through the
// WAL-aware Delete path (so delete-by-filter is logged). pred must be filter's
// compiled predicate (non-nil).
func (h *hnsw) matchingIDs(filter Filter, pred Predicate) ([]uint64, error) {
	return h.matchingIDsAt(filter, pred, uint64(h.now()))
}

// matchingIDsAt is matchingIDs judging admission (the tombstone/TTL liveness gate)
// against the caller-supplied `now` (unix millis), so a replicated delete-by-filter
// selects the SAME id set on every replica regardless of wall-clock skew (#4 vector
// TTL determinism). Must NOT hold h.mu (it takes the read lock).
func (h *hnsw) matchingIDsAt(filter Filter, pred Predicate, now uint64) ([]uint64, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	var ids []uint64
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	if cands, ok := h.payloadIdx.candidates(filter, limit); ok && len(cands) <= limit {
		for _, slot := range cands {
			// The idMap fallback below emits every admitted id verbatim, so this
			// fast path must too — dropping id 0 here made the two lanes select
			// different id sets for the same filter (and silently under-deleted).
			if h.admits(slot, pred, now) {
				ids = append(ids, h.slotID(slot))
			}
		}
		return ids, nil
	}
	for id, slot := range h.arena.idMap {
		if h.admits(slot, pred, now) {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// scrollDocs enumerates live documents whose metadata satisfies filter (a zero
// filter matches all), each enriched with content + metadata — a query-less
// listing primitive used by framework adapters (e.g. Haystack's count_documents
// / filter_documents). limit <= 0 means no cap. Iteration order is unspecified
// (it follows the id map); Distance/Score are 0 since there is no query.
func (h *hnsw) scrollDocs(filter Filter, limit int) ([]Document, error) {
	return h.scrollDocsWith(filter, limit, nil)
}

// scrollDocsWith is scrollDocs with an OPTIONAL external metadata provider (the
// named-vector hook). metaOf == nil evaluates the predicate against arena
// metadata; metaOf != nil evaluates it against the EXTERNAL per-point payload via
// metaOf(id) (the sub-arena carries no metadata).
//
// It now delegates to scrollPage with no cursor (afterID=0, hasAfter=false), so
// the no-cursor legacy scroll is DETERMINISTIC id-ASCENDING (previously it
// followed Go-map iteration order, which is randomized). This is the documented
// behavior change: a limit-capped scroll returns the smallest-id `limit` live
// documents instead of a random subset. The cursor path reuses the same
// scrollPage with an afterID lower bound.
//
// Note on the dropped payload-index fast path: the previous nil-provider fast
// path walked only the filter's candidate slots (O(candidates)) when the filter
// was narrowable below the filter-first threshold. scrollPage instead walks the
// sorted live-id snapshot applying the predicate (O(live ids scanned)). We
// REMOVE the fast path for a single correct ordering path: filtered scroll must
// be id-ascending too (the candidate set is unordered), and the cursor path
// needs the ordered walk regardless. The perf tradeoff is bounded —
// the walk stops at `limit` collected docs — and snapshot reuse keeps repeated
// pages warm.
func (h *hnsw) scrollDocsWith(filter Filter, limit int, metaOf func(id uint64) Metadata) ([]Document, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	docs, _, _ := h.scrollPage(filter, pred, metaOf, nil, 0, 0, false, limit)
	return docs, nil
}

// metaProvider supplies a point's payload by id for predicate evaluation when the
// payload lives OUTSIDE the arena (the named-vector shared-payload hook). nil ⇒
// read arena metadata. Matches admitsWith's parameter.
type metaProvider = func(id uint64) Metadata

// scrollPage walks the live id set in ASCENDING id order, applying the predicate
// (and the tombstone/TTL liveness gate via admitsWith), collecting up to `limit`
// matching documents whose id is strictly greater than afterID (when hasAfter).
// It is the single deterministic-order scroll primitive shared by the no-cursor
// legacy scroll (afterID=0, hasAfter=false) and the cursor path.
//
//   - hasAfter=false: start from the smallest live id (id 0 included — scroll/get
//     do not exclude id 0).
//   - hasAfter=true: start strictly AFTER afterID (afterID itself excluded), so
//     paging with the previous page's last id resumes without overlap.
//
// Returns the collected docs, nextAfter (the largest id collected — the next
// cursor's lower bound), and hasMore (true iff the walk stopped at `limit` AND at
// least one further id exists beyond the last one walked). When the walk reaches
// the end of the snapshot without filling `limit`, hasMore is false (exhausted).
// limit <= 0 means no cap (collect every matching doc; hasMore is always false).
//
// Locking: the warm path (snapshot fresh) walks under h.mu.RLock, preserving
// scroll read-concurrency. When the cached snapshot is version-stale it is
// rebuilt under h.mu.Lock (a double-checked rebuild closes the unlock/relock
// window), then the walk runs under that same write lock. The snapshot's ids
// slice is replaced wholesale on rebuild (never mutated in place), so a concurrent
// RLock walker always reads a stable slice consistent with its version. -race
// clean: every read/rebuild is under h.mu (R or W), exclusive against the
// write-locked mutators (Insert/Delete/Reclaim/sweep) that bump idSetVersion.
func (h *hnsw) scrollPage(filter Filter, pred Predicate, metaOf metaProvider, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	// Filter-first narrowing (id-ascending path only): when a filter is present and
	// the payload index narrows it to a selective candidate SUPERSET, walk that
	// id-sorted superset (predicate-rechecked) instead of the full snapshot. The
	// recheck makes the page byte-identical to the full walk; the gate falls back
	// otherwise. SKIPPED on the external-provider path (metaOf != nil): the sub-arena
	// payload index is empty there (mirrors the search filter-first skip).
	// Warm path: snapshot fresh ⇒ walk under the read lock. order_by keys its
	// freshness on dataVersion (payload-value sensitive); the id-scroll keys on
	// idSetVersion (id-set only).
	h.mu.RLock()
	if order != nil {
		// Filter-first order narrowing: when a filter is present and the payload index
		// narrows it to a selective candidate SUPERSET, build the value-sorted order
		// rows over THOSE candidate slots (∩ live) FRESH (never cached — the cache key
		// is filter-independent) instead of the full N-row snapshot, then
		// collectOrderedLocked seeks + predicate-rechecks + pages identically. The
		// narrowed rows are a superset of the matches in the SAME value-order, so the
		// emitted docs / nextAfter / cursor are byte-identical to the predicate-eval
		// order page; hasMore is i+1<len(narrowedRows) (the candidate superset), which
		// can skip a trailing EMPTY page the full path would emit — that page carries
		// zero docs and is invisible on the wire (the leaf discards hasMore; the
		// coordinator derives next_cursor from len(docs)==limit). SKIPPED on the
		// external-provider path (metaOf != nil): the sub-arena payload index is empty.
		if rows, ok := h.filterFirstOrderRowsLocked(filter, pred, metaOf, order); ok {
			docs, nextAfter, hasMore = h.collectOrderedLocked(rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
			h.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		if snap := h.orderSnapWarmLocked(order); snap != nil {
			docs, nextAfter, hasMore = h.collectOrderedLocked(snap.rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
			h.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
	} else if h.scrollSnap.ver == h.idSetVersion {
		// Filter-first narrowing (id-ascending path only): walk the payload-index
		// candidate SUPERSET instead of the full snapshot when the filter is
		// index-narrowable + selective. hasMore is still computed against the FULL
		// snapshot so the page (ids/order/cursor/hasMore) is byte-identical to the
		// predicate-eval walk. SKIPPED on the provider path (empty sub-arena index).
		if cands, ok := h.filterFirstScrollCandsLocked(filter, pred, metaOf); ok {
			docs, nextAfter, hasMore = h.walkScrollNarrowedLocked(cands, h.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
			h.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		docs, nextAfter, hasMore = h.walkScrollLocked(h.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
		h.mu.RUnlock()
		return docs, nextAfter, hasMore
	}
	h.mu.RUnlock()

	// Cold path: rebuild under the write lock (double-check the version after the
	// relock — another goroutine may have rebuilt it in the gap above).
	h.mu.Lock()
	defer h.mu.Unlock()
	if order != nil {
		// Filter-first order narrowing (cold path): build narrowed rows fresh; else the
		// cached full snapshot. See the warm-path comment above for the byte-identity +
		// hasMore reconciliation.
		if rows, ok := h.filterFirstOrderRowsLocked(filter, pred, metaOf, order); ok {
			return h.collectOrderedLocked(rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
		}
		snap := h.orderSnapLocked(order)
		return h.collectOrderedLocked(snap.rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
	}
	if h.scrollSnap.ver != h.idSetVersion {
		h.rebuildScrollSnapLocked()
	}
	if cands, ok := h.filterFirstScrollCandsLocked(filter, pred, metaOf); ok {
		return h.walkScrollNarrowedLocked(cands, h.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
	}
	return h.walkScrollLocked(h.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
}

// filterFirstScrollCandsLocked consults the payload index for an id-ASCENDING
// candidate superset for filter, returning (sortedIDs, true) only when filter-first
// applies: a filter is present (pred != nil), the arena holds the payload (metaOf ==
// nil), and the filter is index-narrowable AND selective (the superset is bounded by
// filterFirstThreshold). Returns (nil, false) to fall back to the full predicate-eval
// walk. The returned slice is a SUPERSET of the live matches — the narrowed walk's
// recheck drops over-cover. Must hold h.mu (read).
func (h *hnsw) filterFirstScrollCandsLocked(filter Filter, pred Predicate, metaOf metaProvider) ([]uint64, bool) {
	if pred == nil || metaOf != nil {
		return nil, false
	}
	threshold := h.effectiveFilterFirstLimit(h.arena.Size())
	slots, ok := h.payloadIdx.candidates(filter, threshold)
	if !ok || len(slots) > threshold {
		return nil, false
	}
	// Slot-keyed: map candidate slots to ids, then sort ascending (the cursor is a
	// strictly-after-id constraint, so the walk must be id-ascending). A candidate
	// slot is a currently-indexed (live) slot, so arena.ID is its current id; the
	// recheck still drops anything no longer admitted.
	ids := make([]uint64, 0, len(slots))
	for _, slot := range slots {
		ids = append(ids, h.arena.ID(slot))
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids, true
}

// walkScrollNarrowedLocked is the filter-first id-scroll walk: it emits docs from the
// id-sorted candidate SUPERSET (cands), applying the SAME admitsWith recheck as the
// full walk, but computes hasMore against the FULL snapshot (fullIDs) so the boundary
// matches walkScrollLocked EXACTLY. The full predicate-eval walk's hasMore is
// "i+1 < len(snapshot)" — i.e. a further snapshot id exists beyond the last emitted
// id (even a non-matching one); reproducing that against fullIDs makes the page
// byte-identical (a filtered page can be followed by a trailing empty page in BOTH
// paths). Must hold h.mu.
func (h *hnsw) walkScrollNarrowedLocked(cands, fullIDs []uint64, pred Predicate, metaOf metaProvider, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	now := uint64(h.now()) // one clock read for the whole scroll walk
	start := 0
	if hasAfter {
		start = sort.Search(len(cands), func(i int) bool { return cands[i] > afterID })
	}
	for i := start; i < len(cands); i++ {
		id := cands[i]
		slot, ok := h.arena.idMap[id]
		if !ok {
			continue
		}
		if !h.admitsWith(slot, pred, metaOf, now) {
			continue
		}
		d, dok := h.docForLocked(Result{ID: id})
		if !dok {
			continue
		}
		docs = append(docs, d)
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			// hasMore mirrors the full walk: a further snapshot id exists beyond
			// nextAfter (matching or not). The full walk's trailing-empty-page
			// behaviour is preserved.
			hasMore = moreBeyond(fullIDs, nextAfter)
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// moreBeyond reports whether a sorted id slice contains any id strictly greater than
// afterID (the full walk's "i+1 < len" boundary, re-expressed for the narrowed walk).
func moreBeyond(sortedIDs []uint64, afterID uint64) bool {
	j := sort.Search(len(sortedIDs), func(i int) bool { return sortedIDs[i] > afterID })
	return j < len(sortedIDs)
}

// orderSnapWarmLocked returns the cached order snapshot for this (field, direction)
// IF it exists and is fresh at the current dataVersion; nil otherwise (caller falls
// to the cold rebuild). Must hold h.mu (R or W). The returned *orderSnap's rows slice
// is immutable (a rebuild replaces the pointer), so a warm RLock reader is race-safe.
func (h *hnsw) orderSnapWarmLocked(order *OrderBy) *orderSnap {
	if snap, ok := h.orderSnaps[orderSnapCacheKey(order)]; ok && snap.ver == h.dataVersion {
		return snap
	}
	return nil
}

// orderSnapLocked returns a fresh order snapshot for (field, direction), rebuilding
// it if stale or absent. Double-checked: the warm path may have lost the version race
// and relocked, but another goroutine may have rebuilt in the gap, so re-test the
// version before rebuilding. Must hold h.mu (WRITE). The snapshot is
// FILTER-INDEPENDENT — it caches EVERY live id that HAS the order field, sorted by
// (value, id); the per-query filter + TTL gate run later in collectOrderedLocked.
func (h *hnsw) orderSnapLocked(order *OrderBy) *orderSnap {
	key := orderSnapCacheKey(order)
	if snap, ok := h.orderSnaps[key]; ok && snap.ver == h.dataVersion {
		return snap // rebuilt by another goroutine in the unlock/relock gap
	}
	rows := h.buildOrderRowsLocked(order)
	h.orderSeq++
	snap := &orderSnap{ver: h.dataVersion, seq: h.orderSeq, rows: rows}
	// Replace the entry, then evict to cap. Replacing an existing key reuses its slot
	// (no growth); a new key may push us to cap+1, so evict the oldest afterward.
	h.orderSnaps[key] = snap
	if len(h.orderSnaps) > orderCacheCap {
		evictOldestOrderSnap(h.orderSnaps)
	}
	h.orderRebuilds++
	return snap
}

// buildOrderRowsLocked collects every LIVE id that HAS the order field into a
// (value, id)-sorted slice. NOT tombstoned (tombstone bumps dataVersion, so a
// tombstoned id never belongs in a fresh snapshot). TTL-expired-but-unswept ids are
// NOT excluded (they age without a mutation/bump — same lazy treatment as the id
// scrollSnap); the walk's gate drops them. The per-query FILTER is deliberately NOT
// applied here so the snapshot is reusable across different filters. Missing /
// non-numeric order field ⇒ EXCLUDED (Qdrant default). Must hold h.mu (WRITE).
func (h *hnsw) buildOrderRowsLocked(order *OrderBy) []OrderedID {
	now := uint64(h.now())
	rows := make([]OrderedID, 0, len(h.arena.idMap))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for id, slot := range h.arena.idMap {
			if h.tombstoned[slot] {
				continue
			}
			meta := h.liveMeta(slot, now)
			vals, ok := orderTupleKeys(meta, keys)
			if !ok {
				continue // EXCLUDE: some order key absent or wrong-type
			}
			rows = append(rows, OrderedID{ID: id, Keys: vals})
		}
		SortOrderedIDsTuple(rows, keys)
		return rows
	}
	str := order.Kind == OrderString
	for id, slot := range h.arena.idMap {
		if h.tombstoned[slot] {
			continue
		}
		meta := h.liveMeta(slot, now)
		if str {
			sk, kok := OrderStringKey(meta, order.Key)
			if !kok {
				continue // EXCLUDE: order field absent or non-string
			}
			rows = append(rows, OrderedID{StrKey: sk, ID: id})
			continue
		}
		key, kok := OrderKey(meta, order.Key, order.IsDatetime)
		if !kok {
			continue // EXCLUDE: order field absent or non-numeric
		}
		rows = append(rows, OrderedID{Key: key, ID: id})
	}
	if str {
		SortOrderedIDsStr(rows, order.Desc)
	} else {
		SortOrderedIDs(rows, order.Desc)
	}
	return rows
}

// filterFirstOrderRowsLocked builds the value-sorted order rows ONLY over the payload-
// index candidate SUPERSET (∩ live) for filter, or (nil, false) when filter-first order
// narrowing does not apply (no filter (pred==nil), the external-provider path
// (metaOf!=nil, empty sub-arena index), a non-accelerable filter, or a non-selective one
// (> filterFirstThreshold)). The candidate slots are a superset of the matching slots;
// the per-row field-presence EXCLUDE here + the per-row predicate recheck in
// collectOrderedLocked make the narrowed rows EXACTLY the field-present matches in the
// SAME value-order as the full snapshot, so the page (docs / nextAfter / cursor) is
// byte-identical to the predicate-eval order page. The rows are NOT cached: the orderSnaps
// cache key is filter-independent (orderSnapCacheKey), so a narrowed snapshot must never
// be stored there or a different-filter / no-filter scroll would read a stale, partial
// snapshot. Must hold h.mu (R or W). The order analogue of filterFirstScrollCandsLocked.
func (h *hnsw) filterFirstOrderRowsLocked(filter Filter, pred Predicate, metaOf metaProvider, order *OrderBy) ([]OrderedID, bool) {
	if pred == nil || metaOf != nil {
		return nil, false
	}
	threshold := h.effectiveFilterFirstLimit(h.arena.Size())
	slots, ok := h.payloadIdx.candidates(filter, threshold)
	if !ok || len(slots) > threshold {
		return nil, false
	}
	now := uint64(h.now())
	rows := make([]OrderedID, 0, len(slots))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for _, slot := range slots {
			if h.tombstoned[slot] {
				continue
			}
			meta := h.liveMeta(slot, now)
			vals, vok := orderTupleKeys(meta, keys)
			if !vok {
				continue // EXCLUDE: some order key absent or wrong-type
			}
			rows = append(rows, OrderedID{ID: h.slotID(slot), Keys: vals})
		}
		SortOrderedIDsTuple(rows, keys)
		return rows, true
	}
	str := order.Kind == OrderString
	for _, slot := range slots {
		if h.tombstoned[slot] {
			continue
		}
		meta := h.liveMeta(slot, now)
		if str {
			sk, kok := OrderStringKey(meta, order.Key)
			if !kok {
				continue // EXCLUDE: order field absent or non-string
			}
			rows = append(rows, OrderedID{StrKey: sk, ID: h.slotID(slot)})
			continue
		}
		key, kok := OrderKey(meta, order.Key, order.IsDatetime)
		if !kok {
			continue // EXCLUDE: order field absent or non-numeric
		}
		rows = append(rows, OrderedID{Key: key, ID: h.slotID(slot)})
	}
	if str {
		SortOrderedIDsStr(rows, order.Desc)
	} else {
		SortOrderedIDs(rows, order.Desc)
	}
	return rows, true
}

// collectOrderedLocked seeks the cached (value, id) sorted rows past the cursor /
// start_from, then walks forward applying the TTL/tombstone gate + the per-query
// FILTER (the snapshot is filter-independent and TTL-lazy, so the gate runs HERE),
// materializing up to `limit` Documents. The returned docs carry the order field in
// Metadata so the coordinator can read the last doc's order value for the v2
// next-cursor. Must hold h.mu (R or W). rows is immutable (never mutated in place).
func (h *hnsw) collectOrderedLocked(rows []OrderedID, pred Predicate, metaOf metaProvider, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	now := uint64(h.now())
	start := orderSeekStart(rows, order, afterID, afterKey, hasAfter)
	for i := start; i < len(rows); i++ {
		id := rows[i].ID
		slot, ok := h.arena.idMap[id]
		if !ok {
			continue // reclaimed since the snapshot was built (benign)
		}
		// TTL/tombstone gate + per-query filter over the SAME live-meta view, so the
		// predicate sees one consistent payload (a snapshot id may have TTL-expired
		// lazily, or be filtered out by this query's pred).
		if h.tombstoned[slot] || h.isExpired(slot) {
			continue
		}
		if pred != nil {
			var meta Metadata
			if metaOf != nil {
				meta = metaOf(h.arena.ID(slot))
			} else {
				meta = h.liveMeta(slot, now)
			}
			if !pred(meta) {
				h.filterRejects.Add(1)
				continue
			}
		}
		d, dok := h.docForLocked(Result{ID: id})
		if !dok {
			continue
		}
		docs = append(docs, d)
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			hasMore = i+1 < len(rows)
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// bumpData advances dataVersion, invalidating every cached order snapshot (a later
// scroll re-tests snap.ver == dataVersion and misses). Called under h.mu (WRITE) at
// every id-set mutation (alongside idSetVersion++) AND every payload mutation. Kept
// separate from idSetVersion so a payload write does NOT touch the id scrollSnap.
func (h *hnsw) bumpData() { h.dataVersion++ }

// rebuildScrollSnapLocked recollects the LIVE id set (idMap keys minus tombstoned
// slots) into a freshly-sorted ascending slice and caches it at the current
// idSetVersion. TTL-expired-but-not-yet-swept ids are NOT excluded here (they age
// without a mutation, so no version bump would have fired) — the forward walk's
// admits gate filters them via isExpired. Must hold h.mu (write). O(live · log).
func (h *hnsw) rebuildScrollSnapLocked() {
	ids := make([]uint64, 0, len(h.arena.idMap))
	for id, slot := range h.arena.idMap {
		if h.tombstoned[slot] {
			continue // excluded from the live set, mirroring admits' tombstone gate
		}
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	h.scrollSnap.ids = ids
	h.scrollSnap.ver = h.idSetVersion
	h.scrollRebuilds++
}

// walkScrollLocked walks the sorted-ascending `ids` snapshot, seeking past
// afterID (binary search) when hasAfter, then applying admitsWith/docForLocked to
// collect up to `limit` matching docs. Must hold h.mu (R or W). See scrollPage for
// the contract; this is the lock-agnostic body shared by both lock paths.
func (h *hnsw) walkScrollLocked(ids []uint64, pred Predicate, metaOf metaProvider, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	now := uint64(h.now()) // one clock read for the whole scroll walk
	start := 0
	if hasAfter {
		// First index whose id is strictly greater than afterID.
		start = sort.Search(len(ids), func(i int) bool { return ids[i] > afterID })
	}
	for i := start; i < len(ids); i++ {
		id := ids[i]
		slot, ok := h.arena.idMap[id]
		if !ok {
			continue // id reclaimed since the snapshot was built (benign; skip)
		}
		if !h.admitsWith(slot, pred, metaOf, now) {
			continue // tombstoned / TTL-expired / filtered out
		}
		d, dok := h.docForLocked(Result{ID: id})
		if !dok {
			continue
		}
		docs = append(docs, d)
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			// Filled the page. hasMore iff at least one further id exists beyond
			// this one in the snapshot (a later id we have not yet returned).
			hasMore = i+1 < len(ids)
			return docs, nextAfter, hasMore
		}
	}
	// Reached the end without filling limit ⇒ exhausted.
	return docs, nextAfter, false
}

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
	// Version is the point's per-point CAS version (>= 1 for a live point). It is
	// carried through the scan codec and re-applied VERBATIM by the reshard backfill
	// (a version-preserving reinsert) so resharded points keep their version rather
	// than resetting to 1.
	Version uint64
	// KeyExpires is the point's per-key payload TTL map (payload key -> ABSOLUTE
	// unix-millis deadline), an OWNED clone of arena.KeyExpires. nil/empty when the
	// point has no per-key TTL (the common case). It is carried through the scan
	// codec and re-applied VERBATIM by the reshard backfill (NOT recomputed now+ttl)
	// so resharded points keep their original absolute deadlines time-stable.
	KeyExpires map[string]uint64
}

// scanVectors enumerates every LIVE record in the arena as a self-contained
// ScanRecord. Liveness mirrors scrollDocs/admits exactly: a slot in idMap is
// skipped when tombstoned or TTL-expired (a nil predicate matches all metadata,
// so admits reduces to the tombstone+expiry gate). Each record is deep-copied
// off arena storage — the vector slice is a VIEW (append-copied), the metadata
// map is rebuilt, and the sparse vector (an aliased pointer) is cloned — so the
// result is safe to retain and mutate without corrupting the index. The TTL is
// reconstructed from the absolute unix-millis deadline as the REMAINING
// duration (deadline - now); expired records never reach here.
func (h *hnsw) scanVectors() []ScanRecord {
	h.mu.RLock()
	defer h.mu.RUnlock()
	now := uint64(h.now())
	recs := make([]ScanRecord, 0, len(h.arena.idMap))
	for id, slot := range h.arena.idMap {
		if !h.admits(slot, nil, now) { // tombstoned or expired → skip (same gate as scrollDocs)
			continue
		}
		rec := ScanRecord{
			ID:      id,
			Vec:     append([]float32(nil), h.vecFor(slot)...), // COPY: vecFor aliases arena (exact) or reconstructs from the code when PQDropVecs dropped the floats
			Version: h.arena.Version(slot),
		}
		if exp := h.arena.ExpiresAt(slot); exp > now { // exp==0 (no expiry) or already-past (filtered by admits)
			rec.TTL = time.Duration(exp-now) * time.Millisecond
		}
		if meta := h.liveMeta(slot, now); len(meta) > 0 {
			out := make(Metadata, len(meta))
			for k, v := range meta {
				out[k] = v
			}
			rec.Metadata = out
		}
		if sv := h.arena.Sparse(slot); sv != nil {
			rec.Sparse = sv.Clone() // clone: arena owns the pointer
		}
		if ke := h.arena.KeyExpires(slot); len(ke) > 0 {
			// CLONE: KeyExpires aliases arena storage (the set path clones-on-write,
			// so the live map must not be retained). Carries ABSOLUTE unix-ms deadlines
			// verbatim so the reshard reinsert restores them time-stable.
			out := make(map[string]uint64, len(ke))
			for k, v := range ke {
				out[k] = v
			}
			rec.KeyExpires = out
		}
		recs = append(recs, rec)
	}
	return recs
}
