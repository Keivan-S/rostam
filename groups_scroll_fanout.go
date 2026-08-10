// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"sort"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// groupFanOut runs vector_group_candidates on every physical partition, unions
// the per-partition candidate documents, truncates the union to the GLOBAL
// top-fetchK by distance, then groups ONCE — reproducing single-partition group
// search EXACTLY.
//
// EXACTNESS INVARIANT: the union is truncated to fetchK BEFORE GroupDocuments.
// GroupDocuments ranks groups (and fills them) over exactly the top-fetchK
// candidate pool, best-first; feeding the full P×fetchK union would let
// candidates that single-partition search never sees enter the pool and change
// which groups appear, their order, and which hits fill them. fetchK is resolved
// with the SAME formula vector.SearchGroups uses, and both the per-partition op
// (it carries the resolved opts.FetchK in its encoded args) and this coordinator
// use that same value, so the candidate pool is identical to single-partition.
func (e *embedded) groupFanOut(collection string, P int, gen uint32, query []float32, k int, opts VectorGroupOpts) ([]VectorGroup, cluster.FanResult, error) {
	if k <= 0 {
		return nil, cluster.FanResult{}, nil
	}
	gopts := opts
	groupSize := gopts.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	// Replicate vector.SearchGroups' fetchK formula EXACTLY: if FetchK is below
	// k*groupSize, set it to 4*k*groupSize, then floor at 50.
	if want := k * groupSize; gopts.FetchK < want {
		if gopts.FetchK = 4 * want; gopts.FetchK < 50 {
			gopts.FetchK = 50
		}
	}

	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_group_candidates",
		Consistency:   cluster.Consistency(opts.ReadConsistency),
		OnUnavailable: cluster.OnUnavailable(opts.OnPartitionUnavailable),
		Encode: func(physCol string) []byte {
			// gopts carries the resolved FetchK so each partition returns exactly
			// its own top-fetchK candidates (matching single-partition's pool size).
			// rc/opa trailer carries Linearizable to the shard so the readIndex
			// barrier runs on each partition's leader. The vector_group_candidates
			// handler decodes with DecodeGroupSearchArgs, which ignores the trailer,
			// so this is wire-compatible while ops.ReadConsistencyOf(
			// "vector_group_candidates", ...) reads the byte.
			return ops.EncodeGroupSearchArgsOpts(physCol, k, query, gopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, _ int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	all, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	// EXACTNESS INVARIANT: global top-fetchK by ascending distance (ascending ID
	// tiebreak for cross-partition determinism — partition append order varies),
	// THEN group. IDs are disjoint across partitions, so the tiebreak only orders
	// genuinely equal-distance candidates, which admit any valid top-k order.
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Distance != all[j].Distance {
			return all[i].Distance < all[j].Distance
		}
		return all[i].ID < all[j].ID
	})
	if len(all) > gopts.FetchK {
		all = all[:gopts.FetchK]
	}

	return vector.GroupDocuments(all, gopts, k), fr, nil
}

// scrollFanOut scatters vector_scroll to all P physical partitions, unions the
// per-partition documents, GLOBAL-sorts by ascending id, and truncates to limit —
// the cursor-aware deep-pagination merge.
//
// Each partition receives the SAME global (filter, limit, afterID) and returns
// its first `limit` matching docs with id > afterID, ascending. IDs are disjoint
// across partitions (PartitionOf(id,P)), so the union has no duplicates; the
// global first `limit` ids > afterID are guaranteed within the union of the
// per-partition first-`limit` (each partition contributes its smallest `limit`
// ids > afterID), so the truncated global-sorted union IS the correct next page.
//
// nextCursor = the last id (id-scroll) / (value, id) position (order_by) of the
// truncated page, encoded, IFF the page is full (len==limit, so more may exist);
// else "" (exhausted). The next page sends the SAME global cursor to ALL partitions
// again — gap-free + dup-free.
//
// ORDER_BY MERGE CORRECTNESS (gap-free + dup-free re-derivation for the (value, id)
// total order over disjoint per-partition id sets):
//
//	Let < be the (value, id) total order for the request's direction (vector.OrderLess).
//	Every partition returns its OWN first `limit` admitted docs that are < -greater
//	than the SAME global cursor C (or, on page 1, at-or-after start_from), in < order,
//	with missing-field points EXCLUDED identically everywhere. Because ids are GLOBALLY
//	UNIQUE and DISJOINT per partition (PartitionOf), no two docs across partitions
//	compare equal under < (the id tiebreak is total). So the union, sorted by < and
//	truncated to `limit`, is exactly the globally-smallest `limit` docs greater than C:
//	  - dup-free: disjoint ids ⇒ no doc appears in two partitions' pages.
//	  - gap-free: the global k-th doc (in < order) greater than C lives in SOME
//	    partition p; it is among p's first `limit` results greater than C (at most
//	    `limit`-1 of p's own docs precede it, so it is within p's page); thus it is in
//	    the union and survives the global sort+truncate to its correct global rank.
//	The next page resumes strictly after the truncated page's last (value, id) — the
//	same lower-bound seek every partition applies — so consecutive pages neither skip
//	nor repeat a doc.
func (e *embedded) scrollFanOut(collection string, P int, gen uint32, filter VectorFilter, limit int, rc, opa uint8, afterID uint64, hasAfter bool, order *ops.ScrollOrder, bound uint64) ([]VectorDocument, cluster.FanResult, string, error) {
	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             limit,
		Op:            "vector_scroll",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			return ops.EncodeScrollArgsOrderBounded(physCol, filter, limit, rc, opa, afterID, hasAfter, order, bound)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, _ int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	all, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, "", err
	}
	if order != nil {
		// GLOBAL (value, id) merge: sort the union by the order's total order (NOT id),
		// then truncate to limit. Disjoint per-partition ids ⇒ no equal-comparing dups
		// (see the correctness re-derivation above). Missing-field points were already
		// EXCLUDED per partition, so the union contains only orderable docs.
		ob := scrollOrderByFromOps(order)
		all = sortDocsByOrder(all, ob)
		if limit > 0 && len(all) > limit {
			all = all[:limit]
		}
		return all, fr, scrollNextCursorOrder(all, limit, ob), nil
	}
	// GLOBAL id-ascending merge of the per-partition pages. IDs are disjoint across
	// partitions so the sort only orders the union, not equal-ID duplicates.
	sort.SliceStable(all, func(i, j int) bool { return all[i].ID < all[j].ID })
	if limit > 0 && len(all) > limit {
		all = all[:limit]
	}
	return all, fr, scrollNextCursor(all, limit), nil
}

// sortDocsByOrder sorts docs in place by the order's total order for the order_by
// field, choosing the (stringValue, id) lexicographic order (vector.OrderLessStr) for
// an OrderString order or the (value, id) numeric/datetime order (vector.OrderLess)
// otherwise. The order field's value is read from each doc's Metadata via OrderStringKey
// / OrderKey; every doc in an order_by fan-out carries it (missing-field points are
// EXCLUDED at the engine). A doc whose order field is somehow absent sorts last
// (defensive — it should never occur in a well-formed order_by union). This is the
// coordinator-side P>1 fan-out merge, so it MUST mirror the engine's per-partition sort
// for the same kind or the merged page is mis-ordered.
func sortDocsByOrder(docs []VectorDocument, ob *vector.OrderBy) []VectorDocument {
	if len(ob.Tail) > 0 {
		// MULTI-KEY: sort by the tuple-lexicographic (k1,…,kN,id) total order, mirroring
		// the engine's per-partition tuple sort EXACTLY. Each doc's per-key tuple is read
		// from its Metadata by each key's kind; every doc in a multi-key fan-out carries
		// every order field (missing-field points are EXCLUDED at the engine). A doc that
		// is somehow missing a key sorts last (defensive — never in a well-formed union).
		keys := vector.OrderKeyList(ob)
		tupleOf := func(d VectorDocument) ([]vector.OrderVal, bool) {
			vals := make([]vector.OrderVal, len(keys))
			for i := range keys {
				if keys[i].Kind == vector.OrderString {
					sk, ok := vector.OrderStringKey(d.Metadata, keys[i].Key)
					if !ok {
						return nil, false
					}
					vals[i] = vector.OrderVal{Str: sk, Kind: vector.OrderString}
					continue
				}
				k, ok := vector.OrderKey(d.Metadata, keys[i].Key, keys[i].IsDatetime)
				if !ok {
					return nil, false
				}
				vals[i] = vector.OrderVal{Num: k, Kind: keys[i].Kind}
			}
			return vals, true
		}
		sort.SliceStable(docs, func(i, j int) bool {
			ti, oki := tupleOf(docs[i])
			tj, okj := tupleOf(docs[j])
			if oki != okj {
				return oki // present (fully-keyed) sorts before absent
			}
			if !oki {
				return docs[i].ID < docs[j].ID
			}
			return vector.OrderLessTuple(
				vector.OrderedID{ID: docs[i].ID, Keys: ti},
				vector.OrderedID{ID: docs[j].ID, Keys: tj},
				keys)
		})
		return docs
	}
	if ob.Kind == vector.OrderString {
		key := func(d VectorDocument) (string, bool) {
			return vector.OrderStringKey(d.Metadata, ob.Key)
		}
		sort.SliceStable(docs, func(i, j int) bool {
			ki, oki := key(docs[i])
			kj, okj := key(docs[j])
			if oki != okj {
				return oki // present sorts before absent
			}
			if !oki {
				return docs[i].ID < docs[j].ID
			}
			return vector.OrderLessStr(ki, docs[i].ID, kj, docs[j].ID, ob.Desc)
		})
		return docs
	}
	key := func(d VectorDocument) (float64, bool) {
		return vector.OrderKey(d.Metadata, ob.Key, ob.IsDatetime)
	}
	sort.SliceStable(docs, func(i, j int) bool {
		ki, oki := key(docs[i])
		kj, okj := key(docs[j])
		if oki != okj {
			return oki // present sorts before absent
		}
		if !oki {
			return docs[i].ID < docs[j].ID
		}
		return vector.OrderLess(ki, docs[i].ID, kj, docs[j].ID, ob.Desc)
	})
	return docs
}

// buildScrollOrder reconciles an order_by request (ob) with a decoded scroll cursor
// (dec) into the (*ops.ScrollOrder, afterID, hasAfter) the args codec needs, choosing
// the v2 (numeric/datetime) or v3 (string) resume path by ob.Kind. It is the SHARED
// store-side bridge used by every directStore / embedded scroll family so the cursor
// version-validation + resume mapping stay in one place:
//
//   - OrderString: validate a v3 cursor (ValidateOrderCursorString — a v1/v2 cursor is
//     a mismatch); the resume STRING rides ScrollOrder.ResumeStr + the args afterID.
//   - numeric/datetime: validate a v2 cursor (ValidateOrderCursor — byte/behaviour-
//     identical to the pre-string path); the resume value rides ScrollOrder.ResumeKey.
//
// ob MUST be non-nil (the caller checks order_by presence). A cursor/order mismatch is
// returned loud (ops.ErrCursorOrderMismatch) so a bad combination fails at the edge.
func buildScrollOrder(ob *vector.OrderBy, dec ops.DecodedScrollCursor) (order *ops.ScrollOrder, afterID uint64, hasAfter bool, err error) {
	if len(ob.Tail) > 0 {
		// MULTI-KEY: validate a v4 cursor (a v1/v2/v3 cursor, or a wrong-arity v4, is a
		// mismatch); the resume TUPLE rides ScrollOrder.ResumeKeys + the args afterID.
		keys := vector.OrderKeyList(ob)
		keyHash := vector.OrderKeyListHash(keys)
		if verr := ops.ValidateOrderCursorTuple(dec, ob.Desc, keyHash, len(keys)); verr != nil {
			return nil, 0, false, verr
		}
		order = &ops.ScrollOrder{Key: ob.Key, Desc: ob.Desc, IsDatetime: ob.IsDatetime, Kind: ob.Kind, Tail: ops.OrderByToScrollOrderTail(ob)}
		if dec.Present {
			afterID, hasAfter = dec.LastID, true
			order.ResumeKeys = scrollOrderValsFromCursorTuple(dec.Tuple)
			order.HasResumeKeys = true
		}
		return order, afterID, hasAfter, nil
	}
	keyHash := vector.OrderKeyHash(ob.Key)
	if ob.Kind == vector.OrderString {
		if verr := ops.ValidateOrderCursorString(dec, ob.Desc, keyHash); verr != nil {
			return nil, 0, false, verr
		}
		order = &ops.ScrollOrder{Key: ob.Key, Desc: ob.Desc, Kind: vector.OrderString}
		if dec.Present {
			afterID, hasAfter = dec.LastID, true
			order.ResumeStr, order.HasResumeStr = dec.StrValue, true
		}
		return order, afterID, hasAfter, nil
	}
	if verr := ops.ValidateOrderCursor(dec, ob.Desc, keyHash); verr != nil {
		return nil, 0, false, verr
	}
	order = &ops.ScrollOrder{Key: ob.Key, Desc: ob.Desc, IsDatetime: ob.IsDatetime, Kind: ob.Kind, StartFrom: ob.StartFrom, HasStart: ob.HasStart}
	if dec.Present {
		afterID, hasAfter = dec.LastID, true
		order.ResumeKey, order.HasResume = dec.Value, true
	}
	return order, afterID, hasAfter, nil
}

// reencodeScrollCursor rebuilds the opaque resume cursor a coordinator fans out to
// every partition from a decoded order block + the resume id. It is the inverse of
// buildScrollOrder for the dispatcher's re-encode step: v2 (value, id) for a
// numeric/datetime resume, v3 (stringValue, id) for a string resume. No resume
// (page 1) ⇒ "" (no cursor). An oversized v3 string cannot occur here (it came off a
// decoded block whose strLen was already bounded), so the encoder error is ignored;
// a corrupt value yields "" (page-1 semantics) rather than a panic.
func reencodeScrollCursor(order *ops.ScrollOrder, afterID uint64) string {
	if order == nil {
		return ""
	}
	if len(order.Tail) > 0 {
		// MULTI-KEY: re-encode the v4 (k1,…,kN, id) tuple cursor. No resume tuple (page 1)
		// ⇒ "". A corrupt/oversized tuple cannot occur here (it came off a decoded block
		// whose values were already bounded), so the encoder error yields "" (page-1
		// semantics) rather than a panic.
		if !order.HasResumeKeys {
			return ""
		}
		keyHash := vector.OrderKeyListHash(vector.OrderKeyList(ops.ScrollOrderToOrderBy(order)))
		tok, err := ops.EncodeScrollCursorOrderTuple(cursorTupleFromScrollOrderVals(order.ResumeKeys), afterID, order.Desc, keyHash)
		if err != nil {
			return ""
		}
		return tok
	}
	keyHash := vector.OrderKeyHash(order.Key)
	if order.Kind == vector.OrderString {
		if order.HasResumeStr {
			tok, err := ops.EncodeScrollCursorOrderString(order.ResumeStr, afterID, order.Desc, keyHash)
			if err != nil {
				return ""
			}
			return tok
		}
		return ""
	}
	if order.HasResume {
		return ops.EncodeScrollCursorOrder(order.ResumeKey, afterID, order.Desc, keyHash)
	}
	return ""
}

// scrollOrderByFromOps maps the ops args ScrollOrder onto vector.OrderBy for the
// coordinator-side merge/cursor (resume value/id are tracked separately). It delegates to
// ops.ScrollOrderToOrderBy so the coordinator and the leaf share ONE mapping, including
// the multi-key Tail + v4 resume tuple (a single-key order maps byte-identically).
func scrollOrderByFromOps(o *ops.ScrollOrder) *vector.OrderBy {
	return ops.ScrollOrderToOrderBy(o)
}

// scrollOrderValsFromCursorTuple translates a decoded v4 cursor's resume tuple
// ([]ops.OrderKeyVal) into the ops args resume tuple ([]ops.ScrollOrderVal) so the
// multi-key resume rides the scroll args wire. Kind is the wire kind byte (== vector
// OrderKind), copied through; the live field (Num vs Str) follows the kind.
func scrollOrderValsFromCursorTuple(tuple []ops.OrderKeyVal) []ops.ScrollOrderVal {
	out := make([]ops.ScrollOrderVal, len(tuple))
	for i, kv := range tuple {
		out[i] = ops.ScrollOrderVal{Num: kv.Num, Str: kv.Str, Kind: vector.OrderKind(kv.Kind)}
	}
	return out
}

// cursorTupleFromScrollOrderVals is the inverse of scrollOrderValsFromCursorTuple: it
// translates the ops args resume tuple ([]ops.ScrollOrderVal) back into the v4 cursor
// codec's tuple ([]ops.OrderKeyVal) so the dispatcher can re-encode the global cursor it
// fans out to every partition.
func cursorTupleFromScrollOrderVals(vals []ops.ScrollOrderVal) []ops.OrderKeyVal {
	out := make([]ops.OrderKeyVal, len(vals))
	for i, rv := range vals {
		out[i] = ops.OrderKeyVal{Num: rv.Num, Str: rv.Str, Kind: byte(rv.Kind)}
	}
	return out
}
