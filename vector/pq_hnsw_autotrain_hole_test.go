// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
	"time"
)

// freshCodeForSlot encodes slot's CURRENTLY-RESIDENT float vector under the
// index's trained codec, independent of whatever code is stored in the arena. It
// is the oracle for "this slot holds ITS OWN vector's code": the stored code at
// slot must byte-equal this fresh re-encode. Requires the floats to be resident
// (PQDropVecs=false) and the codec trained. Must hold no lock in the test (single
// goroutine).
func freshCodeForSlot(t *testing.T, h *hnsw, slot uint32) []byte {
	t.Helper()
	buf := make([]byte, h.quant.CodeLen())
	h.quant.Encode(buf, h.arena.Vec(slot))
	return buf
}

// assertEverySlotOwnsItsCode is THE direct bug-catcher: for EVERY live arena slot
// it asserts the stored PQ code byte-equals a fresh Encode(Vec(slot)). With the
// pre-fix index-keyed encode, a hole below a live slot shifts sample[i] off slot
// i, so the stored code is some OTHER vector's code → this fails. With the
// slot-keyed encode every slot holds its own code → this passes.
func assertEverySlotOwnsItsCode(t *testing.T, h *hnsw) (checked int) {
	t.Helper()
	capacity := h.arena.Capacity()
	for s := 0; s < capacity; s++ {
		slot := uint32(s)
		if !h.liveSlotLocked(slot) {
			continue
		}
		want := freshCodeForSlot(t, h, slot)
		got := h.arena.Code(slot)
		if !bytes.Equal(got, want) {
			t.Fatalf("slot %d holds the WRONG vector's PQ code: stored %x != fresh re-encode %x "+
				"(pre-train hole shifted sample index off the live slot)", slot, got, want)
		}
		checked++
	}
	return checked
}

// TestPQHNSWAutoTrainHoleDenseSlotCorrect is the reviewer's exact repro for the
// DENSE PQ-HNSW family: insert ids 0..N, Delete a MID-stream id BEFORE crossing
// the auto-train threshold, then insert more to cross IVFTrainThreshold so the
// incremental auto-train fires WITH a hole below live slots. It asserts every live
// slot owns its own code (the slot-keyed encode), and that ADC search after the
// hole-crossing returns correct results vs a brute-force oracle over the LIVE set
// (the deleted id absent, the live ids ranked correctly).
//
// PRE-FIX: the soft Delete leaves a tombstoned hole at a low slot; trainAndEncodePQ
// encoded sample[i] (compacted) into slot i, so every slot above the hole stored
// the NEXT vector's code → assertEverySlotOwnsItsCode fails (~all shifted slots).
// POST-FIX: encodeLiveSlotsLockedPQ keys by real slot → every slot owns its code.
func TestPQHNSWAutoTrainHoleDenseSlotCorrect(t *testing.T) {
	const (
		dim       = 64
		threshold = 600
		nClusters = 30
		m         = 16
		k         = 10
	)
	// Clustered data so ADC has real structure (ADC recall on isotropic noise is
	// meaningless). pqDropClusteredCorpus returns 0-based vecs; ids are i+1.
	const before = 400 // inserted, then one deleted, leaving 399 live
	const after = 250  // 399 + 250 = 649 live > threshold ⇒ auto-train trips
	const n = before + after
	vecs, queries := pqDropClusteredCorpus(n, dim, nClusters, 60, 60)

	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false)) // floats resident
	if err != nil {
		t.Fatal(err)
	}

	// Insert the first batch (ids 1..before). Still below threshold ⇒ untrained.
	for i := 0; i < before; i++ {
		if _, _, err := h.Insert(uint64(i+1), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	if !h.pqUntrained() {
		t.Fatalf("at %d live the index must still be untrained (threshold %d)", before, threshold)
	}

	// Delete a MID-stream id BEFORE the auto-train threshold is crossed. This soft
	// delete tombstones a LOW arena slot but keeps it in idMap → liveCount still
	// counts it for a moment, and crucially the slot becomes a HOLE below later
	// live slots, which is exactly what shifts the index-keyed encode.
	const deletedID = uint64(200) // a mid-stream id (arena slot ~199)
	removed, err := h.Delete(deletedID, CASCond{})
	if err != nil {
		t.Fatalf("delete %d: %v", deletedID, err)
	}
	if !removed {
		t.Fatalf("delete %d reported not-removed", deletedID)
	}

	// Insert the rest, crossing the threshold so the auto-train fires WITH the hole
	// already present below the new live slots.
	for i := before; i < n; i++ {
		if _, _, err := h.Insert(uint64(i+1), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	if h.pqUntrained() {
		t.Fatal("index must be trained after crossing the threshold with a pre-train hole")
	}

	// THE direct bug-catcher: every live slot holds ITS OWN vector's code.
	checked := assertEverySlotOwnsItsCode(t, h)
	t.Logf("dense: verified %d live slots each hold their own code (1 hole below)", checked)

	// ADC search correctness after the hole-crossing: build a brute-force oracle
	// over the LIVE set (deleted id excluded) and confirm recall is sane and the
	// deleted id never appears.
	live := make(map[uint64][]float32, n)
	for i := 0; i < n; i++ {
		id := uint64(i + 1)
		if id == deletedID {
			continue
		}
		live[id] = vecs[i]
	}
	var matches, total int
	for _, q := range queries {
		truth := bruteTopKLive(live, q, k)
		res, err := h.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if r.ID == deletedID {
				t.Fatalf("deleted id %d returned by ADC search after the hole-crossing", deletedID)
			}
			if truth[r.ID] {
				matches++
			}
			total++
		}
	}
	rec := float64(matches) / float64(len(queries)*k)
	t.Logf("dense post-hole ADC recall@%d=%.3f (over %d results)", k, rec, total)
	if rec < 0.30 {
		t.Fatalf("dense post-hole ADC recall@%d=%.3f below floor 0.30 — slot-shift corrupted codes", k, rec)
	}
}

// TestPQHNSWAutoTrainHoleTTLSlotCorrect mirrors the dense repro but creates the
// pre-train hole via TTL EXPIRY (the other liveSlotLocked exclusion) rather than
// Delete: an early id is inserted with a short TTL, expires before the threshold
// crossing, leaving a hole below the live slots. Asserts every live slot owns its
// code after auto-train.
func TestPQHNSWAutoTrainHoleTTLSlotCorrect(t *testing.T) {
	const (
		dim       = 64
		threshold = 600
		nClusters = 30
		m         = 16
	)
	const before = 400
	const after = 250
	const n = before + after
	vecs, _ := pqDropClusteredCorpus(n, dim, nClusters, 5, 61)

	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false))
	if err != nil {
		t.Fatal(err)
	}

	// Insert an early id with a short TTL so it expires (becomes a hole) before the
	// threshold crossing.
	const ttlID = uint64(1)
	if _, _, err := h.Insert(ttlID, vecs[0], 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert ttl id: %v", err)
	}
	// Insert the rest of the first batch (ids 2..before).
	for i := 1; i < before; i++ {
		if _, _, err := h.Insert(uint64(i+1), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	// Let the TTL id expire → slot 0 becomes a live-excluded hole below all others.
	time.Sleep(120 * time.Millisecond)
	if h.liveSlotLocked(0) {
		t.Fatal("ttl id should have expired (slot 0 must be a hole) before the crossing")
	}

	// Cross the threshold with the hole present.
	for i := before; i < n; i++ {
		if _, _, err := h.Insert(uint64(i+1), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	if h.pqUntrained() {
		t.Fatal("index must be trained after crossing the threshold with a TTL hole")
	}

	checked := assertEverySlotOwnsItsCode(t, h)
	t.Logf("ttl: verified %d live slots each hold their own code (1 expired hole)", checked)
}

// TestPQHNSWAutoTrainHoleNamedSlotCorrect proves the NAMED family inherits the
// slot-correct incremental encode: a named QuantPQ dense space, populated via the
// incremental named Insert path, with a mid-stream named Delete before the
// threshold crossing. After auto-train every live inner-arena slot owns its own
// code.
func TestPQHNSWAutoTrainHoleNamedSlotCorrect(t *testing.T) {
	const (
		dim       = 64
		threshold = 600
		nClusters = 30
	)
	const before = 400
	const after = 250
	const n = before + after
	vecs, queries := pqDropClusteredCorpus(n, dim, nClusters, 40, 62)

	cfg := map[string]NamedVectorParams{
		"title": {
			Dim: dim, Metric: L2, Quant: QuantPQ, RescoreFactor: 3,
			IVFTrainThreshold: threshold, // PQDropVecs=false: keep floats for the oracle
		},
	}
	nc, err := NewNamedCollection("default/named-hole", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	for i := 0; i < before; i++ {
		if err := nc.Insert(uint64(i+1), map[string][]float32{"title": vecs[i]}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	h := nc.indexes["title"].(*hnsw)
	if !h.pqUntrained() {
		t.Fatal("named space must be untrained before the crossing")
	}

	// Mid-stream named Delete BEFORE the crossing → pre-train hole in the inner arena.
	const deletedID = uint64(200)
	removed, err := nc.Delete(deletedID)
	if err != nil {
		t.Fatalf("named delete %d: %v", deletedID, err)
	}
	if !removed {
		t.Fatalf("named delete %d reported not-removed", deletedID)
	}

	for i := before; i < n; i++ {
		if err := nc.Insert(uint64(i+1), map[string][]float32{"title": vecs[i]}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	if h.pqUntrained() {
		t.Fatal("named space must auto-train after crossing the threshold with a hole")
	}

	checked := assertEverySlotOwnsItsCode(t, h)
	t.Logf("named: verified %d live inner slots each hold their own code", checked)

	// Search through the named path returns the live ids, never the deleted one.
	const k = 10
	for _, q := range queries[:20] {
		res, err := nc.SearchNamed("title", q, k, Filter{})
		if err != nil {
			t.Fatalf("named search: %v", err)
		}
		for _, r := range res {
			if r.ID == deletedID {
				t.Fatalf("named ADC search returned the deleted id %d after the hole-crossing", deletedID)
			}
		}
	}
}

// TestPQHNSWAutoTrainHoleMVSlotCorrect proves the MV family inherits the
// slot-correct incremental encode: an MV QuantPQ inner token index, populated via
// the incremental MV Add path, with a mid-stream MV Delete before the inner token
// count crosses the threshold. After auto-train every live inner-arena (token)
// slot owns its own code.
func TestPQHNSWAutoTrainHoleMVSlotCorrect(t *testing.T) {
	const (
		dim          = 64
		threshold    = 600
		tokensPerDoc = 8
		nClusters    = 30
	)
	// docsBefore*tokensPerDoc must stay < threshold; after the delete + more docs
	// the live token count crosses it.
	const docsBefore = 60 // 60*8 = 480 tokens (< 600)
	const docsAfter = 40  // +40 docs ⇒ +320 tokens; minus 8 deleted ⇒ ~792 live tokens
	const docs = docsBefore + docsAfter
	pool, queries := pqDropClusteredCorpus(docs*tokensPerDoc, dim, nClusters, 5, 63)

	cfg := MultiVectorConfig{
		Dim: dim, Seed: 9, Quant: QuantPQ, RescoreFactor: 3,
		IVFTrainThreshold: threshold, // PQDropVecs=false: floats resident for the oracle
	}
	mv, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer mv.Close()

	docToks := func(id uint64) [][]float32 {
		toks := make([][]float32, tokensPerDoc)
		for j := 0; j < tokensPerDoc; j++ {
			toks[j] = pool[(int(id)-1)*tokensPerDoc+j]
		}
		return toks
	}

	for id := uint64(1); id <= docsBefore; id++ {
		if err := mv.Add(id, docToks(id), Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	inner := mv.idx.(*hnsw)
	if !inner.pqUntrained() {
		t.Fatal("MV inner must be untrained before the crossing")
	}

	// Mid-stream MV Delete BEFORE the crossing → its tokens become pre-train holes
	// in the inner token arena.
	const deletedDoc = uint64(30)
	if !mv.Delete(deletedDoc) {
		t.Fatalf("MV delete %d reported not-removed", deletedDoc)
	}

	for id := uint64(docsBefore + 1); id <= docs; id++ {
		if err := mv.Add(id, docToks(id), Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	if inner.pqUntrained() {
		t.Fatal("MV inner must auto-train after crossing the token threshold with holes")
	}

	checked := assertEverySlotOwnsItsCode(t, inner)
	t.Logf("mv: verified %d live inner token slots each hold their own code", checked)

	// MaxSim search still returns results after the hole-crossing.
	got, err := mv.Search(queries[:5], 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatalf("MV search: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("MV MaxSim search returned no results after the hole-crossing")
	}
	for _, r := range got {
		if r.ID == deletedDoc {
			t.Fatalf("MV search returned the deleted doc %d after the hole-crossing", deletedDoc)
		}
	}
}

// bruteTopKLive is bruteTopK over an EXPLICIT live id→vec map (not the dense
// 1-based vecs slice), so the deleted id is excluded from the oracle. L2-squared
// ranking, matching the test configs' Metric: L2.
func bruteTopKLive(live map[uint64][]float32, q []float32, k int) map[uint64]bool {
	type cd struct {
		id uint64
		d  float32
	}
	cs := make([]cd, 0, len(live))
	for id, v := range live {
		cs = append(cs, cd{id, l2SquaredScalar(v, q)})
	}
	// Partial selection of the k smallest (simple full sort is fine for test sizes).
	for i := 0; i < len(cs); i++ {
		for j := i + 1; j < len(cs); j++ {
			if cs[j].d < cs[i].d {
				cs[i], cs[j] = cs[j], cs[i]
			}
		}
	}
	out := make(map[uint64]bool, k)
	for i := 0; i < k && i < len(cs); i++ {
		out[cs[i].id] = true
	}
	return out
}
