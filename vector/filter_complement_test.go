// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// The complement gate's tests. The gate itself is documented in
// filter_bitset.go; what this file exists to pin is the ONE property whose
// failure is silent — that a slot the predicate rejects can never leave the gate
// with its bit still set.

// TestComplementGateEquivalence is the differential run for the rejection-side
// gate: for every filter shape, forcing the complement gate on must change
// nothing observable, exactly as TestAdmitGateEquivalence requires of the
// positive one.
//
// gateForceComplement (rather than gateForce) is what makes this test mean
// anything. Every filter the complement can handle, the positive gate can handle
// too, and under gateForce the positive one arms first — so a gateForce run
// would compare the positive gate against gate-off and report a green about a
// code path it never entered.
func TestComplementGateEquivalence(t *testing.T) {
	h := gateSharedCorpus(t)
	queries := gateQueries(gateQueryN(120), gateSharedDim)

	armed := 0
	for _, tc := range gateFilters() {
		t.Run(tc.name, func(t *testing.T) {
			off := runGateArm(t, h, queries, tc.f, gateOff, false)
			comp := runGateArm(t, h, queries, tc.f, gateForceComplement, false)
			if nDiff, rejectsDiffer := diffGateArms(off, comp); nDiff != 0 || rejectsDiffer {
				t.Fatalf("%s: complement gate changed results (%d queries differ) or rejects (%d vs %d)",
					tc.name, nDiff, off.rejects, comp.rejects)
			}
			if comp.gates > 0 {
				armed++
			}
		})
	}
	// A suite where the complement never armed would pass forever. The ordering
	// shapes in gateFilters (the seq sweep, the nanscore shapes, the size/score
	// ranges) are all over TOTAL fields, so a healthy run arms on many of them.
	if armed < 8 {
		t.Fatalf("only %d filter shapes armed a complement gate — the equivalence assertions are near-vacuous", armed)
	}
}

// TestComplementGateBitsetMechanics covers buildComplement away from the search
// path: all bits start SET, each rejection posting clears one, and the
// combination across sets is a UNION (fail any conjunct, fail the And) rather
// than the intersection the positive build uses.
func TestComplementGateBitsetMechanics(t *testing.T) {
	var g admitGate
	const capacity = 200
	a := map[uint32]struct{}{3: {}, 5: {}, 199: {}}
	b := map[uint32]struct{}{5: {}, 64: {}}
	g.buildComplement([]map[uint32]struct{}{a, b}, capacity)

	if !g.active() || !g.exact {
		t.Fatalf("complement gate: active=%v exact=%v, want both true", g.active(), g.exact)
	}
	rejected := map[uint32]bool{3: true, 5: true, 64: true, 199: true}
	for slot := uint32(0); slot < capacity; slot++ {
		if got, want := g.test(slot), !rejected[slot]; got != want {
			t.Errorf("slot %d: admitted=%v, want %v", slot, got, want)
		}
	}
	// Reuse must not leak the previous query's marks: the all-ones fill covers
	// the whole word range, so a slot cleared last time is set this time.
	g.disable()
	g.buildComplement([]map[uint32]struct{}{{7: {}}}, capacity)
	for _, slot := range []uint32{3, 5, 64, 199} {
		if !g.test(slot) {
			t.Errorf("slot %d still marked after rebuild — buildComplement did not refill", slot)
		}
	}
	if g.test(7) {
		t.Error("slot 7 admitted, want rejected")
	}
}

// optionalFieldCorpus builds a corpus where `opt` is present on only PART of the
// points — the shape the totality precondition exists for. A range over `opt`
// rejects every point that lacks it, and those points appear in NO posting set
// of either sign, so a complement gate built without checking totality would
// admit them.
func optionalFieldCorpus(t testing.TB, n, dim int, presentEvery int) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{
		Dim: dim, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		meta := Metadata{"always": NewInt(int64(i % 100))} //nolint:gosec // bounded
		if i%presentEvery != 0 {
			meta["opt"] = NewInt(int64(i % 100)) //nolint:gosec // bounded
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	return h
}

// TestComplementGateRequiresTotalField pins the precondition itself: a range
// over a field only some points carry must arm NO complement gate, and the same
// range over a field every point carries must arm one.
func TestComplementGateRequiresTotalField(t *testing.T) {
	h := optionalFieldCorpus(t, 2_000, 8, 5)
	h.mu.RLock()
	defer h.mu.RUnlock()
	n := h.arena.Size()

	if !h.payloadIdx.fieldTotalNumeric("always", n) {
		t.Fatal("precondition: `always` is on every point and must read as total")
	}
	if h.payloadIdx.fieldTotalNumeric("opt", n) {
		t.Fatal("`opt` is on 80% of the points but reads as total — the posting counter is wrong")
	}

	prev := admitGateMode
	admitGateMode = gateForceComplement
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.buildComplementGate(s, Filter{Op: FilterGte, Field: "opt", Value: NewInt(1)}, 10)
	if s.gate.active() {
		t.Error("a range over a PARTIAL field armed a complement gate — every point lacking the field would be admitted")
	}
	s.gate.disable()

	h.buildComplementGate(s, Filter{Op: FilterGte, Field: "always", Value: NewInt(1)}, 10)
	if !s.gate.active() {
		t.Error("a range over a TOTAL field armed no complement gate")
	}
	s.gate.disable()
}

// TestComplementGateTotalityMutantIsCaught is the mutation test the safety
// argument demands. Flipping the grading — treating a partial field as though
// its rejection postings covered every rejected slot — must produce OBSERVABLY
// WRONG results, or the totality check is decoration and the differential suite
// proves nothing about it.
//
// This is the complement's analogue of TestAdmitGateExactMutantsAreCaught, and
// it mutates the same kind of thing: the CONCLUSION (this field is total), not
// one of the premises, because the premises are over-determined and a mutated
// premise gets vetoed by the others.
func TestComplementGateTotalityMutantIsCaught(t *testing.T) {
	h := optionalFieldCorpus(t, 2_000, 8, 5)
	queries := gateQueries(gateQueryN(60), 8)
	// A range that a point WITHOUT the field must fail, and that most points
	// carrying the field pass — so the mutant's wrongly-admitted points are the
	// visible difference.
	f := Filter{Op: FilterGte, Field: "opt", Value: NewInt(1)}

	off := runGateArm(t, h, queries, f, gateOff, false)
	if honest := runGateArm(t, h, queries, f, gateForceComplement, false); honest.gates != 0 {
		t.Fatalf("the honest run armed %d complement gates over a partial field", honest.gates)
	}

	prevSkip := admitGateSkipTotality
	admitGateSkipTotality = true
	mutant := runGateArm(t, h, queries, f, gateForceComplement, false)
	admitGateSkipTotality = prevSkip

	if mutant.gates == 0 {
		t.Fatal("the mutant armed no gate — the seam did not take effect, so nothing was tested")
	}
	nDiff, rejectsDiffer := diffGateArms(off, mutant)
	if nDiff == 0 && !rejectsDiffer {
		t.Fatal("dropping the totality precondition changed NOTHING observable — " +
			"the complement gate's safety rests on a check no test can see fail")
	}
}

// TestComplementGateArmsWhenNothingIsRejected pins the empty-rejection branch,
// which is the one place the complement gate draws a conclusion from an ABSENCE
// of postings — normally a mistake, and sound here only because totality was
// checked first.
//
// A range that every live point satisfies produces no in-range key on the
// negated side. Over a TOTAL field that proves the filter matches everything, so
// the gate arms with every bit set and the traversal skips the predicate
// entirely. Over a PARTIAL field the identical emptiness would mean the
// opposite, which is why the totality check runs before any key is looked at.
func TestComplementGateArmsWhenNothingIsRejected(t *testing.T) {
	h := optionalFieldCorpus(t, 1_000, 8, 5) // `always` = i%100, on every point
	queries := gateQueries(gateQueryN(40), 8)
	f := Filter{Op: FilterGte, Field: "always", Value: NewInt(0)} // matches every point

	off := runGateArm(t, h, queries, f, gateOff, false)
	on := runGateArm(t, h, queries, f, gateForceComplement, false)
	if on.gates == 0 {
		t.Fatal("a range nothing rejects armed no gate — the empty-rejection proof was skipped")
	}
	if nDiff, rejectsDiffer := diffGateArms(off, on); nDiff != 0 || rejectsDiffer {
		t.Fatalf("admit-all complement gate changed results (%d differ) or rejects (%d vs %d)",
			nDiff, off.rejects, on.rejects)
	}
	if off.rejects != 0 {
		t.Fatalf("precondition: the filter should reject nothing, but the gate-off arm tallied %d", off.rejects)
	}
}

// TestComplementGateSurvivesNaNValues is the interaction between the two halves
// of m5. A NaN-valued field carries no index key, so it is in neither the accept
// nor the reject postings — which means a field with NaN values is NOT total,
// and the complement gate must decline rather than admit those points.
func TestComplementGateSurvivesNaNValues(t *testing.T) {
	h, err := newHNSW(Config{Dim: 8, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(5))
	const n = 1_000
	for i := 0; i < n; i++ {
		v := make([]float32, 8)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		score := float64(i % 50)
		if i%37 == 0 {
			score = math.NaN()
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, Metadata{"score": NewFloat(score)}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.payloadIdx.fieldTotalNumeric("score", h.arena.Size()) {
		t.Fatal("a field with NaN values reads as total — NaN slots carry no key, so they cannot be covered")
	}
	prev := admitGateMode
	admitGateMode = gateForceComplement
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)
	h.buildComplementGate(s, Filter{Op: FilterGte, Field: "score", Value: NewFloat(10)}, 10)
	if s.gate.active() {
		t.Error("a range over a field containing NaN armed a complement gate — the NaN points would be admitted")
	}
	s.gate.disable()
}

// TestComplementGateCostModelDeclinesUniqueValuedRanges is the m5 cost-model
// assertion, and it is about the term the m4 model did not have. Walking one
// DISTINCT KEY of a range costs ~7.6 predicate evaluations (gateRangeKeyDNS vs
// gateVisitSaveDNS), so a 99%-pass range over a UNIQUE-valued field — the
// VectorDBBench shape — must be declined under gateAuto even though its
// rejection MASS looks affordable. A mass-only model arms it and loses.
func TestComplementGateCostModelDeclinesUniqueValuedRanges(t *testing.T) {
	h, err := newHNSW(Config{Dim: 8, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(11))
	const n = 20_000
	for i := 0; i < n; i++ {
		v := make([]float32, 8)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		// id: unique per point (the VDBBench payload). enum: 4 distinct values,
		// the low-cardinality shape the complement gate is FOR.
		meta := Metadata{
			"id":   NewInt(int64(i)),      //nolint:gosec // bounded
			"enum": NewInt(int64(i % 40)), //nolint:gosec // bounded
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	prev := admitGateMode
	admitGateMode = gateAuto
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	// 99% pass over a unique-valued field: the rejection side is 1% of the corpus
	// in slots but ALSO 1% of the corpus in distinct keys, and the key budget is
	// what says no.
	h.buildComplementGate(s, Filter{Op: FilterGte, Field: "id", Value: NewInt(n / 100)}, 10)
	if s.gate.active() {
		t.Error("a 99 percent-pass range over a unique-valued field armed a gate — the per-key cost term is missing or too small")
	}
	s.gate.disable()

	// The same pass rate over a field with 40 distinct values: the rejection side
	// is one key, so the gate is cheap and must arm.
	h.buildComplementGate(s, Filter{Op: FilterGte, Field: "enum", Value: NewInt(1)}, 10)
	if !s.gate.active() {
		t.Error("a 97.5 percent-pass range over a 40-valued field did not arm — the complement gate never fires in production")
	}
	s.gate.disable()
}

// TestComplementGateArmsForMultiConjunctRanges pins the budget arithmetic, and
// specifically that the MASS budget is charged once rather than twice.
//
// massLimit is enforced against a running total threaded across every leaf
// (appendComplementSets), so pre-dividing it by the leaf count charges the same
// division a second time and caps an And at a `leaves`-fold tighter budget than
// the cost model priced. Declining is always safe, which is exactly why the bug
// was invisible: the gate simply never armed for the multi-conjunct ranges it
// exists to serve. keyLimit is the opposite — checked per leaf, so it IS divided
// — and this test pins the asymmetry by choosing a filter that fits the honest
// mass budget and would not fit a halved one.
func TestComplementGateArmsForMultiConjunctRanges(t *testing.T) {
	h, err := newHNSW(Config{Dim: 8, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(4242))
	const n = 20_000
	for i := 0; i < n; i++ {
		v := make([]float32, 8)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		// Two low-cardinality fields: each rejects 1/40th of the corpus under
		// `>= 1`, so each conjunct contributes one key and n/40 postings.
		meta := Metadata{
			"e1": NewInt(int64(i % 40)),        //nolint:gosec // bounded
			"e2": NewInt(int64((i / 40) % 40)), //nolint:gosec // bounded
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	prev := admitGateMode
	admitGateMode = gateAuto
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	single := Filter{Op: FilterGte, Field: "e1", Value: NewInt(1)}
	h.buildComplementGate(s, single, 10)
	if !s.gate.active() {
		t.Fatal("precondition: the single-conjunct form must arm, or this test cannot isolate the And")
	}
	s.gate.disable()

	both := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterGte, Field: "e1", Value: NewInt(1)},
		{Op: FilterGte, Field: "e2", Value: NewInt(1)},
	}}
	h.buildComplementGate(s, both, 10)
	if !s.gate.active() {
		t.Error("an And of two affordable ranges armed no gate — the mass budget is being divided by the " +
			"leaf count on top of the running total that already accounts for every leaf")
	}
	s.gate.disable()

	// The equivalence of the two-conjunct gate is what makes arming it worth
	// anything: it must reject exactly what the predicate rejects.
	pred, err := both.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	h.buildComplementGate(s, both, 10)
	if !s.gate.active() {
		t.Fatal("the And gate stopped arming on the second attempt")
	}
	defer s.gate.disable()
	now := uint64(h.now())
	accepted := 0
	for slot := 0; slot < h.arena.Capacity(); slot++ {
		u := uint32(slot) //nolint:gosec // bounded by capacity
		if h.tombstoned[u] || h.isExpiredAt(u, now) {
			continue
		}
		want := pred(h.liveMeta(u, now))
		if want {
			accepted++
		}
		if got := s.gate.test(u); got != want {
			t.Fatalf("slot %d: gate=%v predicate=%v", slot, got, want)
		}
	}
	if accepted == 0 {
		t.Fatal("nothing matched the And — the equivalence assertion proved nothing")
	}
}

// TestComplementGateSurvivesRebuildWithTombstones pins a perf cliff that used to
// survive restarts and had no counter to notice it by.
//
// fieldTotalNumeric compares a field's posting count against arena.Size(), which
// counts idMap and therefore INCLUDES tombstoned slots. Delete tombstones
// without touching the payload index, so incrementally the two agree — but
// rebuild (Restore, Reclaim) used to skip tombstoned slots, after which the
// count could never reach Size() and the complement gate silently declined for
// EVERY field, permanently, on any collection that was snapshotted with a single
// deleted point.
//
// The assertion is deliberately about behaviour before and after, not about the
// counter: totality is a property of the index, and it must not depend on which
// route the index took to its current contents.
func TestComplementGateSurvivesRebuildWithTombstones(t *testing.T) {
	h := optionalFieldCorpus(t, 800, 8, 5) // `always` is on every point
	if _, err := h.Delete(3, CASCond{}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := h.Delete(77, CASCond{}); err != nil {
		t.Fatalf("delete: %v", err)
	}

	h.mu.RLock()
	n := h.arena.Size()
	totalBefore := h.payloadIdx.fieldTotalNumeric("always", n)
	h.mu.RUnlock()
	if !totalBefore {
		t.Fatal("precondition: `always` is on every point, so it must read as total before the rebuild")
	}

	// Take the index through the same rebuild a Restore does.
	h.mu.Lock()
	h.payloadIdx.rebuild(h.arena)
	h.mu.Unlock()

	h.mu.RLock()
	defer h.mu.RUnlock()
	if !h.payloadIdx.fieldTotalNumeric("always", h.arena.Size()) {
		t.Fatal("`always` stopped reading as total after a rebuild with tombstones present — " +
			"the complement gate is now disabled for every field until the next Reclaim")
	}
	// And the gate itself must still arm, which is the thing anyone would actually
	// notice missing.
	prev := admitGateMode
	admitGateMode = gateForceComplement
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)
	h.buildComplementGate(s, Filter{Op: FilterGte, Field: "always", Value: NewInt(1)}, 10)
	if !s.gate.active() {
		t.Error("no complement gate after a rebuild with tombstones")
	}
	s.gate.disable()
	// The partial field must STILL be refused — the fix must not have widened
	// totality into something it is not.
	if h.payloadIdx.fieldTotalNumeric("opt", h.arena.Size()) {
		t.Error("`opt` reads as total after the rebuild — the tombstone fix over-claimed coverage")
	}
}

// TestBuildConcurrentRefusesAPopulatedPayloadIndex pins the invariant
// BuildConcurrent's bulk placement loop rests on: it writes slots WITHOUT
// reindexing them, which is sound only because the build carries no payloads and
// starts from an index with nothing to keep in sync. If that ever stops holding,
// the column sidecar would answer for a reused slot with its pre-build value —
// a wrong row, not a slow one — so the precondition is checked rather than
// assumed.
func TestBuildConcurrentRefusesAPopulatedPayloadIndex(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	ids := []uint64{1, 2, 3}
	vecs := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	// A payload index carrying state from somewhere other than this build.
	h.payloadIdx.reindex(42, Metadata{"v": NewInt(7)})
	if err := h.BuildConcurrent(ids, vecs, 2); !errors.Is(err, ErrBuildNonEmpty) {
		t.Fatalf("BuildConcurrent on a populated payload index returned %v, want ErrBuildNonEmpty", err)
	}
	// Cleared, it proceeds — so the guard is about the index's contents and not
	// an unconditional refusal.
	h.payloadIdx.reindex(42, nil)
	if !h.payloadIdx.isEmpty() {
		t.Fatal("clearing the only indexed slot did not empty the index")
	}
	if err := h.BuildConcurrent(ids, vecs, 2); err != nil {
		t.Fatalf("BuildConcurrent on an empty index: %v", err)
	}
}

// TestPayloadIndexHasNoSlotsOutsideTheIdMap pins link (1) of the totality
// argument in fieldTotalNumeric: `posts >= arena.Size()` only proves coverage of
// the LIVE slots if no posting points at a slot the arena has already forgotten.
// Every path that removes an id from idMap must drop its postings first.
//
// Without this test the totality check is comparing a count against a bound it
// merely assumes; with it, a future path that frees a slot without reindexing
// fails here rather than by silently admitting a filtered-out point.
func TestPayloadIndexHasNoSlotsOutsideTheIdMap(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	vec := func(i int) []float32 { return []float32{float32(i), 1, 2, 3} }
	const n = 300
	for i := 0; i < n; i++ {
		if _, _, err := h.Insert(uint64(i+1), vec(i), 0, Metadata{"v": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("insert: %v", err)
		}
	}
	// A mixed workload over every mutation that can retire or rewrite a slot.
	for i := 0; i < n; i += 3 {
		if _, err := h.Delete(uint64(i+1), CASCond{}); err != nil {
			t.Fatalf("delete: %v", err)
		}
	}
	for i := 0; i < n; i += 3 { // upsert over the dead slots: reclaim + reinsert
		if _, _, err := h.Insert(uint64(i+1), vec(i), 0, Metadata{"v": NewInt(int64(i * 2))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("re-insert: %v", err)
		}
	}
	for i := 1; i < n; i += 7 {
		if _, _, _, err := h.ClearPayload(uint64(i+1), CASCond{}); err != nil {
			t.Fatalf("clear payload: %v", err)
		}
	}
	for i := 2; i < n; i += 11 {
		if _, _, _, err := h.SetPayload(uint64(i+1), Metadata{"v": NewInt(int64(i + 1000))}, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("set payload: %v", err)
		}
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	inMap := make(map[uint32]bool, h.arena.Size())
	for _, slot := range h.arena.idMap {
		inMap[slot] = true
	}
	counted := 0
	for field, byKey := range h.payloadIdx.fields {
		for key, set := range byKey {
			if len(set) == 0 {
				t.Errorf("field %q key %v has an EMPTY posting set — drop left it behind, so the key-count "+
					"pre-check's 'every key has at least one slot' premise is false", field, key)
			}
			for slot := range set {
				counted++
				if !inMap[slot] {
					t.Fatalf("field %q posts slot %d, which is not in arena.idMap — a slot was freed without "+
						"dropping its payload postings, and fieldTotalNumeric can now over-count", field, slot)
				}
			}
		}
	}
	if counted == 0 {
		t.Fatal("no postings survived the workload — the assertions above proved nothing")
	}
	// The incremental counter must agree with a full recount, or every totality
	// decision is made against a number that drifted.
	recount := 0
	for key, set := range h.payloadIdx.fields["v"] {
		if key.kind == ValueInt || key.kind == ValueFloat {
			recount += len(set)
		}
	}
	if got := h.payloadIdx.posts["v"].num; got != recount {
		t.Errorf("posts[v].num = %d, recount = %d — the incremental counter drifted", got, recount)
	}
}
