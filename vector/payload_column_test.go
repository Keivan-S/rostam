// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"math"
	"math/rand"
	"testing"
	"time"
)

// The numeric column sidecar's tests. The column REPLACES the predicate outright
// on the paths it covers, so every test here is ultimately the same assertion:
// column and predicate must agree on every slot, including the three ways a slot
// can have no numeric value (absent field, wrong kind, NaN).

// columnCorpus builds a corpus that exercises every way the column can disagree
// with the predicate if it is built or maintained wrongly: a field on every
// point, a field on only some, a field whose values are sometimes strings, and a
// field carrying NaN.
func columnCorpus(t testing.TB, n, dim int) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{
		Dim: dim, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, Metric: L2,
		// Pinned to 1 so the planner always declines filter-first and the GRAPH
		// path — the only place an admission oracle runs — is what gets compared.
		FilterFirstThreshold: 1,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(31337))
	base := time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		meta := Metadata{
			"seq":  NewInt(int64(i)), //nolint:gosec // bounded loop index
			"when": NewInt(base + int64(i)*60_000),
		}
		switch i % 13 {
		case 0:
			// absent: `score` missing entirely
		case 1:
			meta["score"] = NewString("not-a-number") // wrong kind
		case 2:
			meta["score"] = NewFloat(math.NaN())
		default:
			meta["score"] = NewFloat(float64(i%29) / 2)
		}
		if i%7 == 0 {
			meta["flag"] = NewBool(true) // a non-numeric field a column must refuse
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	for i := 0; i < n; i += 19 { // tombstones: admission must still run liveness first
		if _, err := h.Delete(uint64(i+1), CASCond{}); err != nil {
			t.Fatalf("delete: %v", err)
		}
	}
	return h
}

func columnFilters() []struct {
	name string
	f    Filter
} {
	return []struct {
		name string
		f    Filter
	}{
		{"gte/dense", Filter{Op: FilterGte, Field: "seq", Value: NewInt(50)}},
		{"gt/dense", Filter{Op: FilterGt, Field: "seq", Value: NewInt(1_800)}},
		{"lt/dense", Filter{Op: FilterLt, Field: "seq", Value: NewInt(1_900)}},
		{"lte/dense", Filter{Op: FilterLte, Field: "seq", Value: NewInt(20)}},
		{"gte/holes", Filter{Op: FilterGte, Field: "score", Value: NewFloat(4)}},
		{"lte/holes", Filter{Op: FilterLte, Field: "score", Value: NewFloat(4)}},
		{"gt/holes", Filter{Op: FilterGt, Field: "score", Value: NewFloat(-1)}},
		{"band/and", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "seq", Value: NewInt(200)},
			{Op: FilterLt, Field: "seq", Value: NewInt(1_500)},
		}}},
		{"and/two-fields", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "seq", Value: NewInt(100)},
			{Op: FilterLte, Field: "score", Value: NewFloat(6)},
		}}},
		{"dt/gte", Filter{Op: FilterDtGte, Field: "when", Value: NewString("2026-03-01T10:00:00Z")}},
		{"dt/and", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterDtGte, Field: "when", Value: NewString("2026-03-01T02:00:00Z")},
			{Op: FilterDtLt, Field: "when", Value: NewString("2026-03-01T20:00:00Z")},
		}}},
		{"nested/and", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "seq", Value: NewInt(10)},
			{Op: FilterAnd, And: []Filter{{Op: FilterLt, Field: "seq", Value: NewInt(1_000)}}},
		}}},
	}
}

// TestColumnGateEquivalence is the primary assertion: turning the column oracle
// on must change nothing observable — not the result sets, not the filter-reject
// tally — for every range shape, over a corpus whose `score` field is absent,
// string-valued, or NaN on a fifth of the points.
func TestColumnGateEquivalence(t *testing.T) {
	h := columnCorpus(t, 2_000, 12)
	queries := gateQueries(gateQueryN(120), 12)
	for _, tc := range columnFilters() {
		t.Run(tc.name, func(t *testing.T) {
			off := runGateArm(t, h, queries, tc.f, gateOff, false)
			on := runGateArm(t, h, queries, tc.f, gateAuto, false)
			if on.gates != uint64(len(queries)) {
				t.Fatalf("%s: %d of %d queries armed a gate — the column path was not exercised",
					tc.name, on.gates, len(queries))
			}
			if nDiff, rejectsDiffer := diffGateArms(off, on); nDiff != 0 || rejectsDiffer {
				t.Fatalf("%s: the column oracle changed results (%d queries differ) or rejects (%d vs %d)",
					tc.name, nDiff, off.rejects, on.rejects)
			}
		})
	}
}

// TestColumnMatchesPredicateSlotBySlot compares the two oracles directly rather
// than through search results, so a disagreement on a slot that never reaches
// anyone's top-k still fails. This is the assertion the search-level equivalence
// test can only sample.
func TestColumnMatchesPredicateSlotBySlot(t *testing.T) {
	h := columnCorpus(t, 3_000, 8)
	h.mu.RLock()
	defer h.mu.RUnlock()
	now := uint64(h.now())
	capacity := h.arena.Capacity()

	for _, tc := range columnFilters() {
		pred, err := tc.f.Compile()
		if err != nil {
			t.Fatalf("%s: compile: %v", tc.name, err)
		}
		terms, ok := h.payloadIdx.collectColumnTerms(tc.f, capacity, -1, nil)
		if !ok {
			t.Fatalf("%s: not column-expressible", tc.name)
		}
		var g admitGate
		g.armColumns(terms)
		checked, accepted := 0, 0
		for slot := 0; slot < capacity; slot++ {
			u := uint32(slot) //nolint:gosec // bounded by capacity
			if h.tombstoned[u] || h.isExpiredAt(u, now) {
				continue
			}
			want := pred(h.liveMeta(u, now))
			got := g.testCols(u)
			checked++
			if want {
				accepted++
			}
			if got != want {
				t.Fatalf("%s slot %d: column=%v predicate=%v (meta %v)", tc.name, slot, got, want, h.liveMeta(u, now))
			}
		}
		if checked == 0 || accepted == 0 || accepted == checked {
			t.Fatalf("%s: %d/%d accepted — the comparison is degenerate", tc.name, accepted, checked)
		}
	}
}

// TestColumnHoldsMatchesOrderingHoldsFloat pins the ONE place the column path
// re-implements the predicate's arithmetic instead of calling it. columnTerm.
// holds spells the four comparisons out with Go's float64 operators for speed;
// this is the exhaustive check that spelling them out did not change what they
// mean — over ±0, ±Inf, NaN on either side, and ordinary values.
func TestColumnHoldsMatchesOrderingHoldsFloat(t *testing.T) {
	vals := []float64{
		math.NaN(), math.Inf(-1), math.Inf(1),
		-1e308, -1, math.Copysign(0, -1), 0.0, 1e-300, 1, 2, 1e308,
	}
	ops := []FilterOp{FilterGt, FilterGte, FilterLt, FilterLte}
	for _, op := range ops {
		for _, bound := range vals {
			term := columnTerm{vals: vals, bound: bound, op: op}
			for i, v := range vals {
				got := term.holds(uint32(i)) //nolint:gosec // small loop index
				want := orderingHoldsFloat(op, v, bound)
				if got != want {
					t.Errorf("op %d: holds(%v vs %v) = %v, orderingHoldsFloat = %v", op, v, bound, got, want)
				}
			}
			// Past the end of the column: no payload, hence rejected.
			if term.holds(uint32(len(vals))) { //nolint:gosec // small constant
				t.Errorf("op %d: a slot past the column end was admitted", op)
			}
		}
	}
}

// TestColumnTracksPayloadMutations is the maintenance test. The column is a
// SECOND copy of data the arena already holds, so the failure mode it introduces
// is staleness: a payload rewritten after the column was built, answered from
// the old value. Every mutator that can change a numeric field is exercised
// AFTER the column exists, and the column is then re-compared against the
// predicate slot by slot.
func TestColumnTracksPayloadMutations(t *testing.T) {
	h := columnCorpus(t, 800, 8)
	f := Filter{Op: FilterGte, Field: "seq", Value: NewInt(400)}

	// Force the column into existence.
	if _, err := h.SearchFiltered(gateQueries(1, 8)[0], 5, f); err != nil {
		t.Fatalf("warm search: %v", err)
	}
	h.mu.RLock()
	if _, ok := h.payloadIdx.cols["seq"]; !ok {
		h.mu.RUnlock()
		t.Fatal("precondition: the `seq` column was not built")
	}
	h.mu.RUnlock()

	// Now mutate underneath it, through every path that reaches reindex.
	for i := 0; i < 800; i += 5 { // patch the numeric field
		if _, _, _, err := h.SetPayload(uint64(i+1), Metadata{"seq": NewInt(int64(900 - i))}, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			if i%19 == 0 {
				continue // tombstoned by columnCorpus; SetPayload legitimately fails
			}
			t.Fatalf("set payload %d: %v", i, err)
		}
	}
	for i := 1; i < 800; i += 11 { // remove the field entirely
		if _, _, _, err := h.ClearPayload(uint64(i+1), CASCond{}); err != nil {
			if i%19 == 0 {
				continue
			}
			t.Fatalf("clear payload %d: %v", i, err)
		}
	}
	for i := 2; i < 800; i += 13 { // replace the field with a non-numeric value
		if _, _, _, err := h.OverwritePayload(uint64(i+1), Metadata{"seq": NewString("x")}, nil, CASCond{}); err != nil {
			if i%19 == 0 {
				continue
			}
			t.Fatalf("overwrite payload %d: %v", i, err)
		}
	}
	for i := 3; i < 800; i += 17 { // delete, then resurrect the id on a fresh slot
		if _, err := h.Delete(uint64(i+1), CASCond{}); err != nil {
			t.Fatalf("delete %d: %v", i, err)
		}
		v := make([]float32, 8)
		for d := range v {
			v[d] = float32(i + d)
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, Metadata{"seq": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("resurrect %d: %v", i, err)
		}
	}

	pred, err := f.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	terms, ok := h.payloadIdx.collectColumnTerms(f, h.arena.Capacity(), -1, nil)
	if !ok {
		t.Fatal("not column-expressible after the mutations")
	}
	var g admitGate
	g.armColumns(terms)
	now := uint64(h.now())
	for slot := 0; slot < h.arena.Capacity(); slot++ {
		u := uint32(slot) //nolint:gosec // bounded by capacity
		if h.tombstoned[u] || h.isExpiredAt(u, now) {
			continue
		}
		if got, want := g.testCols(u), pred(h.liveMeta(u, now)); got != want {
			t.Fatalf("slot %d after mutation: column=%v predicate=%v (meta %v)",
				slot, got, want, h.liveMeta(u, now))
		}
	}
}

// TestColumnGateVetoedByPerKeyTTL pins the one thing a column cannot see. A key
// past its per-key deadline is hidden by liveMeta but still sits in the arena's
// metadata map — and therefore still sits in the column — so admitting from the
// column would return a point whose filtered-on key has logically expired.
func TestColumnGateVetoedByPerKeyTTL(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 0; i < 200; i++ {
		v := []float32{float32(i), 1, 2, 3}
		if _, _, err := h.Insert(uint64(i+1), v, 0, Metadata{"seq": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("insert: %v", err)
		}
	}
	f := Filter{Op: FilterGte, Field: "seq", Value: NewInt(50)}
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	armed := h.buildColumnGate(s, f)
	h.mu.RUnlock()
	if !armed {
		t.Fatal("precondition: a plain numeric range must arm the column oracle")
	}
	s.gate.disable()

	// Give ONE slot a per-key deadline. The veto is collection-wide by design —
	// the check is a single counter, not a per-slot test — so one is enough.
	if _, _, _, err := h.SetPayload(1, Metadata{"seq": NewInt(0)}, map[string]int64{"seq": 60_000}, CASCond{}); err != nil {
		t.Fatalf("set payload with key ttl: %v", err)
	}
	h.mu.RLock()
	armed = h.buildColumnGate(s, f)
	h.mu.RUnlock()
	if armed {
		t.Error("the column oracle armed on a collection carrying a per-key TTL — an expired key would still be admitted")
	}
	s.gate.disable()
}

// TestColumnDeclinesWhatItCannotExpress pins the all-or-nothing boundary. A
// filter with any leaf outside the numeric-range family must get NO column gate,
// because a partial column answer still has to consult the predicate and the
// predicate is the cost being removed.
func TestColumnDeclinesWhatItCannotExpress(t *testing.T) {
	h := columnCorpus(t, 500, 8)
	h.mu.RLock()
	defer h.mu.RUnlock()
	capacity := h.arena.Capacity()
	declined := []struct {
		name string
		f    Filter
	}{
		{"eq", Filter{Op: FilterEq, Field: "seq", Value: NewInt(3)}},
		{"in", Filter{Op: FilterIn, Field: "seq", Value: NewInts([]int64{1, 2})}},
		{"ne", Filter{Op: FilterNe, Field: "seq", Value: NewInt(3)}},
		{"or", Filter{Op: FilterOr, Or: []Filter{{Op: FilterGte, Field: "seq", Value: NewInt(3)}}}},
		{"not", Filter{Op: FilterNot, Not: &Filter{Op: FilterGte, Field: "seq", Value: NewInt(3)}}},
		{"and-with-eq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "seq", Value: NewInt(3)},
			{Op: FilterEq, Field: "flag", Value: NewBool(true)},
		}}},
		{"string-bound", Filter{Op: FilterGte, Field: "seq", Value: NewString("abc")}},
		{"nan-bound", Filter{Op: FilterGte, Field: "seq", Value: NewFloat(math.NaN())}},
		{"absent-field", Filter{Op: FilterGte, Field: "nope", Value: NewInt(1)}},
		{"non-numeric-field", Filter{Op: FilterGte, Field: "flag", Value: NewInt(1)}},
		{"content-field", Filter{Op: FilterGte, Field: contentField, Value: NewInt(1)}},
		{"empty-and", Filter{Op: FilterAnd}},
		{"bad-datetime", Filter{Op: FilterDtGte, Field: "when", Value: NewString("nope")}},
	}
	for _, tc := range declined {
		if _, ok := h.payloadIdx.collectColumnTerms(tc.f, capacity, -1, nil); ok {
			t.Errorf("%s: collectColumnTerms accepted a filter it cannot answer exactly", tc.name)
		}
	}
}

// TestColumnUnderConcurrentReadersAndWriters is the race-detector test for the
// sidecar's one genuinely new concurrency surface: the column is BUILT on the
// query path (under the owner's read lock, which excludes writers but not other
// readers) and MUTATED IN PLACE on the write path.
//
// Two claims are under test. First, that colsMu is enough for the build — two
// readers racing to create the same field's column, or different fields'
// columns, both write the same map. Second, that in-place maintenance is safe
// without any synchronisation of its own, because h.mu already orders every
// writer against every reader; if that were ever untrue, a writer's
// updateColumns store would race a traversal's testCols load and the detector
// would say so.
//
// Run it with -race; without the detector it only proves nothing panics.
func TestColumnUnderConcurrentReadersAndWriters(t *testing.T) {
	h, err := newHNSW(Config{Dim: 8, M: 8, EfConstruction: 60, EfSearch: 32, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	vecFor := func(i int) []float32 {
		v := make([]float32, 8)
		for d := range v {
			v[d] = float32((i*7+d*13)%101) / 50
		}
		return v
	}
	const seed = 600
	for i := 0; i < seed; i++ {
		if _, _, err := h.Insert(uint64(i+1), vecFor(i), 0,
			Metadata{"a": NewInt(int64(i)), "b": NewFloat(float64(i % 37))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("seed insert: %v", err)
		}
	}

	filters := []Filter{
		{Op: FilterGte, Field: "a", Value: NewInt(100)},
		{Op: FilterLt, Field: "b", Value: NewFloat(20)},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "a", Value: NewInt(50)},
			{Op: FilterLte, Field: "b", Value: NewFloat(30)},
		}},
	}
	// Snapshot the oracle counter across the concurrent phase. Without this the
	// test degenerates the moment a planner change stops routing these filters to
	// the column path: the goroutines would still race, but not over the column
	// build this test exists to exercise, and it would pass forever.
	gatesBefore := h.Stats().ColumnGates
	done := make(chan struct{})
	errs := make(chan error, 16)
	// Readers: several at once, so the lazy build genuinely races itself.
	for r := 0; r < 6; r++ {
		go func(r int) {
			q := vecFor(r * 3)
			for {
				select {
				case <-done:
					return
				default:
				}
				if _, err := h.SearchFiltered(q, 10, filters[r%len(filters)]); err != nil {
					errs <- err
					return
				}
			}
		}(r)
	}
	// Writers: inserts (new slots, column growth) and payload rewrites (in-place
	// column stores), both under the write lock.
	for w := 0; w < 2; w++ {
		go func(w int) {
			for i := seed + w; i < seed+400; i += 2 {
				select {
				case <-done:
					return
				default:
				}
				if _, _, err := h.Insert(uint64(i+1), vecFor(i), 0,
					Metadata{"a": NewInt(int64(i)), "b": NewFloat(float64(i % 37))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
					errs <- err
					return
				}
				if _, _, _, err := h.SetPayload(uint64(i%seed+1), Metadata{"a": NewInt(int64(i * 3))}, nil, CASCond{}); err != nil { //nolint:gosec // bounded
					errs <- err
					return
				}
			}
		}(w)
	}
	time.Sleep(300 * time.Millisecond)
	close(done)
	select {
	case err := <-errs:
		t.Fatalf("concurrent worker: %v", err)
	default:
	}
	if armed := h.Stats().ColumnGates - gatesBefore; armed == 0 {
		t.Fatal("no column gate armed during the concurrent phase — the readers never built or resolved " +
			"a column, so the race this test exists for was never run")
	}

	// And the column must still agree with the predicate afterwards.
	f := filters[2]
	pred, err := f.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	terms, ok := h.payloadIdx.collectColumnTerms(f, h.arena.Capacity(), -1, nil)
	if !ok {
		t.Fatal("not column-expressible after the concurrent workload")
	}
	var g admitGate
	g.armColumns(terms)
	now := uint64(h.now())
	for slot := 0; slot < h.arena.Capacity(); slot++ {
		u := uint32(slot) //nolint:gosec // bounded by capacity
		if h.tombstoned[u] || h.isExpiredAt(u, now) {
			continue
		}
		if got, want := g.testCols(u), pred(h.liveMeta(u, now)); got != want {
			t.Fatalf("slot %d: column=%v predicate=%v", slot, got, want)
		}
	}
}

// manyFieldIndex builds a small index whose every point carries `nfields`
// numeric fields, so the column cap and its eviction policy can be exercised
// without a large corpus.
func manyFieldIndex(t testing.TB, nfields, npoints int, cfg Config) *hnsw {
	t.Helper()
	cfg.Dim, cfg.M, cfg.EfConstruction, cfg.EfSearch, cfg.Seed, cfg.Metric = 4, 8, 50, 16, 1, L2
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 0; i < npoints; i++ {
		meta := make(Metadata, nfields)
		for f := 0; f < nfields; f++ {
			meta[fieldNameFor(f)] = NewInt(int64(i + f)) //nolint:gosec // bounded
		}
		if _, _, err := h.Insert(uint64(i+1), []float32{float32(i), 1, 2, 3}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	return h
}

// TestColumnCapEvictsTheColdest pins the residency bound AND the policy. The cap
// itself is the memory guarantee — never more than maxNumColumns columns — but
// the policy is what makes the cap usable: a first-come-first-served cap would
// let eight ad-hoc queries own every slot for the life of the process, so the
// coldest column is evicted instead and the request always succeeds.
func TestColumnCapEvictsTheColdest(t *testing.T) {
	nfields := maxNumColumns + 3
	h := manyFieldIndex(t, nfields, 100, Config{})
	h.mu.RLock()
	defer h.mu.RUnlock()
	capacity := h.arena.Capacity()

	for f := 0; f < nfields; f++ {
		if _, ok := h.payloadIdx.ensureColumn(fieldNameFor(f), capacity, -1); !ok {
			t.Fatalf("field %d: ensureColumn declined — eviction should always make room", f)
		}
		if got := len(h.payloadIdx.cols); got > maxNumColumns {
			t.Fatalf("field %d: %d columns resident, cap is %d", f, got, maxNumColumns)
		}
	}
	if got := len(h.payloadIdx.cols); got != maxNumColumns {
		t.Fatalf("%d columns resident at the end, want exactly the cap of %d", got, maxNumColumns)
	}
	// The survivors must be the LAST maxNumColumns fields touched, not the first.
	for f := 0; f < nfields-maxNumColumns; f++ {
		if _, ok := h.payloadIdx.cols[fieldNameFor(f)]; ok {
			t.Errorf("field %d survived; the coldest columns should have been evicted first", f)
		}
	}
	for f := nfields - maxNumColumns; f < nfields; f++ {
		if _, ok := h.payloadIdx.cols[fieldNameFor(f)]; !ok {
			t.Errorf("field %d was evicted; it is among the most recently used", f)
		}
	}

	// Re-touching an old column makes it hot: it must then survive the next
	// eviction while the field that was coldest goes instead.
	hot := fieldNameFor(nfields - maxNumColumns)
	if _, ok := h.payloadIdx.ensureColumn(hot, capacity, -1); !ok {
		t.Fatalf("re-resolving %s failed", hot)
	}
	if _, ok := h.payloadIdx.ensureColumn(fieldNameFor(0), capacity, -1); !ok {
		t.Fatal("rebuilding an evicted field failed")
	}
	if _, ok := h.payloadIdx.cols[hot]; !ok {
		t.Errorf("%s was evicted right after being used — lastUse is not being stamped", hot)
	}
}

// TestColumnBytesAreChargedAgainstMaxBytes pins the quota hole the sidecar would
// otherwise open. Columns are allocated on the READ path, which MaxBytes never
// used to constrain, so query traffic alone could push a byte-capped collection
// past its ceiling. Both halves of the fix are asserted: the build refuses when
// there is no room, and the columns that DO exist are charged so inserts start
// failing at the right point.
func TestColumnBytesAreChargedAgainstMaxBytes(t *testing.T) {
	// Room for the points, but nowhere near enough for a column.
	h := manyFieldIndex(t, 2, 200, Config{MaxBytes: 1 << 20})
	h.mu.RLock()
	used, capacity := h.bytesUsed, h.arena.Capacity()
	h.mu.RUnlock()

	tight := int64(0)
	if want := int64(capacity) * columnBytesPerSlot; want > 0 {
		tight = want - 1 // one byte short of affording a single column
	}
	h.mu.RLock()
	_, ok := h.payloadIdx.ensureColumn("fa", capacity, tight)
	h.mu.RUnlock()
	if ok {
		t.Fatal("a column was built with less budget than it needs")
	}
	if got := h.payloadIdx.columnBytes(); got != 0 {
		t.Fatalf("columnBytes = %d after a refused build, want 0", got)
	}

	// With room, it builds — and the bytes show up in the counter.
	h.mu.RLock()
	_, ok = h.payloadIdx.ensureColumn("fa", capacity, -1)
	h.mu.RUnlock()
	if !ok {
		t.Fatal("an unbudgeted build was refused")
	}
	want := int64(capacity) * columnBytesPerSlot
	if got := h.payloadIdx.columnBytes(); got < want {
		t.Fatalf("columnBytes = %d after building a %d-slot column, want at least %d", got, capacity, want)
	}

	// And the insert path sees them — but it RECLAIMS rather than refuses. A
	// MaxBytes set below usage+columns means the column is the only thing standing
	// between this write and its budget, and a read-side cache does not outrank a
	// durable write: the insert drops the sidecar and succeeds.
	h.mu.Lock()
	h.cfg.MaxBytes = used + h.payloadIdx.columnBytes()/2
	h.mu.Unlock()
	dropsBefore := h.Stats().ColumnDrops
	if _, _, err := h.Insert(9_999, []float32{1, 2, 3, 4}, 0, Metadata{"fa": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert under column memory pressure returned %v, want success — "+
			"a read-side cache must yield to a durable write, not refuse it", err)
	}
	if got := h.payloadIdx.columnBytes(); got != 0 {
		t.Errorf("columnBytes = %d after the reclaiming insert, want 0", got)
	}
	if drops := h.Stats().ColumnDrops - dropsBefore; drops != 1 {
		t.Errorf("ColumnDrops moved by %d, want 1", drops)
	}
}

// TestColumnsYieldToWritesUnderMemoryPressure is the availability property, and
// it is about a failure mode the first version of the quota fix introduced: a
// single read-path column build could push a byte-capped collection into
// refusing EVERY subsequent insert, permanently, because nothing in the write
// path could release the column's memory.
//
// A column is a pure CACHE — every byte in it is reconstructible from the
// payload index by the next query that wants it — while an insert is durable
// data the caller cannot reconstruct. So writes win: the insert reclaims the
// sidecar and proceeds, and ErrCollectionFull goes back to meaning "the
// collection's own data filled its budget".
func TestColumnsYieldToWritesUnderMemoryPressure(t *testing.T) {
	h := manyFieldIndex(t, 2, 200, Config{MaxBytes: 1 << 20})
	h.mu.RLock()
	used, capacity := h.bytesUsed, h.arena.Capacity()
	_, ok := h.payloadIdx.ensureColumn("fa", capacity, -1)
	h.mu.RUnlock()
	if !ok {
		t.Fatal("precondition: the column must build")
	}
	colBytes := h.payloadIdx.columnBytes()
	if colBytes == 0 {
		t.Fatal("precondition: the column must have a non-zero footprint")
	}

	// A ceiling that the points fit under but points+column do not.
	h.mu.Lock()
	h.cfg.MaxBytes = used + colBytes/2
	h.mu.Unlock()

	// EVERY subsequent insert must succeed, not just the first. The regression
	// this pins was a permanent rejection, so one success proves little.
	for i := 0; i < 5; i++ {
		if _, _, err := h.Insert(uint64(20_000+i), []float32{float32(i), 1, 2, 3}, 0,
			Metadata{"fa": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("insert %d under column memory pressure: %v — reads are permanently displacing writes", i, err)
		}
	}
	if got := h.payloadIdx.columnBytes(); got != 0 {
		t.Errorf("columnBytes = %d after the reclaiming inserts, want 0", got)
	}

	// ErrCollectionFull must still fire for GENUINE exhaustion — the fix must not
	// have turned the quota off.
	h.mu.Lock()
	h.cfg.MaxBytes = 1
	h.mu.Unlock()
	if _, _, err := h.Insert(30_000, []float32{1, 2, 3, 4}, 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("insert into a genuinely exhausted collection returned %v, want ErrCollectionFull", err)
	}
}

// TestColumnEvictionIsNotPreemptedByTheBudget pins the ORDER of the two controls
// on the cap. `budget` already has the resident columns subtracted out of it, so
// testing it before choosing a victim made a byte-capped collection refuse the
// ninth field even though the swap is memory-NEUTRAL — quietly restoring the
// first-come-forever policy that eviction exists to prevent, for every
// collection with MaxBytes set.
func TestColumnEvictionIsNotPreemptedByTheBudget(t *testing.T) {
	nfields := maxNumColumns + 1
	h := manyFieldIndex(t, nfields, 100, Config{})
	h.mu.RLock()
	defer h.mu.RUnlock()
	capacity := h.arena.Capacity()

	for f := 0; f < maxNumColumns; f++ {
		if _, ok := h.payloadIdx.ensureColumn(fieldNameFor(f), capacity, -1); !ok {
			t.Fatalf("precondition: field %d must build", f)
		}
	}
	if got := len(h.payloadIdx.cols); got != maxNumColumns {
		t.Fatalf("precondition: %d columns resident, want the cap of %d", got, maxNumColumns)
	}

	// A budget of ZERO — the collection has no headroom at all. The swap is still
	// memory-neutral (all columns are the same size here), so it must go ahead.
	before := h.payloadIdx.columnBytes()
	if _, ok := h.payloadIdx.ensureColumn(fieldNameFor(nfields-1), capacity, 0); !ok {
		t.Fatal("a memory-NEUTRAL swap at the cap was refused on a zero budget — the budget test runs " +
			"before eviction, so the resident columns it already subtracted are counted against the swap")
	}
	if got := h.payloadIdx.columnBytes(); got != before {
		t.Errorf("columnBytes moved from %d to %d across a same-size swap, want unchanged", before, got)
	}
	if got := len(h.payloadIdx.cols); got != maxNumColumns {
		t.Errorf("%d columns resident after the swap, want %d", got, maxNumColumns)
	}
	if _, ok := h.payloadIdx.cols[fieldNameFor(nfields-1)]; !ok {
		t.Error("the new field is not resident after a successful swap")
	}

	// A budget that cannot cover the swap's SHORTFALL must still refuse, or the
	// ordering fix has simply disabled the quota. Force a shortfall by asking for
	// a column four times the size of the victim it would replace — over a field
	// that really exists (fa was the coldest and has just been evicted), so the
	// refusal can only be the budget and not a missing posting list.
	evicted := fieldNameFor(0)
	if _, resident := h.payloadIdx.cols[evicted]; resident {
		t.Fatalf("precondition: %s should have been the eviction victim", evicted)
	}
	if _, ok := h.payloadIdx.ensureColumn(evicted, capacity*4, 0); ok {
		t.Error("a swap that GROWS the footprint fourfold was allowed on a zero budget — the quota is not being applied")
	}
}

// TestColumnGateRefusesToBuildOverQuota is the end-to-end form of the half
// above: a SEARCH on a byte-capped collection must not be able to allocate a
// column that crosses the ceiling. It still returns correct results — the
// predicate answers instead — which is the whole point of the column being an
// optimization rather than a mechanism.
func TestColumnGateRefusesToBuildOverQuota(t *testing.T) {
	h := manyFieldIndex(t, 2, 300, Config{FilterFirstThreshold: 1})
	f := Filter{Op: FilterGte, Field: "fa", Value: NewInt(50)}
	q := []float32{5, 1, 2, 3}

	// Squeeze MaxBytes to just above what the points already use.
	h.mu.Lock()
	h.cfg.MaxBytes = h.bytesUsed + 8
	h.mu.Unlock()

	before := h.Stats()
	got, err := h.SearchFiltered(q, 10, f)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	after := h.Stats()
	if after.ColumnGates != before.ColumnGates {
		t.Error("a column gate armed on a collection with no byte headroom — the read path escaped MaxBytes")
	}
	if h.payloadIdx.columnBytes() != 0 {
		t.Errorf("columnBytes = %d, want 0 — a column was allocated over quota", h.payloadIdx.columnBytes())
	}
	// Correctness is unaffected: the same query with headroom must agree.
	h.mu.Lock()
	h.cfg.MaxBytes = 0
	h.mu.Unlock()
	want, err := h.SearchFiltered(q, 10, f)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("quota-refused search returned %d results, column-accelerated returned %d", len(got), len(want))
	}
	for i := range got {
		if got[i].ID != want[i].ID {
			t.Fatalf("result %d: %d vs %d — refusing the column changed the answer", i, got[i].ID, want[i].ID)
		}
	}
}

// TestAdmitGateDisableClearsDeclinedColumnTerms pins a RETENTION bug, not a
// correctness one, and it is the kind that never shows up in a test that only
// checks answers.
//
// buildColumnGate accumulates terms into the pooled scratch's own slice and, on
// a decline, truncates it to zero LENGTH — leaving the resolved columnTerms, each
// holding a whole field's []float64, live in [len,cap). disable() must therefore
// clear over the CAPACITY; a length-bounded clear is a no-op on exactly this
// path and pins tens of megabytes of column inside an idle sync.Pool entry,
// where even Reclaim (which nils payloadIndex.cols) cannot reach it.
func TestAdmitGateDisableClearsDeclinedColumnTerms(t *testing.T) {
	// `fa` is numeric and columnisable; `sb` is string-valued, so it passes the
	// SHAPE check (a numeric bound against a plain field) and then fails when
	// ensureColumn finds no numeric postings — a decline AFTER a term was
	// appended, which is the state the bug needs.
	h, err := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 0; i < 100; i++ {
		meta := Metadata{"fa": NewInt(int64(i)), "sb": NewString("x")} //nolint:gosec // bounded
		if _, _, err := h.Insert(uint64(i+1), []float32{float32(i), 1, 2, 3}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterGte, Field: "fa", Value: NewInt(1)},
		{Op: FilterGte, Field: "sb", Value: NewInt(1)},
	}}
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	armed := h.buildColumnGate(s, f)
	h.mu.RUnlock()
	if armed {
		t.Fatal("precondition: the filter must DECLINE (its second leaf has no numeric column)")
	}
	if cap(s.gate.cols) == 0 {
		t.Fatal("precondition: the decline must have left a term in the slice's capacity")
	}
	s.gate.disable()
	for i, term := range s.gate.cols[:cap(s.gate.cols)] {
		if term.vals != nil {
			t.Fatalf("term %d still holds a %d-element column after disable() — the clear is bounded by "+
				"len (which is 0 here) instead of cap, so the column is retained by the pooled scratch",
				i, len(term.vals))
		}
	}
}

func fieldNameFor(i int) string { return "f" + string(rune('a'+i)) }

// TestColumnCoversSlotsAddedAfterTheBuild pins growth. The column is sized to
// the arena's capacity when it is built; points inserted afterwards land at
// higher slots, and a column that did not grow would answer for them from the
// out-of-range branch (reject) regardless of their actual values.
func TestColumnCoversSlotsAddedAfterTheBuild(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	insert := func(i int) {
		if _, _, err := h.Insert(uint64(i+1), []float32{float32(i), 1, 2, 3}, 0,
			Metadata{"seq": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	for i := 0; i < 50; i++ {
		insert(i)
	}
	h.mu.RLock()
	if _, ok := h.payloadIdx.ensureColumn("seq", h.arena.Capacity(), -1); !ok {
		h.mu.RUnlock()
		t.Fatal("column build failed")
	}
	before := len(h.payloadIdx.cols["seq"].vals)
	h.mu.RUnlock()

	for i := 50; i < 400; i++ {
		insert(i)
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	col := h.payloadIdx.cols["seq"].vals
	if len(col) <= before {
		t.Fatalf("column length %d did not grow past the build-time %d", len(col), before)
	}
	f := Filter{Op: FilterGte, Field: "seq", Value: NewInt(300)}
	pred, err := f.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	terms, ok := h.payloadIdx.collectColumnTerms(f, h.arena.Capacity(), -1, nil)
	if !ok {
		t.Fatal("not column-expressible")
	}
	var g admitGate
	g.armColumns(terms)
	now := uint64(h.now())
	matched := 0
	for slot := 0; slot < h.arena.Capacity(); slot++ {
		u := uint32(slot) //nolint:gosec // bounded by capacity
		if h.tombstoned[u] {
			continue
		}
		want := pred(h.liveMeta(u, now))
		if want {
			matched++
		}
		if got := g.testCols(u); got != want {
			t.Fatalf("slot %d: column=%v predicate=%v", slot, got, want)
		}
	}
	if matched == 0 {
		t.Fatal("no slot matched — the growth assertion proved nothing")
	}
}
