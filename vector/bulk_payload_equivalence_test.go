// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

// PAYLOAD-BEARING BULK LOAD MUST BE INDISTINGUISHABLE FROM AN INLINE LOAD.
//
// Carrying payloads used to force the inline route — one indexed insert per
// point — because the staging wire had room for ids and vectors and nothing
// else. Measured on 1M x 768d that cost ~6x the time to searchable, and it made
// the filtered benchmark the one case that could not use the multi-core build.
// StageBulkPayloads + BuildConcurrentMeta close that gap by applying payloads in
// the build's single-threaded placement pass.
//
// The optimization is only worth anything if the two paths agree, so this file
// asserts that they do, on everything a caller can observe: the live set, the
// stored vector, the stored payload, and — the part that actually matters — the
// answer to a filtered search, across every filter family the payload index has
// a different code path for.
//
// The graph itself is deliberately NOT compared. Parallel edge selection is
// non-deterministic by construction (see BuildConcurrentMeta), so an edge-level
// comparison would be asserting something false. What must be identical is the
// LOGICAL state, which is what a filtered search reads.

// bpPayload is the payload for one corpus id. It is built to touch a different
// posting structure in payload_index.go per field:
//
//	id      int    -> the scalar postings + the numeric column sidecar (eq, range, in)
//	bucket  string -> the string scalar postings (eq, in)
//	score   float  -> the float scalar postings (range)
//	flag    bool   -> the two-valued postings, where the complement gate lives
//	tags    []str  -> the CONTAINS postings (a separate map from the scalars)
//	text    string -> the MATCH token postings (a third separate map)
//
// Every fifth id gets NO payload at all. That is not padding: it is the mixed
// batch, and it is what proves the staging buffer's lazy payload column keeps
// point i's payload on point i rather than sliding it onto its neighbour.
func bpPayload(id uint64) Metadata {
	if id%5 == 0 {
		return nil
	}
	// The moduli are coprime with the corpus's id stride (7, see invSeedIDs) on
	// purpose: a modulus sharing a factor with the stride collapses the whole
	// corpus into ONE bucket, and every string/token filter below then matches
	// nothing and asserts nothing.
	return Metadata{
		"id":     NewInt(int64(id)), //nolint:gosec // test fixture ids are small
		"bucket": NewString(fmt.Sprintf("b%d", id%11)),
		"score":  NewFloat(float64(id%100) / 4),
		"flag":   NewBool(id%2 == 0),
		"tags":   NewStrings([]string{fmt.Sprintf("t%d", id%3), "all"}),
		"text":   NewString(fmt.Sprintf("alpha bucket%d omega", id%11)),
	}
}

// bpFilters is the battery. Each entry is asserted three ways: the two load
// paths must agree with each other, and both must agree with the filter
// evaluated directly over the corpus (the brute-force truth). The last two
// entries are the degenerate ends — a filter nothing matches and a filter
// everything matches — which are where an index that quietly returns its
// candidate set instead of the matching set gives itself away.
func bpFilters() []struct {
	name string
	f    Filter
} {
	return []struct {
		name string
		f    Filter
	}{
		{"eq-int", Filter{Op: FilterEq, Field: "id", Value: NewInt(7 * 13)}},
		{"eq-string", Filter{Op: FilterEq, Field: "bucket", Value: NewString("b3")}},
		{"eq-bool", Filter{Op: FilterEq, Field: "flag", Value: NewBool(true)}},
		{"ne-string", Filter{Op: FilterNe, Field: "bucket", Value: NewString("b3")}},
		{"range-gte-int", Filter{Op: FilterGte, Field: "id", Value: NewInt(700)}},
		{"range-lt-float", Filter{Op: FilterLt, Field: "score", Value: NewFloat(6)}},
		{"range-band", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "id", Value: NewInt(210)},
			{Op: FilterLt, Field: "id", Value: NewInt(980)},
		}}},
		{"in-int", Filter{Op: FilterIn, Field: "id", Value: NewInts([]int64{7, 70, 700, 1400, 999999})}},
		{"in-string", Filter{Op: FilterIn, Field: "bucket", Value: NewStrings([]string{"b1", "b5"})}},
		{"contains", Filter{Op: FilterContains, Field: "tags", Value: NewString("t1")}},
		{"contains-universal", Filter{Op: FilterContains, Field: "tags", Value: NewString("all")}},
		{"match", Filter{Op: FilterMatch, Field: "text", Value: NewString("bucket4")}},
		{"match-universal", Filter{Op: FilterMatch, Field: "text", Value: NewString("alpha")}},
		{"is-empty", Filter{Op: FilterIsEmpty, Field: "bucket"}},
		{"or-mixed", Filter{Op: FilterOr, Or: []Filter{
			{Op: FilterEq, Field: "bucket", Value: NewString("b2")},
			{Op: FilterContains, Field: "tags", Value: NewString("t0")},
		}}},
		{"not", Filter{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "flag", Value: NewBool(true)}}},
		// Matches nothing: the field exists, the value does not.
		{"matches-nothing", Filter{Op: FilterEq, Field: "bucket", Value: NewString("no-such-bucket")}},
		// Matches nothing via a field NO point carries — the empty-set inference.
		{"unknown-field", Filter{Op: FilterEq, Field: "nope", Value: NewInt(1)}},
		// Matches every point that has a payload at all.
		{"matches-everything", Filter{Op: FilterGte, Field: "id", Value: NewInt(0)}},
	}
}

// bpCorpus builds the shared corpus: non-contiguous shuffled ids (so id N is not
// on slot N on either path) with distinct unit-norm vectors.
func bpCorpus(t *testing.T, n int, dim int) *invCorpus {
	t.Helper()
	return newInvCorpus(t, invSeedIDs(n, 77), dim, 4)
}

func bpConfig(dim int) Config {
	// EfSearch stays at the ordinary default: the assertions pass k = n instead,
	// and the search widens ef from k (see the note on the filtered comparison),
	// so the comparison is about the PAYLOAD state and not about beam width —
	// while the collection is still configured the way a real one would be.
	// M/EfConstruction are the invariant suite's baseline.
	return Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
}

// bpLoadInline is the reference path: one indexed insert per point, payload and
// all — exactly what /points/batch does.
func bpLoadInline(t *testing.T, name string, cfg Config, c *invCorpus) *Collection {
	t.Helper()
	coll, err := NewCollection(name, cfg)
	if err != nil {
		t.Fatalf("NewCollection(%s): %v", name, err)
	}
	for _, id := range c.ids {
		if err := coll.Insert(id, c.vec[id], 0, bpPayload(id), nil); err != nil {
			t.Fatalf("inline insert id %d: %v", id, err)
		}
	}
	return coll
}

// bpLoadStaged is the path under test: several staging batches — deliberately
// alternating payload-bearing and vectors-only, so the staging buffer's lazy
// payload column has to be materialized mid-load and back-filled — then one
// concurrent build.
func bpLoadStaged(t *testing.T, name string, cfg Config, c *invCorpus, workers int) *Collection {
	t.Helper()
	coll, err := NewCollection(name, cfg)
	if err != nil {
		t.Fatalf("NewCollection(%s): %v", name, err)
	}
	const batch = 37
	for start := 0; start < len(c.ids); start += batch {
		end := start + batch
		if end > len(c.ids) {
			end = len(c.ids)
		}
		ids := c.ids[start:end]
		vecs := make([][]float32, len(ids))
		metas := make([]Metadata, len(ids))
		anyMeta := false
		for i, id := range ids {
			vecs[i] = c.vec[id]
			metas[i] = bpPayload(id)
			if metas[i] != nil {
				anyMeta = true
			}
		}
		// A batch whose points all happen to be payload-less is staged through the
		// VECTORS-ONLY entry point, which is what a real mixed load does and what
		// makes the column's back-fill path reachable.
		if !anyMeta {
			if err := coll.StageBulk(ids, vecs); err != nil {
				t.Fatalf("StageBulk [%d,%d): %v", start, end, err)
			}
			continue
		}
		if err := coll.StageBulkPayloads(ids, vecs, metas); err != nil {
			t.Fatalf("StageBulkPayloads [%d,%d): %v", start, end, err)
		}
	}
	if err := coll.BuildStaged(workers); err != nil {
		t.Fatalf("BuildStaged: %v", err)
	}
	return coll
}

// bpExpected evaluates a filter directly over the corpus — the brute-force truth
// both paths are held to, so that "the two agree" cannot be satisfied by two
// identically wrong answers.
func bpExpected(t *testing.T, c *invCorpus, f Filter) []uint64 {
	t.Helper()
	pred, err := CompileFilter(f)
	if err != nil {
		t.Fatalf("compile filter: %v", err)
	}
	out := []uint64{}
	for _, id := range c.ids {
		if pred == nil || pred(bpPayload(id)) {
			out = append(out, id)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// bpFilteredIDs returns the filtered result as a SORTED, never-nil id set, so
// that "no matches" compares equal however each layer chose to spell an empty
// slice.
func bpFilteredIDs(t *testing.T, coll *Collection, q []float32, k int, f Filter) []uint64 {
	t.Helper()
	res, err := coll.SearchFiltered(q, k, f)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	out := append([]uint64{}, resultIDs(res)...)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// TestBulkPayloadLoadEqualsInlineLoad is the equivalence gate.
func TestBulkPayloadLoadEqualsInlineLoad(t *testing.T) {
	const n, dim = 420, 16
	cfg := bpConfig(dim)
	c := bpCorpus(t, n, dim)

	inline := bpLoadInline(t, "bp_inline", cfg, c)
	defer func() { _ = inline.Close() }()
	staged := bpLoadStaged(t, "bp_staged", cfg, c, 4)
	defer func() { _ = staged.Close() }()

	if a, b := inline.Stats().Size, staged.Stats().Size; a != b {
		t.Fatalf("live count: inline %d, staged %d", a, b)
	}

	// 1. Every point's stored vector AND stored payload must round-trip
	//    identically. Get is the surface a caller reads a payload back through,
	//    so an equal Get is what "same payloads retrievable" means.
	for _, id := range c.ids {
		iv, im, _, _, _, iok := inline.Get(id)
		sv, sm, _, _, _, sok := staged.Get(id)
		if !iok || !sok {
			t.Fatalf("id %d: Get present inline=%v staged=%v", id, iok, sok)
		}
		if !reflect.DeepEqual(iv, sv) {
			t.Fatalf("id %d: stored vector differs between the two load paths", id)
		}
		if !reflect.DeepEqual(im, sm) {
			t.Fatalf("id %d: stored payload differs\n inline: %#v\n staged: %#v", id, im, sm)
		}
		if want := bpPayload(id); !reflect.DeepEqual(sm, want) {
			t.Fatalf("id %d: staged payload is not what was staged\n  got: %#v\n want: %#v", id, sm, want)
		}
	}

	// 2. Filtered search must return the same SET on both paths, and that set must
	//    be the brute-force truth — so the comparison is about which points the
	//    payload index admits, not about ranking.
	//
	//    k = n, and the BEAM FOLLOWS k RATHER THAN THE CONFIG. bpConfig leaves
	//    EfSearch at 64; the bottom-level expansion raises ef to max(EfSearch, k)
	//    and, when a predicate is present, to 2k — so a filtered query here runs at
	//    ef = 840 against 420 points, under the 1024 MaxEfSearch default. That is
	//    what makes exact set equality a fair assertion at this corpus size, and it
	//    is worth stating because reading "EfSearch: 64" and assuming a beam of 64
	//    would make the assertion below look like an over-claim.
	// COVERAGE FLOOR. Every assertion below is satisfied trivially when a filter
	// matches nothing on both paths, and a filter can silently degenerate to that
	// from one edit to bpPayload (it did: a modulus sharing a factor with the id
	// stride collapsed six of them at once). Require that most of the battery is
	// genuinely DISCRIMINATING — neither empty nor the whole corpus — so the suite
	// fails as a test rather than passing as a tautology.
	const minDiscriminating = 12
	discriminating := 0
	for _, tc := range bpFilters() {
		if n := len(bpExpected(t, c, tc.f)); n > 0 && n < len(c.ids) {
			discriminating++
		}
	}
	if discriminating < minDiscriminating {
		t.Fatalf("only %d of %d filters select a proper non-empty subset of the corpus "+
			"(want >= %d) — the battery has degenerated and most cases assert nothing",
			discriminating, len(bpFilters()), minDiscriminating)
	}

	q := invUnitVec(dim, 12345, 99)
	for _, tc := range bpFilters() {

		t.Run("filter/"+tc.name, func(t *testing.T) {
			want := bpExpected(t, c, tc.f)
			gotInline := bpFilteredIDs(t, inline, q, n, tc.f)
			gotStaged := bpFilteredIDs(t, staged, q, n, tc.f)
			if !reflect.DeepEqual(gotInline, gotStaged) {
				t.Fatalf("filter %s: the two load paths disagree\n inline (%d): %v\n staged (%d): %v",
					tc.name, len(gotInline), gotInline, len(gotStaged), gotStaged)
			}
			if !reflect.DeepEqual(gotStaged, want) {
				t.Fatalf("filter %s: staged load returns %d ids, brute force says %d\n  got: %v\n want: %v",
					tc.name, len(gotStaged), len(want), gotStaged, want)
			}
			if tc.name == "matches-nothing" || tc.name == "unknown-field" {
				if len(gotStaged) != 0 {
					t.Fatalf("filter %s must match nothing, got %v", tc.name, gotStaged)
				}
			}
		})
	}
}

// TestBulkPayloadSearchabilityUnderFilter is the searchability invariant
// restated for this path: every point loaded with a payload must be findable by
// its OWN vector under a filter that selects exactly it. Unfiltered
// searchability is covered by the standing matrix (the bulk-build-payloads
// stage); this is the half that only exists once payloads are in the index, and
// it is the one a payload applied to the WRONG SLOT would fail — such a load
// still returns every point unfiltered, and still returns the right number of
// points per filter, while pairing them with the wrong vectors.
func TestBulkPayloadSearchabilityUnderFilter(t *testing.T) {
	const n, dim = 300, 16
	cfg := bpConfig(dim)
	cfg.EfSearch = 128
	c := bpCorpus(t, n, dim)
	coll := bpLoadStaged(t, "bp_selffilter", cfg, c, 4)
	defer func() { _ = coll.Close() }()

	var missing int
	for _, id := range c.ids {
		if bpPayload(id) == nil {
			continue // no payload: nothing to select it by
		}
		f := Filter{Op: FilterEq, Field: "id", Value: NewInt(int64(id))} //nolint:gosec // small test ids
		res, err := coll.SearchFiltered(c.vec[id], invSearchK, f)
		if err != nil {
			t.Fatalf("id %d: filtered self-query: %v", id, err)
		}
		got := resultIDs(res)
		if invRankOf(got, id) != 0 {
			missing++
			if missing <= invMaxReport {
				t.Errorf("id %d: a filter selecting exactly its own payload, queried with its own "+
					"vector, did not return it at rank 0 — got %v", id, got)
			}
		}
	}
	if missing > invMaxReport {
		t.Errorf("%d points fail the filtered self-query; only the first %d printed", missing, invMaxReport)
	}
}

// TestBulkPayloadStagingColumnStaysAligned pins the staging buffer's LAZY
// payload column at the two transitions the equivalence test above does not
// reach, because every one of its batches happens to carry at least one payload.
//
// The column does not exist until a payload-bearing batch creates it, so a load
// that mixes batch KINDS has two ways to slide payloads onto the wrong points:
//
//	vectors-only THEN payloads — the column is born after N points are already
//	staged, so it must be back-filled with N nils or every payload lands N
//	points early;
//	payloads THEN vectors-only THEN payloads — the middle batch owns rows in a
//	column it contributes nothing to, so it must still occupy them.
//
// Both were verified to be real: removing either fill makes this test fail (and
// leaves the equivalence test above green, which is why this one exists
// separately). The payload is the point's own id, so a misalignment of even one
// row is caught exactly, by name.
func TestBulkPayloadStagingColumnStaysAligned(t *testing.T) {
	const dim = 8
	idPayload := func(id uint64) Metadata {
		return Metadata{"self": NewInt(int64(id))} //nolint:gosec // small test ids
	}
	// Batch kinds, in the orders that exercise both fills. true = payload-bearing.
	orders := [][]bool{
		{false, true},              // back-fill: the column is born mid-load
		{true, false, true},        // nil-fill: a vectors-only batch between two payload ones
		{false, true, false, true}, // both, twice
		{true, true},               // the ordinary case, as a control
	}
	for oi, kinds := range orders {

		t.Run(fmt.Sprintf("order%d", oi), func(t *testing.T) {
			cfg := bpConfig(dim)
			coll, err := NewCollection(fmt.Sprintf("bp_align_%d", oi), cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			defer func() { _ = coll.Close() }()

			const perBatch = 11
			var withPayload, withoutPayload []uint64
			next := uint64(3) // start off zero and stride oddly, so id != slot
			for _, hasPayload := range kinds {
				ids := make([]uint64, perBatch)
				vecs := make([][]float32, perBatch)
				metas := make([]Metadata, perBatch)
				for i := range ids {
					ids[i] = next
					next += 3
					vecs[i] = invUnitVec(dim, ids[i], 55)
					metas[i] = idPayload(ids[i])
				}
				if hasPayload {
					withPayload = append(withPayload, ids...)
					if err := coll.StageBulkPayloads(ids, vecs, metas); err != nil {
						t.Fatalf("StageBulkPayloads: %v", err)
					}
					continue
				}
				withoutPayload = append(withoutPayload, ids...)
				if err := coll.StageBulk(ids, vecs); err != nil {
					t.Fatalf("StageBulk: %v", err)
				}
			}
			if err := coll.BuildStaged(4); err != nil {
				t.Fatalf("BuildStaged: %v", err)
			}

			for _, id := range withPayload {
				_, meta, _, _, _, ok := coll.Get(id)
				if !ok {
					t.Fatalf("id %d absent after the build", id)
				}
				if !reflect.DeepEqual(meta, idPayload(id)) {
					t.Fatalf("id %d carries payload %#v — the staging column slid", id, meta)
				}
			}
			for _, id := range withoutPayload {
				_, meta, _, _, _, ok := coll.Get(id)
				if !ok {
					t.Fatalf("id %d absent after the build", id)
				}
				if len(meta) != 0 {
					t.Fatalf("id %d was staged WITHOUT a payload but carries %#v — a neighbour's "+
						"payload landed on it", id, meta)
				}
			}
		})
	}
}

// TestBulkPayloadBuildRejectsMisalignedPayloads pins the one wire-level mistake
// that would silently pair payloads with the wrong points: a payload column of
// the wrong length. It must be an error at every layer that can see it, never a
// truncation or a zero-fill.
func TestBulkPayloadBuildRejectsMisalignedPayloads(t *testing.T) {
	cfg := bpConfig(4)
	coll, err := NewCollection("bp_misaligned", cfg)
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	defer func() { _ = coll.Close() }()

	ids := []uint64{1, 2, 3}
	vecs := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	short := []Metadata{{"a": NewInt(1)}}

	if err := coll.StageBulkPayloads(ids, vecs, short); err == nil {
		t.Fatal("StageBulkPayloads accepted 3 ids with 1 payload")
	}
	if err := coll.BuildConcurrentMeta(ids, vecs, short, 2); err == nil {
		t.Fatal("BuildConcurrentMeta accepted 3 ids with 1 payload")
	}
	// Nothing may have been staged or built by either rejection.
	if coll.Stats().Size != 0 {
		t.Fatalf("a rejected payload-bearing load left %d points behind", coll.Stats().Size)
	}
}

// TestBulkPayloadEmptyColumnIsVectorsOnly pins the boundary between "no
// payloads" and "a payload column of the wrong length".
//
// An EMPTY but non-nil column is the legal spelling of the former, and it is
// exactly the case a nilness guard gets wrong: `metas != nil` passes the length
// precondition (zero is the exempt case, since nil and empty must both mean "no
// payloads") and then indexes metas[i] of a zero-length slice — a panic, on
// every family, for an input the contract says is fine. The guard is therefore on
// LENGTH everywhere, and this asserts it on all three bulk builders.
func TestBulkPayloadEmptyColumnIsVectorsOnly(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tweak func(*Config)
	}{
		{"hnsw", nil},
		{"vamana", func(c *Config) { c.IndexType = IndexVamana }},
		{"ivf", func(c *Config) {
			c.IndexType = IndexIVF
			c.IVFNlist = 2
			c.IVFNprobe = 2
			c.IVFTrainThreshold = 1 << 30
		}},
	} {

		t.Run(tc.name, func(t *testing.T) {
			cfg := bpConfig(4)
			if tc.tweak != nil {
				tc.tweak(&cfg)
			}
			coll, err := NewCollection("bp_emptycol_"+tc.name, cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			defer func() { _ = coll.Close() }()

			ids := []uint64{1, 2, 3}
			vecs := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
			if err := coll.BuildConcurrentMeta(ids, vecs, []Metadata{}, 2); err != nil {
				t.Fatalf("build with an empty payload column: %v", err)
			}
			if got := coll.Stats().Size; got != len(ids) {
				t.Fatalf("loaded %d of %d points", got, len(ids))
			}
			for _, id := range ids {
				if _, meta, _, _, _, ok := coll.Get(id); !ok || len(meta) != 0 {
					t.Fatalf("id %d: present=%v payload=%#v, want present with no payload", id, ok, meta)
				}
			}
		})
	}
}

// TestBulkPayloadContentEntersBM25Corpus covers the one part of a payload that
// is NOT the payload index: the reserved $content field, which the inline path
// also feeds to BM25.
//
// It is reachable from the bulk route because content rides the metadata map —
// a caller that spells the reserved key itself gets it stored — and it is the
// piece an "index the payloads after the build" design would have had to
// re-derive. Applying it in the placement pass means it is simply there, and
// this test is what stops it from being quietly dropped: without the bm25 branch
// in applyBulkMeta the bulk collection returns nothing for a term the inline one
// returns every document for.
func TestBulkPayloadContentEntersBM25Corpus(t *testing.T) {
	const n, dim = 96, 8
	cfg := bpConfig(dim)
	cfg.FullText = &FullTextConfig{}
	c := newInvCorpus(t, invSeedIDs(n, 91), dim, 6)

	text := func(id uint64) string {
		return fmt.Sprintf("shared token unique%d", id)
	}
	payload := func(id uint64) Metadata {
		return Metadata{contentField: NewString(text(id))}
	}

	load := func(name string, staged bool) *Collection {
		coll, err := NewCollection(name, cfg)
		if err != nil {
			t.Fatalf("NewCollection(%s): %v", name, err)
		}
		if !staged {
			for _, id := range c.ids {
				if err := coll.Insert(id, c.vec[id], 0, payload(id), nil); err != nil {
					t.Fatalf("inline insert %d: %v", id, err)
				}
			}
			return coll
		}
		vecs := make([][]float32, len(c.ids))
		metas := make([]Metadata, len(c.ids))
		for i, id := range c.ids {
			vecs[i] = c.vec[id]
			metas[i] = payload(id)
		}
		if err := coll.StageBulkPayloads(c.ids, vecs, metas); err != nil {
			t.Fatalf("StageBulkPayloads: %v", err)
		}
		if err := coll.BuildStaged(4); err != nil {
			t.Fatalf("BuildStaged: %v", err)
		}
		return coll
	}

	inline := load("bp_bm25_inline", false)
	defer func() { _ = inline.Close() }()
	staged := load("bp_bm25_staged", true)
	defer func() { _ = staged.Close() }()

	textIDs := func(coll *Collection, q string, k int) []uint64 {
		docs, err := coll.SearchText(q, k, Filter{})
		if err != nil {
			t.Fatalf("SearchText(%q): %v", q, err)
		}
		out := []uint64{}
		for _, d := range docs {
			out = append(out, d.ID)
		}
		sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
		return out
	}

	// A term every document carries: both corpora must return the full k.
	if got, want := textIDs(staged, "shared", n), textIDs(inline, "shared", n); !reflect.DeepEqual(got, want) {
		t.Fatalf("corpus-wide term: staged returned %d ids, inline %d", len(got), len(want))
	} else if len(got) != n {
		t.Fatalf("corpus-wide term matched %d of %d documents on BOTH paths — the BM25 corpus "+
			"was not populated by either and the comparison proves nothing", len(got), n)
	}
	// A term exactly one document carries, for every document.
	for _, id := range c.ids {
		q := fmt.Sprintf("unique%d", id)
		got, want := textIDs(staged, q, invSearchK), textIDs(inline, q, invSearchK)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("term %q: staged %v, inline %v", q, got, want)
		}
		if len(got) != 1 || got[0] != id {
			t.Fatalf("term %q matched %v, want exactly [%d]", q, got, id)
		}
	}
}

// bpPayloadIndexOf reaches the index family's payload index so the test below can
// put it into the one state no public API can produce: postings present with an
// EMPTY arena.
func bpPayloadIndexOf(t *testing.T, coll *Collection) *payloadIndex {
	t.Helper()
	switch ix := coll.idx.(type) {
	case *hnsw:
		return ix.payloadIdx
	case *ivf:
		return ix.payloadIdx
	default:
		t.Fatalf("index family %T has no payload index the test can reach", coll.idx)
		return nil
	}
}

// TestBulkPayloadBuildRequiresEmptyIndex pins the precondition RESTATEMENT.
//
// BuildConcurrentMeta now writes payload postings itself, which makes the
// "payload index must be empty" clause look like something the change made
// redundant. It is not, and deleting it is the mistake this test exists to
// prevent: the clause is about the index being non-empty ON ENTRY, where a bulk
// build would hand slot 0 to a new point while the index still holds a previous
// occupant's keys for slot 0 — a WRONG ROW, not a slow one. Both branches of
// "non-empty" are asserted, for all three families that run a bulk placement
// loop.
func TestBulkPayloadBuildRequiresEmptyIndex(t *testing.T) {
	families := []struct {
		name  string
		tweak func(*Config)
	}{
		{"hnsw", nil},
		{"vamana", func(c *Config) { c.IndexType = IndexVamana }},
		{"ivf", func(c *Config) {
			c.IndexType = IndexIVF
			c.IVFNlist = 4
			c.IVFNprobe = 4
			c.IVFTrainThreshold = 1 << 30 // never auto-train; the build is the trigger
		}},
	}
	newColl := func(t *testing.T, name string, tweak func(*Config)) *Collection {
		t.Helper()
		cfg := bpConfig(4)
		if tweak != nil {
			tweak(&cfg)
		}
		coll, err := NewCollection(name, cfg)
		if err != nil {
			t.Fatalf("NewCollection: %v", err)
		}
		return coll
	}
	ids := []uint64{2}
	vecs := [][]float32{{0, 1, 0, 0}}
	metas := []Metadata{{"b": NewInt(2)}}

	for _, tc := range families {

		t.Run(tc.name+"/populated-arena", func(t *testing.T) {
			coll := newColl(t, "bp_arena_"+tc.name, tc.tweak)
			defer func() { _ = coll.Close() }()
			if err := coll.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"a": NewInt(1)}, nil); err != nil {
				t.Fatalf("seed insert: %v", err)
			}
			if err := coll.BuildConcurrentMeta(ids, vecs, metas, 2); err != ErrBuildNonEmpty {
				t.Fatalf("build over a populated arena: got %v, want ErrBuildNonEmpty", err)
			}
		})
		t.Run(tc.name+"/postings-without-vectors", func(t *testing.T) {
			coll := newColl(t, "bp_postings_"+tc.name, tc.tweak)
			defer func() { _ = coll.Close() }()
			// No public path leaves postings behind an empty arena — Reclaim and
			// Restore both rebuild the index from the arena — so the state is
			// constructed directly. That is the point: the guard has to hold even for
			// a state only a future caller could introduce, which is exactly the
			// tripwire role it was given.
			bpPayloadIndexOf(t, coll).reindex(0, Metadata{"ghost": NewInt(1)})
			if err := coll.BuildConcurrentMeta(ids, vecs, metas, 2); err != ErrBuildNonEmpty {
				t.Fatalf("build over a populated payload index: got %v, want ErrBuildNonEmpty", err)
			}
		})
	}
}
