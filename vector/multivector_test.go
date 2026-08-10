// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"math/rand"
	"sort"
	"testing"
)

// randTokens makes n random Dim-dim token vectors from rng.
func randTokens(rng *rand.Rand, n, dim int) [][]float32 {
	toks := make([][]float32, n)
	for i := range toks {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		toks[i] = v
	}
	return toks
}

func normCopy(toks [][]float32) [][]float32 {
	out := make([][]float32, len(toks))
	for i, t := range toks {
		c := make([]float32, len(t))
		copy(c, t)
		normalize(c)
		out[i] = c
	}
	return out
}

// bruteMaxSim is the reference scorer: Σ_q max_d cos(q,d) over normalized vectors.
func bruteMaxSim(query, doc [][]float32) float32 {
	nq, nd := normCopy(query), normCopy(doc)
	var score float32
	for _, q := range nq {
		var best float32
		for di, d := range nd {
			s := dotProduct(q, d)
			if di == 0 || s > best {
				best = s
			}
		}
		score += best
	}
	return score
}

func TestMultiVectorMatchesBruteForce(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(7))
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// 40 documents, 3–7 token vectors each.
	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 40; id++ {
		toks := randTokens(rng, 3+rng.Intn(5), dim)
		docs[id] = toks
		if err := m.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	if m.NumDocs() != 40 {
		t.Fatalf("NumDocs = %d, want 40", m.NumDocs())
	}

	query := randTokens(rng, 5, dim)

	// Reference ranking: exact MaxSim over every document.
	type scored struct {
		id    uint64
		score float32
	}
	ref := make([]scored, 0, len(docs))
	for id, toks := range docs {
		ref = append(ref, scored{id, bruteMaxSim(query, toks)})
	}
	sort.Slice(ref, func(i, j int) bool {
		if ref[i].score != ref[j].score {
			return ref[i].score > ref[j].score
		}
		return ref[i].id < ref[j].id
	})

	// With a generous first stage, the two-stage index must reproduce the exact
	// top-k ranking and scores.
	got, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 5 {
		t.Fatalf("got %d results, want 5", len(got))
	}
	for i, r := range got {
		if r.ID != ref[i].id {
			t.Errorf("rank %d: got id %d (score %.4f), want id %d (score %.4f)",
				i, r.ID, r.Score, ref[i].id, ref[i].score)
		}
		if diff := r.Score - ref[i].score; diff > 1e-4 || diff < -1e-4 {
			t.Errorf("rank %d id %d: score %.5f, want %.5f", i, r.ID, r.Score, ref[i].score)
		}
		if r.Metadata["id"].Int != int64(r.ID) {
			t.Errorf("rank %d: metadata id = %v, want %d", i, r.Metadata["id"], r.ID)
		}
	}
}

func TestMultiVectorUpsertAndDelete(t *testing.T) {
	const dim = 8
	rng := rand.New(rand.NewSource(3))
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, Seed: 1})
	defer m.Close()

	// Doc 1 starts far from the query; doc 2 near it.
	query := [][]float32{{1, 0, 0, 0, 0, 0, 0, 0}}
	far := [][]float32{{0, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0}}
	near := [][]float32{{0.9, 0.1, 0, 0, 0, 0, 0, 0}}
	if err := m.Add(1, far, nil); err != nil {
		t.Fatal(err)
	}
	if err := m.Add(2, near, nil); err != nil {
		t.Fatal(err)
	}
	_ = rng

	res, _ := m.Search(query, 2, MultiSearchOpts{CandidatesPerToken: 50})
	if len(res) != 2 || res[0].ID != 2 {
		t.Fatalf("before upsert: want doc 2 first, got %+v", res)
	}

	// Upsert doc 1 to be the closest now; it must overtake doc 2.
	if err := m.Add(1, [][]float32{{1, 0, 0, 0, 0, 0, 0, 0}}, nil); err != nil {
		t.Fatal(err)
	}
	if m.NumDocs() != 2 {
		t.Fatalf("NumDocs after upsert = %d, want 2", m.NumDocs())
	}
	res, _ = m.Search(query, 2, MultiSearchOpts{CandidatesPerToken: 50})
	if res[0].ID != 1 {
		t.Errorf("after upsert: want doc 1 first, got %+v", res)
	}

	// Delete doc 1; only doc 2 remains.
	if !m.Delete(1) {
		t.Error("Delete(1) = false, want true")
	}
	if m.Delete(99) {
		t.Error("Delete(99) = true, want false")
	}
	res, _ = m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 50})
	if len(res) != 1 || res[0].ID != 2 {
		t.Errorf("after delete: want only doc 2, got %+v", res)
	}
}

func TestMultiVectorValidation(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()

	if err := m.Add(1, nil, nil); !errors.Is(err, ErrEmptyDocument) {
		t.Errorf("empty Add = %v, want ErrEmptyDocument", err)
	}
	if err := m.Add(1, [][]float32{{1, 2, 3}}, nil); !errors.Is(err, ErrDimMismatch) {
		t.Errorf("wrong-dim Add = %v, want ErrDimMismatch", err)
	}
	_ = m.Add(1, [][]float32{{1, 0, 0, 0}}, nil)
	if _, err := m.Search([][]float32{{1, 0, 0}}, 3, MultiSearchOpts{}); !errors.Is(err, ErrDimMismatch) {
		t.Errorf("wrong-dim Search = %v, want ErrDimMismatch", err)
	}
	if r, err := m.Search([][]float32{{1, 0, 0, 0}}, 0, MultiSearchOpts{}); err != nil || r != nil {
		t.Errorf("k=0 = (%v,%v), want (nil,nil)", r, err)
	}
	if r, err := m.Search(nil, 3, MultiSearchOpts{}); err != nil || r != nil {
		t.Errorf("empty query = (%v,%v), want (nil,nil)", r, err)
	}
}

// TestMultiVectorGetLive proves Get returns the token matrix + metadata for a live
// document, and that the returned tokens/payload are DEEP COPIES (caller mutation
// does not corrupt the inner arena or docMeta).
func TestMultiVectorGetLive(t *testing.T) {
	const dim = 4
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, Seed: 1})
	defer m.Close()

	toks := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	if err := m.Add(7, toks, Metadata{"a": NewInt(1), "tag": NewString("x")}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	gotToks, payload, _, ok := m.Get(7)
	if !ok {
		t.Fatal("Get(7) ok=false, want true")
	}
	if len(gotToks) != 3 {
		t.Fatalf("Get(7) returned %d tokens, want 3", len(gotToks))
	}
	for _, tk := range gotToks {
		if len(tk) != dim {
			t.Fatalf("token row dim = %d, want %d", len(tk), dim)
		}
	}
	if payload["a"].Int != 1 || payload["tag"].Str != "x" {
		t.Fatalf("Get(7) payload = %v, want {a:1, tag:x}", payload)
	}

	// Deep-copy isolation: mutating returned tokens/payload must not corrupt state.
	gotToks[0][0] = -99
	payload["a"] = NewInt(999)
	_, payload2, _, _ := m.Get(7)
	if payload2["a"].Int != 1 {
		t.Fatalf("docMeta corrupted by caller mutation: a=%d, want 1", payload2["a"].Int)
	}
	gotToks2, _, _, _ := m.Get(7)
	if gotToks2[0][0] == -99 {
		t.Fatal("inner arena token corrupted by caller mutation")
	}
}

// TestMultiVectorGetAbsent covers the not-found case: an absent docID yields
// ok=false (the MV index has no tombstones — absent = not live).
func TestMultiVectorGetAbsent(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if _, _, _, ok := m.Get(1); ok {
		t.Fatal("Get(absent) ok=true, want false")
	}
	// A deleted doc is also absent.
	_ = m.Add(1, [][]float32{{1, 0, 0, 0}}, nil)
	m.Delete(1)
	if _, _, _, ok := m.Get(1); ok {
		t.Fatal("Get(deleted) ok=true, want false")
	}
}

// TestMultiVectorPayloadSemantics is the merge-vs-overwrite-vs-delete-keys-vs-clear
// table over docMeta, asserting both Get AND a subsequent MV search's returned
// metadata reflect each mutation.
func TestMultiVectorPayloadSemantics(t *testing.T) {
	const dim = 4
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, Seed: 1})
	defer m.Close()
	query := [][]float32{{1, 0, 0, 0}}
	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"a": NewInt(1)}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// searchMeta runs a search and returns doc 1's metadata from the result.
	searchMeta := func() Metadata {
		res, _ := m.Search(query, 1, MultiSearchOpts{CandidatesPerToken: 50})
		if len(res) != 1 || res[0].ID != 1 {
			t.Fatalf("search did not return doc 1: %+v", res)
		}
		return res[0].Metadata
	}

	// SET (merge): add b, keep a.
	if err := m.SetPayload(1, Metadata{"b": NewInt(2)}, nil); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	_, p, _, _ := m.Get(1)
	if p["a"].Int != 1 || p["b"].Int != 2 {
		t.Fatalf("after merge Get = %v, want {a:1,b:2}", p)
	}
	if sm := searchMeta(); sm["a"].Int != 1 || sm["b"].Int != 2 {
		t.Fatalf("after merge search meta = %v, want {a:1,b:2}", sm)
	}

	// OVERWRITE: replace with {c:3}.
	if err := m.OverwritePayload(1, Metadata{"c": NewInt(3)}, nil); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	}
	_, p, _, _ = m.Get(1)
	if len(p) != 1 || p["c"].Int != 3 {
		t.Fatalf("after overwrite Get = %v, want {c:3}", p)
	}
	if sm := searchMeta(); len(sm) != 1 || sm["c"].Int != 3 {
		t.Fatalf("after overwrite search meta = %v, want {c:3}", sm)
	}

	// SET to get a deletable key set: {c:3, d:4}.
	if err := m.SetPayload(1, Metadata{"d": NewInt(4)}, nil); err != nil {
		t.Fatalf("SetPayload(d): %v", err)
	}
	// DELETE-KEYS: remove c (+ an absent key — no-op).
	if err := m.DeletePayloadKeys(1, []string{"c", "absent"}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	}
	_, p, _, _ = m.Get(1)
	if len(p) != 1 || p["d"].Int != 4 {
		t.Fatalf("after delete-keys Get = %v, want {d:4}", p)
	}

	// CLEAR: payload → empty.
	if err := m.ClearPayload(1); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	}
	_, p, _, _ = m.Get(1)
	if len(p) != 0 {
		t.Fatalf("after clear Get = %v, want empty", p)
	}
	if sm := searchMeta(); len(sm) != 0 {
		t.Fatalf("after clear search meta = %v, want empty", sm)
	}

	// The token vectors are untouched: doc 1 still searchable.
	if m.NumDocs() != 1 {
		t.Fatalf("NumDocs = %d, want 1 (payload ops must not drop the doc)", m.NumDocs())
	}
}

// TestMultiVectorPayloadDeadPoint asserts every payload op on an absent document
// returns ErrIDNotFound.
func TestMultiVectorPayloadDeadPoint(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if err := m.SetPayload(1, Metadata{"a": NewInt(1)}, nil); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("SetPayload(absent) = %v, want ErrIDNotFound", err)
	}
	if err := m.OverwritePayload(1, Metadata{"a": NewInt(1)}, nil); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("OverwritePayload(absent) = %v, want ErrIDNotFound", err)
	}
	if err := m.DeletePayloadKeys(1, []string{"a"}); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("DeletePayloadKeys(absent) = %v, want ErrIDNotFound", err)
	}
	if err := m.ClearPayload(1); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("ClearPayload(absent) = %v, want ErrIDNotFound", err)
	}
}

// mvScrollAddDocs inserts docs with ids and an optional "tag" payload (tag=="" ⇒
// no tag) so scroll tests can exercise the filter predicate.
func mvScrollAddDocs(t *testing.T, m *MultiVectorIndex, ids []uint64, tag map[uint64]string) {
	t.Helper()
	for _, id := range ids {
		var meta Metadata
		if tag != nil {
			if v, ok := tag[id]; ok && v != "" {
				meta = Metadata{"tag": NewString(v)}
			}
		}
		if err := m.Add(id, [][]float32{{1, 0, 0, 0}}, meta); err != nil {
			t.Fatalf("Add(%d): %v", id, err)
		}
	}
}

func mvDocIDs(docs []Document) []uint64 {
	out := make([]uint64, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

// TestMVScrollPageAscendingFull pages the whole collection in limit-sized pages
// following nextAfter and asserts the union is exactly the seeded id set, ASCENDING,
// gap-free and dup-free, with hasMore/nextAfter correct at each step.
func TestMVScrollPageAscendingFull(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	// Insert out of order; scroll must still walk ascending.
	ids := []uint64{5, 1, 9, 3, 7, 2, 8, 4, 6}
	mvScrollAddDocs(t, m, ids, nil)

	const limit = 4
	var got []uint64
	var afterID uint64
	hasAfter := false
	pages := 0
	for {
		docs, nextAfter, hasMore, err := m.ScrollDocsPage(Filter{}, afterID, hasAfter, limit)
		if err != nil {
			t.Fatalf("ScrollDocsPage: %v", err)
		}
		pages++
		// Within a page, ids ascending.
		for i := 1; i < len(docs); i++ {
			if docs[i].ID <= docs[i-1].ID {
				t.Fatalf("page not ascending: %v", mvDocIDs(docs))
			}
		}
		got = append(got, mvDocIDs(docs)...)
		if !hasMore {
			if len(docs) > 0 && nextAfter != docs[len(docs)-1].ID {
				t.Fatalf("final nextAfter=%d, want last id %d", nextAfter, docs[len(docs)-1].ID)
			}
			break
		}
		if len(docs) != limit {
			t.Fatalf("hasMore page len=%d, want limit %d", len(docs), limit)
		}
		if nextAfter != docs[len(docs)-1].ID {
			t.Fatalf("nextAfter=%d, want last id %d", nextAfter, docs[len(docs)-1].ID)
		}
		afterID, hasAfter = nextAfter, true
	}
	want := append([]uint64(nil), ids...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	if len(got) != len(want) {
		t.Fatalf("paged %d ids %v, want %d %v", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("paged ids %v, want %v (gap/dup/order)", got, want)
		}
	}
	// 9 ids / limit 4 ⇒ pages of 4,4,1.
	if pages != 3 {
		t.Fatalf("pages=%d, want 3", pages)
	}
}

// TestMVScrollPageResumeAndTruncate asserts afterID seek resumes strictly after
// the cursor and limit truncation reports the right hasMore/nextAfter.
func TestMVScrollPageResumeAndTruncate(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	mvScrollAddDocs(t, m, []uint64{1, 2, 3, 4, 5}, nil)

	// First page, limit 2 ⇒ {1,2}, hasMore, nextAfter=2.
	docs, nextAfter, hasMore, err := m.ScrollDocsPage(Filter{}, 0, false, 2)
	if err != nil {
		t.Fatalf("page1: %v", err)
	}
	if got := mvDocIDs(docs); len(got) != 2 || got[0] != 1 || got[1] != 2 || !hasMore || nextAfter != 2 {
		t.Fatalf("page1 = %v hasMore=%v nextAfter=%d, want {1,2} true 2", got, hasMore, nextAfter)
	}
	// Resume after 2 ⇒ {3,4}, hasMore, nextAfter=4.
	docs, nextAfter, hasMore, err = m.ScrollDocsPage(Filter{}, 2, true, 2)
	if err != nil {
		t.Fatalf("page2: %v", err)
	}
	if got := mvDocIDs(docs); len(got) != 2 || got[0] != 3 || got[1] != 4 || !hasMore || nextAfter != 4 {
		t.Fatalf("page2 = %v hasMore=%v nextAfter=%d, want {3,4} true 4", got, hasMore, nextAfter)
	}
	// Resume after 4 ⇒ {5}, no more.
	docs, _, hasMore, err = m.ScrollDocsPage(Filter{}, 4, true, 2)
	if err != nil {
		t.Fatalf("page3: %v", err)
	}
	if got := mvDocIDs(docs); len(got) != 1 || got[0] != 5 || hasMore {
		t.Fatalf("page3 = %v hasMore=%v, want {5} false", got, hasMore)
	}
}

// TestMVScrollPageFilter asserts the filter predicate is applied (only matching
// docs returned) and the cursor still pages correctly through the filtered set.
func TestMVScrollPageFilter(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	tag := map[uint64]string{1: "a", 2: "b", 3: "a", 4: "b", 5: "a"}
	mvScrollAddDocs(t, m, []uint64{1, 2, 3, 4, 5}, tag)

	f := Filter{Op: FilterEq, Field: "tag", Value: NewString("a")}
	docs, _, hasMore, err := m.ScrollDocsPage(f, 0, false, 10)
	if err != nil {
		t.Fatalf("ScrollDocsPage: %v", err)
	}
	if got := mvDocIDs(docs); len(got) != 3 || got[0] != 1 || got[1] != 3 || got[2] != 5 || hasMore {
		t.Fatalf("filtered = %v hasMore=%v, want {1,3,5} false", got, hasMore)
	}
}

// TestMVScrollPageDeletedFresh proves the per-call sort always reads the current
// docTokens: a doc deleted between pages is absent from the next page.
func TestMVScrollPageDeletedFresh(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	mvScrollAddDocs(t, m, []uint64{1, 2, 3, 4}, nil)

	// Page 1 (limit 2) ⇒ {1,2}; then delete 3 before page 2.
	docs, nextAfter, _, err := m.ScrollDocsPage(Filter{}, 0, false, 2)
	if err != nil {
		t.Fatalf("page1: %v", err)
	}
	if got := mvDocIDs(docs); len(got) != 2 || got[1] != 2 {
		t.Fatalf("page1 = %v, want {1,2}", got)
	}
	if !m.Delete(3) {
		t.Fatal("Delete(3) = false, want true")
	}
	docs, _, hasMore, err := m.ScrollDocsPage(Filter{}, nextAfter, true, 2)
	if err != nil {
		t.Fatalf("page2: %v", err)
	}
	if got := mvDocIDs(docs); len(got) != 1 || got[0] != 4 || hasMore {
		t.Fatalf("page2 = %v hasMore=%v, want {4} false (3 deleted)", got, hasMore)
	}
}

// TestMVScrollPageEmpty asserts an empty collection yields an empty page with
// hasMore=false and nextAfter=0.
func TestMVScrollPageEmpty(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	docs, nextAfter, hasMore, err := m.ScrollDocsPage(Filter{}, 0, false, 10)
	if err != nil {
		t.Fatalf("ScrollDocsPage: %v", err)
	}
	if len(docs) != 0 || hasMore || nextAfter != 0 {
		t.Fatalf("empty = (%v, %d, %v), want (nil, 0, false)", docs, nextAfter, hasMore)
	}
}

// TestMVScrollDocsPageBadFilter asserts a malformed filter fails loud (Compile
// error propagated, no silent drop).
func TestMVScrollDocsPageBadFilter(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	mvScrollAddDocs(t, m, []uint64{1}, nil)
	// An unknown op is a Compile error.
	bad := Filter{Op: FilterOp(0xFF), Field: "tag", Value: NewString("a")}
	if _, _, _, err := m.ScrollDocsPage(bad, 0, false, 10); err == nil {
		t.Fatal("ScrollDocsPage(bad filter) err=nil, want fail-loud error")
	}
}
