// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// seedGroupQueryCollection creates a P-partition (or P==1) DENSE collection (dim 4,
// L2) and inserts ids 1..n, each carrying a deterministic "doc" group-by metadata
// scalar (doc = (id+1)/2 ⇒ two chunks per doc) plus an inline sparse lane. So a
// grouped query collapses the ordered pool by "doc" and a FUSION spec can also score
// the sparse lane. Mirrors seedQueryCollection but adds the group key + the standalone
// groups oracle's two-chunks-per-doc layout.
func seedGroupQueryCollection(t *testing.T, s Store, coll string, P, n int) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		doc := int64((id + 1) / 2)
		v := []float32{float32(id), 0, 0, 0}
		sp := VectorSparse{Indices: []uint32{uint32(id % 7)}, Values: []float32{float32(id)*0.01 + 1}}
		opts := VectorInsertOpts{Metadata: VectorMetadata{"doc": vector.NewInt(doc)}, Sparse: sp}
		if err := s.VectorInsertExt(ctx, coll, id, v, opts); err != nil {
			t.Fatalf("VectorInsertExt %s/%d: %v", coll, id, err)
		}
	}
}

// groupKeyShape is a flattened (group key, ordered hit ids) projection used to assert
// two grouped results are IDENTICAL: same groups, same key, same order, same hit order.
type groupKeyShape struct {
	key  vector.Value
	hits []uint64
}

func groupShapes(groups []VectorGroup) []groupKeyShape {
	out := make([]groupKeyShape, len(groups))
	for i, g := range groups {
		hits := make([]uint64, len(g.Hits))
		for j, h := range g.Hits {
			hits[j] = h.ID
		}
		out[i] = groupKeyShape{key: g.Key, hits: hits}
	}
	return out
}

func assertSameGroups(t *testing.T, label string, a, b []VectorGroup) {
	t.Helper()
	as, bs := groupShapes(a), groupShapes(b)
	if len(as) != len(bs) {
		t.Fatalf("%s: group count %d != %d", label, len(as), len(bs))
	}
	for i := range as {
		ak, _ := json.Marshal(as[i].key)
		bk, _ := json.Marshal(bs[i].key)
		if string(ak) != string(bk) {
			t.Errorf("%s: group %d key %s != %s", label, i, ak, bk)
		}
		if len(as[i].hits) != len(bs[i].hits) {
			t.Fatalf("%s: group %d hit count %d != %d", label, i, len(as[i].hits), len(bs[i].hits))
		}
		for j := range as[i].hits {
			if as[i].hits[j] != bs[i].hits[j] {
				t.Errorf("%s: group %d hit %d = id%d != id%d", label, i, j, as[i].hits[j], bs[i].hits[j])
			}
		}
	}
}

// TestQueryGroupFanOutMatchesP1 is the grouped P>1==P1 exactness invariant: a GROUPED
// query (group_by=doc) over a P=4 collection returns the EXACT same groups (keys,
// group order, hit order within each group) as the same query over a P=1 collection —
// for BOTH FUSION and RERANK. It proves the coordinator fuses/reranks globally then
// groups ONCE over the exact ordered pool (grouping is post-merge). RED if a partition
// groups locally or the id→key map misses a pooled id.
func TestQueryGroupFanOutMatchesP1(t *testing.T) {
	const n, k, groupSize = 200, 5, 2
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	cases := []struct {
		name  string
		build func() *pb.QuerySpec
	}{
		{"fusion-rrf", func() *pb.QuerySpec {
			return &pb.QuerySpec{
				Mode:         pb.QueryMode_QUERY_MODE_FUSION,
				Prefetch:     []*pb.QueryLeaf{denseLeaf(denseQ, 50), sparseLeaf(sIdx, sVal, 50)},
				FusionMethod: "rrf",
				K:            int32(k),
				GroupBy:      "doc",
				GroupSize:    uint32(groupSize),
			}
		}},
		{"rerank", func() *pb.QuerySpec {
			return &pb.QuerySpec{
				Mode:      pb.QueryMode_QUERY_MODE_RERANK,
				Root:      denseLeaf(denseQ, k),
				Prefetch:  []*pb.QueryLeaf{denseLeaf(denseQ, 50), sparseLeaf(sIdx, sVal, 50)},
				K:         int32(k),
				GroupBy:   "doc",
				GroupSize: uint32(groupSize),
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			specBytes, spec := buildQuerySpec(t, tc.build())

			s1 := newSingleEmbedded(t)
			waitLeaderEmbedded(t, s1)
			seedGroupQueryCollection(t, s1, "g1", 1, n)
			got1, _, err := s1.(*embedded).VectorQueryGrouped(ctx, "g1", specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P1 grouped query: %v", err)
			}
			if len(got1) == 0 {
				t.Fatalf("P1 grouped query returned no groups")
			}

			const P = 4
			sP := newSingleEmbedded(t)
			waitLeaderEmbedded(t, sP)
			seedGroupQueryCollection(t, sP, "g4", P, n)
			gotP, meta, err := sP.(*embedded).VectorQueryGrouped(ctx, "g4", specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P4 grouped query: %v", err)
			}
			if meta.Degraded {
				t.Fatalf("P4 grouped query degraded")
			}
			assertSameGroups(t, "P4==P1 "+tc.name, gotP, got1)
		})
	}
}

// TestQueryGroupedThreeWayOracle is the 3-way oracle at P4: a grouped DENSE query
// (group_by) == the single-node grouped query (P1) == the standalone VectorSearchGroups
// over the equivalent dense query + the SAME group_by/group_size. It proves the Query
// API grouping reuses GroupDocuments over the exact ordered pool identically to the
// standalone groups op (whose fan-out is the trusted P>1==P1 baseline).
func TestQueryGroupedThreeWayOracle(t *testing.T) {
	const n, k, groupSize = 200, 5, 2
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}

	// A DENSE-only grouped query (single dense prefetch lane) so the ordered pool is the
	// plain dense KNN pool — exactly what SearchGroups groups over.
	pspec := &pb.QuerySpec{
		Mode:      pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:  []*pb.QueryLeaf{denseLeaf(denseQ, 50)},
		K:         int32(k),
		GroupBy:   "doc",
		GroupSize: uint32(groupSize),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedGroupQueryCollection(t, sP, "g4", P, n)

	groupedP4, _, err := sP.(*embedded).VectorQueryGrouped(ctx, "g4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 grouped query: %v", err)
	}

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedGroupQueryCollection(t, s1, "g1", 1, n)
	groupedP1, _, err := s1.(*embedded).VectorQueryGrouped(ctx, "g1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 grouped query: %v", err)
	}

	// Standalone groups oracle over the SAME dense query + group_by/group_size (P4).
	standalone, _, err := sP.(*embedded).VectorSearchGroups(ctx, "g4", denseQ, k,
		VectorGroupOpts{GroupBy: "doc", GroupSize: groupSize})
	if err != nil {
		t.Fatalf("standalone SearchGroups: %v", err)
	}

	assertSameGroups(t, "groupedP4 == groupedP1", groupedP4, groupedP1)
	assertSameGroups(t, "groupedP4 == standalone SearchGroups", groupedP4, standalone)
}

// TestQueryGroupedNoGroupByFlatUnchanged is the #1 no-break invariant at the fan-out
// boundary: a query WITHOUT group_by routes through the FLAT VectorQuery path and is
// byte/behaviour-identical P4==P1 (the grouped branch never engages). This is the
// companion to the flat fan-out test, asserting the additive group fields default empty.
func TestQueryGroupedNoGroupByFlatUnchanged(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	pspec := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{denseLeaf(denseQ, k), sparseLeaf([]uint32{3}, []float32{1}, k)},
		K:        int32(k),
		// GroupBy intentionally empty.
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if spec.GroupBy != "" {
		t.Fatalf("spec unexpectedly grouped")
	}

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedGroupQueryCollection(t, s1, "f1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "f1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 flat query: %v", err)
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedGroupQueryCollection(t, sP, "f4", P, n)
	gotP, _, err := sP.(*embedded).VectorQuery(ctx, "f4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 flat query: %v", err)
	}
	a, b := queryResultKeys(got1), queryResultKeys(gotP)
	if len(a) != len(b) {
		t.Fatalf("flat P4 len %d != P1 len %d", len(b), len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("flat P4 result %d = %+v != P1 %+v", i, b[i], a[i])
		}
	}
}

// TestNamedGroupedQueryRejectedAtWire asserts a NAMED grouped query is rejected
// fail-loud at the wire (ErrQueryGroupNotDense propagated through the named handler) —
// NOT a panic or a silently-ungrouped result. Grouping is dense-only in v1.
func TestNamedGroupedQueryRejectedAtWire(t *testing.T) {
	ctx := context.Background()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	e := s.(*embedded)

	if err := e.VectorNamedCreateCollection(ctx, "nm", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
	}, 0); err != nil {
		t.Fatalf("VectorNamedCreateCollection: %v", err)
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{Space: "title", Dense: []float32{1, 0, 0, 0}, K: 10}}},
		},
		K:         5,
		GroupBy:   "doc",
		GroupSize: 2,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	_, _, err := e.VectorNamedQuery(ctx, "nm", specBytes, spec, ReadOpts{})
	if err == nil {
		t.Fatalf("named grouped query: want fail-loud error, got nil")
	}
	if !errors.Is(err, vector.ErrQueryGroupNotDense) {
		t.Fatalf("named grouped query error = %v, want ErrQueryGroupNotDense", err)
	}
}

// TestNamedGroupedQueryP4RejectedAtWire is the P>1 companion to
// TestNamedGroupedQueryRejectedAtWire: proves the per-partition guard (ErrQueryGroupNotDense
// in NamedCollection.Query) fires on EVERY partition during the namedQueryFanOut, causing
// all partitions to be marked missing/degraded with zero results returned. The cluster
// fan-out degrades silently on per-partition errors (OnUnavailable==Partial is the default),
// so the observable signal is: Degraded=true, all P partitions missing, len(results)==0.
// RED if any partition silently returns results instead of rejecting the grouped spec.
func TestNamedGroupedQueryP4RejectedAtWire(t *testing.T) {
	ctx := context.Background()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	e := s.(*embedded)

	const P = 4
	if err := e.VectorNamedCreateCollection(ctx, "nm4", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
	}, P); err != nil {
		t.Fatalf("VectorNamedCreateCollection (P=%d): %v", P, err)
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{Space: "title", Dense: []float32{1, 0, 0, 0}, K: 10}}},
		},
		K:         5,
		GroupBy:   "doc",
		GroupSize: 2,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	// The fan-out (OnUnavailable==Partial) degrades all partitions rather than
	// returning a hard error; verify every partition rejected the grouped spec.
	res, meta, err := e.VectorNamedQuery(ctx, "nm4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named grouped query unexpected hard error: %v", err)
	}
	if !meta.Degraded {
		t.Fatalf("P4 named grouped query: want Degraded=true (all partitions rejected), got Degraded=false")
	}
	if len(meta.Missing) != P {
		t.Fatalf("P4 named grouped query: want %d missing partitions, got %d: %v", P, len(meta.Missing), meta.Missing)
	}
	if len(res) != 0 {
		t.Fatalf("P4 named grouped query: want 0 results (all rejected), got %d", len(res))
	}
}

// seedFloatGroupCollection creates a P-partition DENSE collection (dim 4, L2) and
// inserts ids 1..n, each carrying a deterministic "bucket" float group-by scalar
// (bucket = float(id%3) + 0.5, giving 3 distinct float keys: 0.5, 1.5, 2.5).
// Float keys are groupable via groupKeyString (ValueFloat case), so this exercises
// the full coordinator fan-out path with a float group-by key.
func seedFloatGroupCollection(t *testing.T, s Store, coll string, P, n int) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		bucket := vector.NewFloat(float64(id%3) + 0.5) // 0.5, 1.5, 2.5 cycling
		v := []float32{float32(id), 0, 0, 0}
		opts := VectorInsertOpts{Metadata: VectorMetadata{"bucket": bucket}}
		if err := s.VectorInsertExt(ctx, coll, id, v, opts); err != nil {
			t.Fatalf("VectorInsertExt %s/%d: %v", coll, id, err)
		}
	}
}

// TestQueryGroupFanOutFloatKey is the grouped P4==P1 exactness invariant for a
// FLOAT group-by key (bucket=float(id%3)+0.5). Proves the coordinator correctly
// round-trips float group keys through the id→key map (EncodeQueryResultGroupedFanOut
// / DecodeQueryResultGroupedFanOut) and that GroupDocuments groups by float key
// deterministically. RED if float keys are corrupted, dropped, or merged incorrectly.
func TestQueryGroupFanOutFloatKey(t *testing.T) {
	const n, k, groupSize = 120, 3, 2
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}

	pspec := &pb.QuerySpec{
		Mode:      pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:  []*pb.QueryLeaf{denseLeaf(denseQ, 50)},
		K:         int32(k),
		GroupBy:   "bucket",
		GroupSize: uint32(groupSize),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedFloatGroupCollection(t, s1, "flt1", 1, n)
	got1, _, err := s1.(*embedded).VectorQueryGrouped(ctx, "flt1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 float-key grouped query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatalf("P1 float-key grouped query returned no groups")
	}
	// Sanity: all returned group keys must be float-kind.
	for i, g := range got1 {
		if g.Key.Kind != vector.ValueFloat {
			t.Errorf("P1 group %d key kind = %v, want ValueFloat", i, g.Key.Kind)
		}
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedFloatGroupCollection(t, sP, "flt4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQueryGrouped(ctx, "flt4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 float-key grouped query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 float-key grouped query degraded")
	}
	assertSameGroups(t, "P4==P1 float-key", gotP, got1)
}
