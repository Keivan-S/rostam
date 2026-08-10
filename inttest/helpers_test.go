// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"math/rand"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// createCollectionTolerant creates collection coll with cfg on the creating
// coordinator, tolerating the commit-but-returns-transient race that makes a bare
// retryUntil(create) flake under load: a create that commits then surfaces a
// non-not-leader transient (or a not-leader that clears only after the entry
// landed) makes the NEXT attempt see "already exists" — which retryUntil fatals
// on (it only retries rostam.ErrNotLeader). Here "already exists" is treated as
// success (the physical partitions are there from the prior attempt) and the
// catalog entry is idempotently completed; not-leader is retried. It still fails
// LOUD if the coordinator's catalog never converges to (P, gen 0) — so a create
// that genuinely never committed is still caught. Mirrors createPartitionedDocs'
// tolerant pattern, generalized to any collection + VectorConfig.
func createCollectionTolerant(t *testing.T, ctx context.Context, coord rostam.Store, coll string, cfg rostam.VectorConfig) {
	t.Helper()
	wantP := cfg.Partitions
	if wantP < 1 {
		wantP = 1
	}
	// A partitioned (P>1) collection registers a (P, gen 0) catalog entry we can
	// poll for convergence; a single-partition (P<=1) collection routes via the
	// plain logical name and registers NO partitioned catalog entry, so its
	// convergence is the create returning success (no PartitionsGen to wait on).
	partitioned := wantP > 1
	deadline := time.Now().Add(30 * time.Second)
	created := false
	for time.Now().Before(deadline) {
		err := coord.CreateCollection(ctx, coll, cfg)
		if err == nil {
			created = true
			break
		}
		if strings.Contains(err.Error(), "already exists") {
			// A prior attempt committed but (possibly) returned a transient before
			// finishing — treat as success. For a partitioned collection, also
			// complete the SetPartitionsGen catalog write idempotently (the
			// partitioned convergence is asserted below).
			if partitioned {
				if serr := coord.(*rostam.Embedded).Catalog().SetPartitionsGen(coll, wantP, 0); serr != nil {
					time.Sleep(50 * time.Millisecond)
					continue
				}
			}
			created = true
			break
		}
		if !strings.Contains(err.Error(), "not leader") {
			t.Fatalf("create %s: %v", coll, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !created {
		t.Fatalf("create %s: never succeeded within budget", coll)
	}
	// Fail loud if a partitioned catalog never converged — a create that never
	// committed is NOT silently swallowed.
	if partitioned {
		waitEmbeddedCatalogGen(t, coord.(*rostam.Embedded), coll, wantP, 0, 15*time.Second)
	}
}

// newSingleEmbedded builds a single-node embedded Store suitable for unit
// tests. It uses a temp directory, bootstraps a fresh cluster, and
// registers the built-in ops. The Store is closed via t.Cleanup.
//
// Copied verbatim (with rostam.* qualification) from the root embedded_test.go
// because several moved public-API integration tests depend on it; the bulk of
// embedded_test.go stays in the root package until a later task.
func newSingleEmbedded(t *testing.T) rostam.Store {
	t.Helper()
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}

	s, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "test-node",
		DataDir:   dir,
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatalf("NewEmbedded: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Logf("embedded Close: %v", err)
		}
	})
	return s
}

// waitLeaderEmbedded spins until the node reports a non-empty leader address,
// ensuring Raft has elected a leader before test ops run. Copied (qualified)
// from the root embedded_test.go for the same reason as newSingleEmbedded.
func waitLeaderEmbedded(t *testing.T, s rostam.Store) {
	t.Helper()
	// load-flakiness hardening: leader-election readiness gate widened 10s->30s.
	// Setup-only — cuts false "timed out waiting for leader election" fatals under
	// CPU contention; cannot mask a behavioral regression.
	deadline := time.Now().Add(cpuScaled(30 * time.Second))
	for time.Now().Before(deadline) {
		if s.LeaderAddr(nil) != "" {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("waitLeaderEmbedded: timed out waiting for leader election")
}

// dialGRPC opens an insecure gRPC client to addr and returns the vector
// service client plus a cleanup closure. Copied (verbatim, no rostam refs)
// from the root grpc_test.go because two moved transport tests use it.
func dialGRPC(t *testing.T, addr string) (pb.VectorServiceClient, func()) {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	return pb.NewVectorServiceClient(conn), func() { _ = conn.Close() }
}

// must fails the test on a non-nil error. Copied from the root embedded_test.go
// for the moved write-consistency tests.
func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// physVecExists reports whether id is live in the given physical dense partition
// (bypassing logical→physical routing). Copied (with rostam.* qualification) from
// the root embedded_test.go for the moved fanout/reshard dual-write tests.
func physVecExists(t *testing.T, ee *rostam.Embedded, phys string, id uint64) bool {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_exists", ops.EncodeExistsArgs(phys, id))
	if err != nil {
		t.Fatalf("vector_exists %q id=%d: %v", phys, id, err)
	}
	ok, err := ops.DecodeExistsResult(body)
	if err != nil {
		t.Fatalf("decode exists %q: %v", phys, err)
	}
	return ok
}

// physPayloadTag reads id's "tag" payload value directly from the given physical
// partition. Copied from the root embedded_test.go for the moved reshard tests.
func physPayloadTag(t *testing.T, ee *rostam.Embedded, phys string, id uint64) (int64, bool) {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_get", ops.EncodeVectorGetArgs(phys, id, ops.GetFlagsBoth))
	if err != nil {
		t.Fatalf("vector_get %q id=%d: %v", phys, id, err)
	}
	found, _, meta, _, _, err := ops.DecodeVectorGetResult(body)
	if err != nil {
		t.Fatalf("decode get %q: %v", phys, err)
	}
	if !found {
		return 0, false
	}
	v, ok := meta["tag"]
	if !ok {
		return 0, false
	}
	return v.Int, true
}

// physMVExists reports whether docID is live in the given physical MV partition.
// Copied from the root embedded_test.go for the moved MV reshard tests.
func physMVExists(t *testing.T, ee *rostam.Embedded, phys string, docID uint64) bool {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_mv_exists", ops.EncodeMVExistsArgs(phys, docID))
	if err != nil {
		t.Fatalf("vector_mv_exists %q id=%d: %v", phys, docID, err)
	}
	ok, err := ops.DecodeExistsResult(body)
	if err != nil {
		t.Fatalf("decode mv exists %q: %v", phys, err)
	}
	return ok
}

// mustCall runs a raw op and discards the body, surfacing the error for must().
// Copied from the root embedded_test.go (dependency of setupReshardingDense/MV).
func mustCall(ee *rostam.Embedded, op string, args []byte) error {
	_, err := ee.Call(context.Background(), op, args)
	return err
}

// setupReshardingDense builds a dense collection in Resharding{OldP,OldGen=0,
// NewP,NewGen=1} and returns the embedded backend. Copied (with rostam.*
// qualification) from the root embedded_test.go for the moved fanout/reshard tests.
func setupReshardingDense(t *testing.T, coll string, oldP, newP int) *rostam.Embedded {
	t.Helper()
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*rostam.Embedded)
	cfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	for p := 0; p < oldP; p++ {
		must(t, mustCall(ee, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen(coll, 0, p)), cfg)))
	}
	for p := 0; p < newP; p++ {
		must(t, mustCall(ee, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen(coll, 1, p)), cfg)))
	}
	must(t, ee.Catalog().SetPartitionsGen(coll, oldP, 0))
	must(t, ee.Catalog().SetReshardState(coll, rostam.ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	return ee
}

// setupReshardingMV is the MV analogue of setupReshardingDense. Copied from the
// root embedded_test.go for the moved MV reshard tests.
func setupReshardingMV(t *testing.T, coll string, oldP, newP int) *rostam.Embedded {
	t.Helper()
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*rostam.Embedded)
	mvCfg := rostam.MultiVectorConfig{Dim: 4}
	for p := 0; p < oldP; p++ {
		must(t, mustCall(ee, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(coll, 0, p)), mvCfg)))
	}
	for p := 0; p < newP; p++ {
		must(t, mustCall(ee, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(coll, 1, p)), mvCfg)))
	}
	must(t, ee.Catalog().SetPartitionsGen(coll, oldP, 0))
	must(t, ee.Catalog().SetReshardState(coll, rostam.ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	return ee
}

// idSet collects the IDs of a document slice into a set. Copied (with rostam.*
// qualification) from the root embedded_test.go, which keeps its own copy for the
// unit tests that stay in the root package; the moved fan-out integration tests
// need it here. A small duplicated test helper is fine.
func idSet(docs []rostam.VectorDocument) map[uint64]bool {
	m := map[uint64]bool{}
	for _, d := range docs {
		m[d.ID] = true
	}
	return m
}

// sameFusedResults reports whether two hybrid-search result slices contain the
// same IDs with (near-)equal scores. Copied from the root embedded_test.go for
// the moved fan-out hybrid tests.
func sameFusedResults(a, b []rostam.VectorResult) bool {
	if len(a) != len(b) {
		return false
	}
	bScore := make(map[uint64]float32, len(b))
	for _, r := range b {
		bScore[r.ID] = r.Score
	}
	for _, r := range a {
		bs, ok := bScore[r.ID]
		if !ok {
			return false
		}
		d := r.Score - bs
		if d < 0 {
			d = -d
		}
		if d > 1e-5 {
			return false
		}
	}
	return true
}

// sameGroupsResults reports whether two group-search result slices match by key
// and per-group hit IDs. Copied from the root embedded_test.go for the moved
// group fan-out tests.
func sameGroupsResults(a, b []rostam.VectorGroup) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !a[i].Key.Equal(b[i].Key) {
			return false
		}
		if len(a[i].Hits) != len(b[i].Hits) {
			return false
		}
		for j := range a[i].Hits {
			if a[i].Hits[j].ID != b[i].Hits[j].ID {
				return false
			}
		}
	}
	return true
}

// mvDoc is the per-document multi-vector view (tokens + tag metadata) the moved
// reshard MV scans build. Copied verbatim from the root embedded_test.go.
type mvDoc struct {
	tokens [][]float32
	tag    int64 // metadata "tag" — bumped on overwrite so a clobber is detectable
}

// mvTokensFor builds a deterministic K-token matrix for id, each token a distinct
// unit vector so normalization is a no-op and MaxSim is tie-free. Copied from the
// root embedded_test.go; mvTokenAt lives in the moved fanout_dispatcher_test.go.
func mvTokensFor(id uint64) [][]float32 {
	k := int(id%3) + 1
	out := make([][]float32, k)
	for j := 0; j < k; j++ {
		out[j] = mvTokenAt(int(id)*4 + j)
	}
	return out
}

// mvTokensEqual reports whether two token matrices are byte-identical. Copied from
// the root embedded_test.go for the moved reshard MV correctness tests.
func mvTokensEqual(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

// sameDocs reports whether two document slices match by ID, content, and distance.
// Copied from the root embedded_test.go for the moved fan-out tests.
func sameDocs(a, b []rostam.VectorDocument) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Content != b[i].Content || a[i].Distance != b[i].Distance {
			return false
		}
	}
	return true
}

// docIDs collects the IDs of a document slice (order-preserving). Copied from the
// root embedded_test.go for the moved fan-out tests.
func docIDs(d []rostam.VectorDocument) []uint64 {
	out := make([]uint64, len(d))
	for i := range d {
		out[i] = d[i].ID
	}
	return out
}

// reshardScanGen scans every physical partition of (coll,gen,P) and returns a
// map docID -> vector, failing the test on any mis-routed or duplicated id. Copied
// (with rostam.* qualification) from the root embedded_test.go for the moved remote
// reshard tests.
func reshardScanGen(t *testing.T, ee *rostam.Embedded, coll string, P int, gen uint32) map[uint64][]float32 {
	t.Helper()
	out := map[uint64][]float32{}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, gen, p))
		body, err := ee.Call(context.Background(), "vector_scan_vectors", ops.EncodeScanVectorsArgs(phys))
		if err != nil {
			t.Fatalf("scan %s: %v", phys, err)
		}
		recs, err := ops.DecodeScanVectorsResult(body)
		if err != nil {
			t.Fatalf("decode scan %s: %v", phys, err)
		}
		for _, r := range recs {
			if want := ops.PartitionOf(r.ID, P); want != p {
				t.Fatalf("id %d found in partition %d but PartitionOf says %d (gen %d, P %d)", r.ID, p, want, gen, P)
			}
			if _, dup := out[r.ID]; dup {
				t.Fatalf("id %d present in more than one gen-%d partition", r.ID, gen)
			}
			out[r.ID] = append([]float32(nil), r.Vec...)
		}
	}
	return out
}

// reshardScanGenMV is the MV analogue of reshardScanGen (tokens + tag metadata).
// Copied from the root embedded_test.go for the moved remote MV reshard tests.
func reshardScanGenMV(t *testing.T, ee *rostam.Embedded, coll string, P int, gen uint32) map[uint64]mvDoc {
	t.Helper()
	out := map[uint64]mvDoc{}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, gen, p))
		body, err := ee.Call(context.Background(), "vector_mv_scan_vectors", ops.EncodeMVScanArgs(phys))
		if err != nil {
			t.Fatalf("mv scan %s: %v", phys, err)
		}
		recs, err := ops.DecodeMVScanResult(body)
		if err != nil {
			t.Fatalf("decode mv scan %s: %v", phys, err)
		}
		for _, r := range recs {
			if want := ops.PartitionOf(r.ID, P); want != p {
				t.Fatalf("doc %d found in partition %d but PartitionOf says %d (gen %d, P %d)", r.ID, p, want, gen, P)
			}
			if _, dup := out[r.ID]; dup {
				t.Fatalf("doc %d present in more than one gen-%d partition", r.ID, gen)
			}
			tag := int64(-1)
			if r.Metadata != nil {
				if v, ok := r.Metadata["tag"]; ok && v.Kind == vector.ValueInt {
					tag = v.Int
				}
			}
			toks := make([][]float32, len(r.Tokens))
			for i := range r.Tokens {
				toks[i] = append([]float32(nil), r.Tokens[i]...)
			}
			out[r.ID] = mvDoc{tokens: toks, tag: tag}
		}
	}
	return out
}

// genPartitionExists reports whether the physical partition (coll,gen,p) exists.
// Copied from the root embedded_test.go for the moved linearizable/meta tests.
func genPartitionExists(t *testing.T, ee *rostam.Embedded, coll string, gen uint32, p int) bool {
	t.Helper()
	phys := string(ops.PartitionKeyGen(coll, gen, p))
	_, err := ee.Call(context.Background(), "vector_get_config", ops.EncodeGetConfigArgs(phys))
	return err == nil
}

// pageAllDense scrolls a dense collection to exhaustion, asserting per-page and
// cross-page strict ascending order plus the full/short next-cursor exhaustion
// rule, returning the full id sequence and page count. Copied (with rostam.*
// qualification) from the root embedded_scroll_cursor_test.go for the moved
// remote scroll/cursor fan-out tests.
func pageAllDense(t *testing.T, s rostam.Store, coll string, filter rostam.VectorFilter, limit int) (ids []uint64, pages int) {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var last uint64
	have := false
	for {
		docs, _, next, err := s.VectorScroll(ctx, coll, filter, limit, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			t.Fatalf("VectorScroll page %d: %v", pages, err)
		}
		pages++
		for i, d := range docs {
			if i > 0 && d.ID <= docs[i-1].ID {
				t.Fatalf("page %d not strictly ascending at %d: %d <= %d", pages, i, d.ID, docs[i-1].ID)
			}
			if have && d.ID <= last {
				t.Fatalf("page %d id %d not > previous page's last %d (gap/dup/order bug)", pages, d.ID, last)
			}
			ids = append(ids, d.ID)
			last = d.ID
			have = true
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("page %d full (len=%d) but next_cursor empty", pages, limit)
			}
		} else if next != "" {
			t.Fatalf("page %d short (len=%d<%d) but next_cursor=%q (not exhausted)", pages, len(docs), limit, next)
		}
		if next == "" {
			return ids, pages
		}
		cursor = next
		if pages > limit*1000+100 { // runaway guard
			t.Fatalf("pagination did not terminate after %d pages", pages)
		}
	}
}

// assertExactlyOnceAscending asserts the paged id sequence is globally strictly
// ascending and equals want exactly once each (no gaps, no dups). Copied from the
// root embedded_scroll_cursor_test.go for the moved scroll fan-out tests.
func assertExactlyOnceAscending(t *testing.T, got []uint64, want map[uint64]bool) {
	t.Helper()
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Fatalf("not globally ascending at %d: %d <= %d", i, got[i], got[i-1])
		}
	}
	seen := make(map[uint64]int, len(got))
	for _, id := range got {
		seen[id]++
	}
	for id := range want {
		if seen[id] != 1 {
			t.Fatalf("id %d appeared %d times across pages, want exactly 1", id, seen[id])
		}
	}
	for id, n := range seen {
		if !want[id] {
			t.Fatalf("unexpected id %d (×%d) not in the want set", id, n)
		}
	}
	if len(got) != len(want) {
		t.Fatalf("total paged %d ids, want %d", len(got), len(want))
	}
}

// shuffledIDs returns ids 1..n in a deterministic shuffled order (seed-driven) so a
// stable ascending paged result proves the merge, not insert order. Copied from the
// root embedded_scroll_cursor_test.go for the moved scroll fan-out tests.
func shuffledIDs(n int, seed int64) []uint64 {
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1) // distinct, tie-free, starting at 1
	}
	r := rand.New(rand.NewSource(seed))
	r.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	return ids
}

// itoa is an allocation-light int→string used by the moved transport/cluster
// tests. Copied verbatim from the root http_test.go (which stays put).
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
