// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// MV filter fan-out tests.
//
// These verify that a payload filter threaded through embedded.VectorMVSearch
// rides EVERY per-partition arg in embedded.mvFanOut, so a filtered MV search
// over a PARTITIONED collection (P>1) returns the correct GLOBAL top-k of ONLY
// matching docs — and that the partition-local-filter + score-ordered top-k
// merge is exact (partition-count invariance vs P=1).

// mvFilterDoc is the deterministic test corpus shape: doc id has a single token
// at angle θ_i (mvTokenAt) and metadata {"group": id%2, "docid": id}. The query
// is mvTokenAt(1) so MaxSim = cos((id-1)*Δθ) is STRICTLY decreasing in id (all
// θ in the first quadrant for the chosen N) — i.e. global rank == ascending id,
// tie-free. A group==0 filter keeps the even ids; their expected order is
// 2,4,6,... (ascending id), which is the independent ground truth below.
const (
	mvFilterN      = 24 // ids 1..24; θ_24 = 24·(π/80) < π/2 stays in the first quadrant
	mvFilterWinner = 1  // query token; makes MaxSim strictly decreasing in id
)

// expectedMVFilterEvenIDs is the independent ground truth: the even ids (group
// 0) in ascending id order, which equals descending-MaxSim order for the
// mvTokenAt(1) query. Truncated to k by the caller.
func expectedMVFilterEvenIDs(k int) []uint64 {
	var out []uint64
	for id := 2; id <= mvFilterN; id += 2 {
		out = append(out, uint64(id))
		if len(out) == k {
			break
		}
	}
	return out
}

func populateMVFilterCorpus(t *testing.T, store rostam.Store, name string) {
	t.Helper()
	ctx := context.Background()
	for id := 1; id <= mvFilterN; id++ {
		idc := uint64(id)
		md := rostam.VectorMetadata{
			"group": vector.NewInt(int64(id % 2)),
			"docid": vector.NewInt(int64(id)),
		}
		retryUntil(t, "mv add", func() error {
			return store.VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(id)}, md)
		})
	}
}

// group0Filter selects docs with metadata group==0 (the even ids).
func group0Filter() vector.Filter {
	return vector.Filter{Op: vector.FilterEq, Field: "group", Value: vector.NewInt(0)}
}

// TestMVSearchFilterPartitionedTopK is the core fan-out correctness test: a
// filtered MV search over a P=4 partitioned MV collection returns the GLOBAL
// top-k of ONLY matching docs, verified against independent ground truth
// (expectedMVFilterEvenIDs) and driven from a NON-creating coordinator so the
// per-partition filter encode + cross-node merge are both exercised.
func TestMVSearchFilterPartitionedTopK(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const name = "mvfilt"

	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})
	if p, _, ok := stores[0].(*rostam.Embedded).Catalog().PartitionsGen(name); !ok || p != 4 {
		t.Fatalf("PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}
	populateMVFilterCorpus(t, stores[0], name)

	const k = 5
	// Drive from node 1 (a non-creating coordinator) to exercise the cross-node
	// fan-out + filter encode on each partition's args. Wait for node 1's local
	// catalog to converge to P=4 first, else it briefly routes as if the collection
	// were single-partition and searches the empty logical collection (a flake).
	waitEmbeddedCatalog(t, stores[1].(*rostam.Embedded), name, 4, 5*time.Second)
	res, _, err := stores[1].VectorMVSearch(ctx, name,
		[][]float32{mvTokenAt(mvFilterWinner)}, k,
		rostam.MultiSearchOpts{CandidatesPerToken: 100, Filter: group0Filter()})
	if err != nil {
		t.Fatal(err)
	}
	got := mvResultIDs(res)
	want := expectedMVFilterEvenIDs(k)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("filtered partitioned MV top-k = %v, want %v (only even/group==0 ids, descending MaxSim)", got, want)
	}
	// Every returned doc must actually be group==0 and carry intact metadata.
	for _, r := range res {
		gv, ok := r.Metadata["group"]
		if !ok || !gv.Equal(vector.NewInt(0)) {
			t.Fatalf("result id=%d metadata[group]=%+v (present=%v), want NewInt(0) — a non-matching doc leaked through the fan-out filter", r.ID, gv, ok)
		}
	}
}

// TestMVSearchFilterPartitionCountInvariance proves the fan-out + merge is
// correct: the SAME corpus + SAME filter yields the identical ordered result
// whether the collection has P=4 partitions or P=1 (single shard). If the
// filter were applied inconsistently across partitions, or the merge mis-ordered
// the partition-local results, the two would diverge.
func TestMVSearchFilterPartitionCountInvariance(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	run := func(name string, partitions int) []uint64 {
		retryUntil(t, "mv create", func() error {
			return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: partitions})
		})
		populateMVFilterCorpus(t, stores[0], name)
		// P=1 is a plain logical collection (no partition catalog entry); only the
		// P>1 path needs to wait for the catalog to record the partition count.
		if partitions > 1 {
			waitEmbeddedCatalog(t, stores[0].(*rostam.Embedded), name, partitions, 5*time.Second)
		}
		const k = 6
		res, _, err := stores[0].VectorMVSearch(ctx, name,
			[][]float32{mvTokenAt(mvFilterWinner)}, k,
			rostam.MultiSearchOpts{CandidatesPerToken: 100, Filter: group0Filter()})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return mvResultIDs(res)
	}

	gotP4 := run("mvinv_p4", 4)
	gotP1 := run("mvinv_p1", 1)

	if !reflect.DeepEqual(gotP4, gotP1) {
		t.Fatalf("partition-count variance: P=4 filtered result %v != P=1 filtered result %v (fan-out + merge must equal single-partition)", gotP4, gotP1)
	}
	want := expectedMVFilterEvenIDs(6)
	if !reflect.DeepEqual(gotP1, want) {
		t.Fatalf("P=1 filtered result %v != ground truth %v", gotP1, want)
	}
}

// TestMVSearchFilterPartitionedFillsK proves the adaptive over-fetch
// works through the fan-out: a SELECTIVE filter (group==0 keeps only half the
// corpus) still FILLS the global k when ≥k matches exist globally. The matches
// are spread across all 4 partitions (ids hash by docID), so each partition
// over-fetches its local candidates and the coordinator merge fills k.
func TestMVSearchFilterPartitionedFillsK(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const name = "mvfillk"

	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})
	populateMVFilterCorpus(t, stores[0], name)
	waitEmbeddedCatalog(t, stores[1].(*rostam.Embedded), name, 4, 5*time.Second)

	// There are 12 even ids globally (2..24); ask for k=10 and require a full fill.
	const k = 10
	res, _, err := stores[1].VectorMVSearch(ctx, name,
		[][]float32{mvTokenAt(mvFilterWinner)}, k,
		rostam.MultiSearchOpts{CandidatesPerToken: 100, Filter: group0Filter()})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != k {
		t.Fatalf("selective filter under-filled global k: got %d results, want %d (adaptive over-fetch must fill k across partitions)", len(res), k)
	}
	got := mvResultIDs(res)
	want := expectedMVFilterEvenIDs(k)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("filled-k result = %v, want %v", got, want)
	}
}

// TestMVSearchNoFilterPartitionedUnchanged is the regression guard: a no-filter
// partitioned MV search returns the global top-k of ALL docs (ascending id from
// the mvTokenAt(1) query), unaffected by the filter plumbing.
func TestMVSearchNoFilterPartitionedUnchanged(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const name = "mvnofilt"

	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})
	populateMVFilterCorpus(t, stores[0], name)
	waitEmbeddedCatalog(t, stores[1].(*rostam.Embedded), name, 4, 5*time.Second)

	const k = 5
	res, _, err := stores[1].VectorMVSearch(ctx, name,
		[][]float32{mvTokenAt(mvFilterWinner)}, k,
		rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	got := mvResultIDs(res)
	// No filter: every id 1..k in ascending order (strictly decreasing MaxSim).
	want := []uint64{1, 2, 3, 4, 5}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("no-filter partitioned MV top-k = %v, want %v (regression: filter plumbing changed the no-filter path)", got, want)
	}
}

// TestMVSearchFilterFanOutAntiSilentDrop is the anti-silent-drop gate: it would
// FAIL if the filter were dropped on the per-partition fan-out encode.
//
// We pick a query token (mvTokenAt(2)) whose NEAREST doc by MaxSim is an ODD,
// NON-matching id (here id=2's closest neighbours include id=1 and id=3, which
// are odd / group==1). With the group==0 filter applied, id=1 (the literal
// global rank-0 of an UNFILTERED search, but a NON-match) MUST be absent. If the
// filter did not ride the per-partition args, id=1's partition would return it
// unfiltered and it would surface at the top of the merged result — so this test
// catches a dropped filter on the fan-out path exactly like the linearizable
// reads regression it mirrors.
func TestMVSearchFilterFanOutAntiSilentDrop(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const name = "mvdrop"

	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})
	populateMVFilterCorpus(t, stores[0], name)
	waitEmbeddedCatalog(t, stores[1].(*rostam.Embedded), name, 4, 5*time.Second)

	const k = 5
	// Query at id=2's angle: unfiltered, the top neighbours would be {2,3,1,4,...}
	// — i.e. id=1 and id=3 (ODD, group==1) rank ABOVE most even ids. With the
	// group==0 filter, NONE of the odd ids may appear.
	res, _, err := stores[1].VectorMVSearch(ctx, name,
		[][]float32{mvTokenAt(2)}, k,
		rostam.MultiSearchOpts{CandidatesPerToken: 100, Filter: group0Filter()})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.ID%2 != 0 {
			t.Fatalf("anti-silent-drop FAILED: odd/non-matching id=%d present in filtered fan-out result %v — the filter was DROPPED on the per-partition encode", r.ID, mvResultIDs(res))
		}
	}
	// And the top result must be the nearest EVEN id (2), not the nearer odd id 1.
	if len(res) == 0 || res[0].ID != 2 {
		t.Fatalf("filtered fan-out rank-0 = %v, want id=2 (nearest matching even doc; id=1/id=3 are non-matching odds)", mvResultIDs(res))
	}
}
