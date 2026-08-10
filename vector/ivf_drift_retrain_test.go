// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// IVF deterministic auto-retrain-on-drift tests.
//
// The #1 obligation is DETERMINISM: the two-stage drift trigger is a pure function
// of applied (Raft-replicated) state (the live count + the slot-ordered live
// sample's mean nearest-centroid distance), so two replicas applying the IDENTICAL
// insert sequence with IVFDriftRetrain=ON produce BIT-IDENTICAL trained state after
// a drift-triggered retrain (the replica-consistency oracle, TestIVFDriftReplicaOracle).
// Default-OFF stays byte-identical (TestIVFDriftDefaultOffByteIdentical).

// ivfDriftConfig is an IVF-Flat config with drift-retrain ON and a low train
// threshold + low growth factor so a drift retrain engages quickly in tests.
func ivfDriftConfig(dim, threshold int, growth, drift float64) Config {
	c := ivfAutoTrainConfig(dim, threshold)
	c.IVFDriftRetrain = true
	c.IVFDriftGrowthFactor = growth
	c.IVFDriftFactor = drift
	return c
}

// snapshotBytes serializes ix and returns the raw bytes — the byte-level fingerprint
// of the full trained state (arena + centroids + lists + PQ trailer + drift
// checkpoint). Two indexes with identical bytes are bit-identical.
func snapshotBytes(t *testing.T, ix *ivf) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	return buf.Bytes()
}

// driftedVecs returns n vectors drawn from a single gaussian blob centered far from
// the origin (offset on every dim), simulating a distribution shift away from the
// original clustered centroids.
func driftedVecs(rng *rand.Rand, n, dim int, offset float32) [][]float32 {
	vs := make([][]float32, n)
	for i := range vs {
		v := make([]float32, dim)
		for d := range v {
			v[d] = offset + float32(rng.NormFloat64())
		}
		vs[i] = v
	}
	return vs
}

// TestMeanAssignCostDeterministic: the drift metric is bit-deterministic — the same
// (sample, centroids) yields the identical float32 across repeated calls (fixed
// slot-ordered sum, no map iteration).
func TestMeanAssignCostDeterministic(t *testing.T) {
	dim := 16
	rng := rand.New(rand.NewSource(3))
	ix, err := newIVF(ivfAutoTrainConfig(dim, 256))
	if err != nil {
		t.Fatal(err)
	}
	sample := clusteredVecs(rng, 300, dim, 8)
	centroids := clusteredVecs(rng, 12, dim, 12)
	a := ix.meanAssignCost(sample, centroids)
	b := ix.meanAssignCost(sample, centroids)
	if math.Float32bits(a) != math.Float32bits(b) {
		t.Fatalf("meanAssignCost not bit-deterministic: %v (%x) != %v (%x)", a, math.Float32bits(a), b, math.Float32bits(b))
	}
	// And again on a fresh index with the same metric/dim — kernel choice is stable.
	ix2, _ := newIVF(ivfAutoTrainConfig(dim, 256))
	c := ix2.meanAssignCost(sample, centroids)
	if math.Float32bits(a) != math.Float32bits(c) {
		t.Fatalf("meanAssignCost differs across indexes: %x != %x", math.Float32bits(a), math.Float32bits(c))
	}
}

// TestIVFDriftReplicaOracle (THE #1): two indexes fed the IDENTICAL insert sequence
// (clustered backlog → first auto-train, then a drifting cluster that crosses the
// drift threshold → a drift retrain) end up BIT-IDENTICAL: same snapshot bytes, same
// centroids/lists, same lastTrainCount/lastTrainCost.
func TestIVFDriftReplicaOracle(t *testing.T) {
	dim := 24
	const threshold = 400
	// growth 1.5, drift 1.2 — engage a retrain after a moderate drift+growth.
	cfg := ivfDriftConfig(dim, threshold, 1.5, 1.2)

	rng := rand.New(rand.NewSource(11))
	base := clusteredVecs(rng, threshold, dim, 8) // crosses train threshold
	shift := driftedVecs(rng, 500, dim, 50)       // far-away cluster → drift
	ids := seqIDs(threshold + len(shift))
	allVecs := append(append([][]float32(nil), base...), shift...)

	buildOne := func() *ivf {
		ix, err := newIVF(cfg)
		if err != nil {
			t.Fatal(err)
		}
		insertAll(t, ix, ids, allVecs)
		return ix
	}
	a := buildOne()
	b := buildOne()

	if !a.trained || !b.trained {
		t.Fatalf("both replicas must be trained (a=%v b=%v)", a.trained, b.trained)
	}
	if a.lastTrainCount != b.lastTrainCount {
		t.Fatalf("lastTrainCount diverged: a=%d b=%d", a.lastTrainCount, b.lastTrainCount)
	}
	if math.Float32bits(a.lastTrainCost) != math.Float32bits(b.lastTrainCost) {
		t.Fatalf("lastTrainCost diverged: a=%x b=%x", math.Float32bits(a.lastTrainCost), math.Float32bits(b.lastTrainCost))
	}
	if len(a.centroids) != len(b.centroids) {
		t.Fatalf("nlist diverged: a=%d b=%d", len(a.centroids), len(b.centroids))
	}
	for c := range a.centroids {
		for d := range a.centroids[c] {
			if math.Float32bits(a.centroids[c][d]) != math.Float32bits(b.centroids[c][d]) {
				t.Fatalf("centroid[%d][%d] diverged: a=%x b=%x", c, d, math.Float32bits(a.centroids[c][d]), math.Float32bits(b.centroids[c][d]))
			}
		}
	}
	for c := range a.lists {
		if len(a.lists[c]) != len(b.lists[c]) {
			t.Fatalf("list[%d] length diverged: a=%d b=%d", c, len(a.lists[c]), len(b.lists[c]))
		}
		for i := range a.lists[c] {
			if a.lists[c][i] != b.lists[c][i] {
				t.Fatalf("list[%d][%d] diverged: a=%d b=%d", c, i, a.lists[c][i], b.lists[c][i])
			}
		}
	}
	// The strongest check: the full serialized state is byte-identical.
	if !bytes.Equal(snapshotBytes(t, a), snapshotBytes(t, b)) {
		t.Fatalf("replica snapshot bytes diverged — retrain is NOT a pure function of applied state")
	}
}

// TestIVFDriftFiresOnDrift: inserting a cluster FAR from the original centroids drives
// the mean assignment cost past the drift threshold → the centroids CHANGE (a retrain
// fired) and post-retrain recall on the drifted distribution is >= the stale-centroid
// recall.
func TestIVFDriftFiresOnDrift(t *testing.T) {
	dim := 24
	const threshold = 400
	cfg := ivfDriftConfig(dim, threshold, 1.5, 1.2)

	rng := rand.New(rand.NewSource(21))
	base := clusteredVecs(rng, threshold, dim, 8)
	ids0 := seqIDs(threshold)

	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids0, base)
	if !ix.trained {
		t.Fatalf("index must auto-train at threshold %d", threshold)
	}
	staleCentroids := make([][]float32, len(ix.centroids))
	for c := range ix.centroids {
		staleCentroids[c] = append([]float32(nil), ix.centroids[c]...)
	}
	firstTrainCount := ix.lastTrainCount

	// Insert a far-away drifting cluster, enough to (a) cross the growth checkpoint and
	// (b) push the mean cost past driftFactor.
	shift := driftedVecs(rng, 500, dim, 50)
	shiftIDs := make([]uint64, len(shift))
	for i := range shiftIDs {
		shiftIDs[i] = uint64(threshold + 1 + i)
	}
	insertAll(t, ix, shiftIDs, shift)

	// A retrain must have fired: lastTrainCount advanced past the first-train count and
	// the centroids moved.
	if ix.lastTrainCount <= firstTrainCount {
		t.Fatalf("drift retrain did not fire: lastTrainCount still %d (first-train %d)", ix.lastTrainCount, firstTrainCount)
	}
	changed := false
	for c := range ix.centroids {
		for d := range ix.centroids[c] {
			if math.Abs(float64(ix.centroids[c][d]-staleCentroids[c][d])) > 1e-6 {
				changed = true
				break
			}
		}
		if changed {
			break
		}
	}
	if !changed {
		t.Fatalf("centroids unchanged after a drift retrain — retrain did not actually re-fit")
	}

	// Recall sanity: search a drifted query against the retrained index vs the stale
	// centroids. The retrained index should be at least as good (centroids now cover
	// the drifted cluster).
	q := shift[0]
	gt := bruteForceTopK(base, shift, q, 10) // ground truth over all live vecs
	retr := searchIDs(t, ix, q, 10)
	rRetr := recallSet(gt, retr)

	// Build a control index that is trained ONLY on base (stale centroids), then add
	// the shift WITHOUT drift-retrain, so its centroids stay stale.
	ctrlCfg := ivfAutoTrainConfig(dim, threshold) // drift OFF
	ctrl, err := newIVF(ctrlCfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ctrl, ids0, base)
	insertAll(t, ctrl, shiftIDs, shift)
	ctrlRes := searchIDs(t, ctrl, q, 10)
	rCtrl := recallSet(gt, ctrlRes)
	if rRetr < rCtrl {
		t.Fatalf("post-retrain recall %.2f < stale-centroid recall %.2f — retrain hurt recall", rRetr, rCtrl)
	}
}

// TestIVFDriftDefersOnStableGrowth: inserting MORE of the SAME distribution (no shift)
// past the growth checkpoint takes the DEFER path — lastTrainCount bumps but the
// centroids stay UNCHANGED (no needless retrain).
func TestIVFDriftDefersOnStableGrowth(t *testing.T) {
	dim := 24
	const threshold = 400
	cfg := ivfDriftConfig(dim, threshold, 1.5, 1.5)

	rng := rand.New(rand.NewSource(31))
	// Build the base sample, train, and snapshot its centroids.
	base := clusteredVecs(rng, threshold, dim, 8)
	ids0 := seqIDs(threshold)
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids0, base)
	if !ix.trained {
		t.Fatalf("index must auto-train at threshold")
	}
	trainCentroids := make([][]float32, len(ix.centroids))
	for c := range ix.centroids {
		trainCentroids[c] = append([]float32(nil), ix.centroids[c]...)
	}
	firstCount := ix.lastTrainCount
	firstCost := ix.lastTrainCost

	// Insert MORE of the SAME 8-cluster distribution, crossing the 1.5x checkpoint.
	more := clusteredVecsFrom(rng, 500, dim, base) // same blobs as base's clusters
	moreIDs := make([]uint64, len(more))
	for i := range moreIDs {
		moreIDs[i] = uint64(threshold + 1 + i)
	}
	insertAll(t, ix, moreIDs, more)

	// DEFER: lastTrainCount advanced (checkpoint pushed) but centroids unchanged and
	// lastTrainCost left at its train-time reference.
	if ix.lastTrainCount <= firstCount {
		t.Fatalf("stable growth must bump lastTrainCount (defer): still %d (first %d)", ix.lastTrainCount, firstCount)
	}
	if math.Float32bits(ix.lastTrainCost) != math.Float32bits(firstCost) {
		t.Fatalf("defer must leave lastTrainCost unchanged: %x != %x", math.Float32bits(ix.lastTrainCost), math.Float32bits(firstCost))
	}
	for c := range ix.centroids {
		for d := range ix.centroids[c] {
			if math.Float32bits(ix.centroids[c][d]) != math.Float32bits(trainCentroids[c][d]) {
				t.Fatalf("centroid[%d][%d] changed on a stable-growth DEFER (no retrain expected)", c, d)
			}
		}
	}
}

// TestIVFDriftDefaultOffByteIdentical: with IVFDriftRetrain=false the trained state
// after the same insert sequence is BYTE-IDENTICAL to an index built without the
// feature — the snapshot is a v3 blob (pre-feature format), not v4. This is the
// forward-compat / default-off guarantee: a feature-OFF collection produces a
// snapshot an old binary can still open (rolling-upgrade safety).
func TestIVFDriftDefaultOffByteIdentical(t *testing.T) {
	dim := 24
	const threshold = 400
	rng := rand.New(rand.NewSource(41))
	base := clusteredVecs(rng, threshold, dim, 8)
	shift := driftedVecs(rng, 500, dim, 50)
	ids := seqIDs(threshold + len(shift))
	allVecs := append(append([][]float32(nil), base...), shift...)

	// Index A: plain auto-train config (no drift fields at all).
	offCfg := ivfAutoTrainConfig(dim, threshold)
	a, err := newIVF(offCfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, a, ids, allVecs)

	// Index B: identical config but with the three drift fields explicitly zero/false.
	bCfg := ivfAutoTrainConfig(dim, threshold)
	bCfg.IVFDriftRetrain = false
	bCfg.IVFDriftGrowthFactor = 0
	bCfg.IVFDriftFactor = 0
	b, err := newIVF(bCfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, b, ids, allVecs)

	// Both snap as v3 (drift OFF → no checkpoint trailer). The bytes must be equal
	// to each other AND to a pre-feature v3 snapshot (verified by checking that the
	// snapshot version word is 3, not 4).
	snapA := snapshotBytes(t, a)
	snapB := snapshotBytes(t, b)
	if !bytes.Equal(snapA, snapB) {
		t.Fatalf("drift-OFF snapshot bytes differ from the no-feature index — default-OFF not byte-identical")
	}
	// Verify the version word (offset 8, after the 8-byte magic) is 3, not 4.
	const magicLen = 8
	if len(snapA) < magicLen+4 {
		t.Fatalf("snapshot too short to contain version word")
	}
	ver := uint32(snapA[magicLen])<<24 | uint32(snapA[magicLen+1])<<16 | uint32(snapA[magicLen+2])<<8 | uint32(snapA[magicLen+3])
	if ver != 3 {
		t.Fatalf("drift-OFF snapshot must be v3 (old-reader compatible), got v%d", ver)
	}
}

// TestIVFDriftPersistenceRoundTrip: lastTrainCount/lastTrainCost survive a snapshot +
// restore round-trip.
func TestIVFDriftPersistenceRoundTrip(t *testing.T) {
	dim := 24
	const threshold = 400
	cfg := ivfDriftConfig(dim, threshold, 1.5, 1.2)
	rng := rand.New(rand.NewSource(51))
	base := clusteredVecs(rng, threshold, dim, 8)
	shift := driftedVecs(rng, 500, dim, 50)
	ids := seqIDs(threshold + len(shift))
	allVecs := append(append([][]float32(nil), base...), shift...)

	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids, allVecs)
	if ix.lastTrainCount == 0 {
		t.Fatalf("expected a non-zero lastTrainCount after train+retrain")
	}

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	restored, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if restored.lastTrainCount != ix.lastTrainCount {
		t.Fatalf("lastTrainCount not round-tripped: got %d want %d", restored.lastTrainCount, ix.lastTrainCount)
	}
	if math.Float32bits(restored.lastTrainCost) != math.Float32bits(ix.lastTrainCost) {
		t.Fatalf("lastTrainCost not round-tripped: got %x want %x", math.Float32bits(restored.lastTrainCost), math.Float32bits(ix.lastTrainCost))
	}
}

// TestIVFDriftOldSnapshotBackCompat: an OLD (pre-drift, v3) snapshot restores with
// lastTrainCount/lastTrainCost defaulted to 0 (drift simply re-arms on the next train).
// We forge a v3 snapshot by re-serializing the v4 core WITHOUT the drift trailer.
func TestIVFDriftOldSnapshotBackCompat(t *testing.T) {
	dim := 16
	const threshold = 256
	cfg := ivfDriftConfig(dim, threshold, 2.0, 1.5)
	rng := rand.New(rand.NewSource(61))
	base := clusteredVecs(rng, threshold, dim, 8)
	ids := seqIDs(threshold)

	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids, base)
	ix.lastTrainCount = 12345 // a sentinel that an OLD blob would NOT carry
	ix.lastTrainCost = 7.5

	// Forge a v3 (pre-drift) snapshot: magic + version=3 + scalars + arena + IVF core
	// (NO drift trailer) + PQ trailer.
	old := writeIVFV3Snapshot(t, ix)
	restored, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(old)); err != nil {
		t.Fatalf("Restore v3: %v", err)
	}
	if restored.lastTrainCount != 0 || restored.lastTrainCost != 0 {
		t.Fatalf("old-format snapshot must read drift checkpoint as 0: got count=%d cost=%v", restored.lastTrainCount, restored.lastTrainCost)
	}
	if !restored.trained {
		t.Fatalf("restored v3 index must be trained")
	}
}

// TestIVFDriftValidationFailLoud: a drift factor <= 1.0 is rejected at create.
func TestIVFDriftValidationFailLoud(t *testing.T) {
	dim := 8
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"growth==1.0", func(c *Config) { c.IVFDriftGrowthFactor = 1.0 }},
		{"growth<1.0", func(c *Config) { c.IVFDriftGrowthFactor = 0.5 }},
		{"drift==1.0", func(c *Config) { c.IVFDriftFactor = 1.0 }},
		{"drift<1.0", func(c *Config) { c.IVFDriftFactor = 0.9 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := ivfAutoTrainConfig(dim, 256)
			c.IVFDriftRetrain = true
			tc.mutate(&c)
			if _, err := newIVF(c); err == nil {
				t.Fatalf("expected create to fail for %s", tc.name)
			}
		})
	}
	// 0 (default) is allowed.
	ok := ivfAutoTrainConfig(dim, 256)
	ok.IVFDriftRetrain = true // factors left 0 → resolve to defaults
	if _, err := newIVF(ok); err != nil {
		t.Fatalf("zero drift factors (defaults) must be accepted: %v", err)
	}
}

// TestMeanAssignCostOrderPinned (M1): meanAssignCost must compute in canonical
// forward/slot order (index 0 → len-1). This test pins the expected value by
// computing an independent reference in the same order and asserting bit equality.
// A reversed or parallel reduction would differ due to float32 non-associativity
// and this test would go RED.
func TestMeanAssignCostOrderPinned(t *testing.T) {
	dim := 8
	ix, err := newIVF(ivfAutoTrainConfig(dim, 256))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(99))
	sample := clusteredVecs(rng, 40, dim, 4)
	centroids := clusteredVecs(rng, 5, dim, 8)

	// Compute the reference mean in the SAME forward order using the SAME distance
	// kernel (ix.metricDist()) to be bit-identical to what meanAssignCost must produce.
	dist := ix.metricDist()
	var refSum float32
	for i := 0; i < len(sample); i++ { // forward order — load-bearing
		best := dist(sample[i], centroids[0])
		for c := 1; c < len(centroids); c++ {
			if d := dist(sample[i], centroids[c]); d < best {
				best = d
			}
		}
		refSum += best
	}
	reference := refSum / float32(len(sample))

	got := ix.meanAssignCost(sample, centroids)
	if math.Float32bits(got) != math.Float32bits(reference) {
		t.Fatalf("meanAssignCost result %v (%x) != forward-order reference %v (%x) — iteration order diverged",
			got, math.Float32bits(got), reference, math.Float32bits(reference))
	}
}

// TestIVFDriftGrowthBoundary (M4): the growth checkpoint must fire at exactly
// floor(lastTrainCount * growthFactor) live vectors, not one before or after.
// This test inserts to the boundary-1, asserts no checkpoint, then inserts the
// boundary vector, asserts the checkpoint fires (retrain or defer).
func TestIVFDriftGrowthBoundary(t *testing.T) {
	dim := 16
	const threshold = 200
	const growth = 2.0
	cfg := ivfDriftConfig(dim, threshold, growth, 1.5)

	rng := rand.New(rand.NewSource(77))
	base := clusteredVecs(rng, threshold, dim, 4)
	ids := seqIDs(threshold)

	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids, base)
	if !ix.trained {
		t.Fatalf("must be trained at threshold %d", threshold)
	}
	firstTrainCount := ix.lastTrainCount // = threshold after initial train

	// The checkpoint fires at liveCount >= lastTrainCount * growthFactor.
	// With lastTrainCount=threshold and growth=2.0 that is threshold*2.
	// Insert up to (threshold*2 - 1): the boundary must NOT have fired yet.
	checkpointAt := int(float64(firstTrainCount) * growth) // = threshold * 2
	fillTo := checkpointAt - 1                             // one short of checkpoint
	extraNeeded := fillTo - threshold                      // already have `threshold` live
	if extraNeeded > 0 {
		extra := clusteredVecs(rng, extraNeeded, dim, 4)
		extraIDs := make([]uint64, extraNeeded)
		for i := range extraIDs {
			extraIDs[i] = uint64(threshold + 1 + i)
		}
		insertAll(t, ix, extraIDs, extra)
	}
	// At fillTo live vectors: lastTrainCount must be unchanged (no checkpoint yet).
	if ix.lastTrainCount != firstTrainCount {
		t.Fatalf("checkpoint fired early: lastTrainCount changed to %d at liveCount=%d (expected no change until %d)",
			ix.lastTrainCount, ix.liveCount(), checkpointAt)
	}

	// Insert the one boundary vector. The checkpoint must now fire (retrain or defer).
	boundaryVec := clusteredVecs(rng, 1, dim, 4)
	boundaryID := uint64(checkpointAt + 1)
	insertAll(t, ix, []uint64{boundaryID}, boundaryVec)
	if ix.lastTrainCount == firstTrainCount {
		t.Fatalf("checkpoint did NOT fire at liveCount=%d (expected at %d, growth=%.1f, firstTrainCount=%d)",
			ix.liveCount(), checkpointAt, growth, firstTrainCount)
	}
}

// TestIVFDriftTTLDeterminism (#2 TTL oracle): two replicas of a drift-ON IVF
// collection that both have a TTL'd point (wall-clock expiring) must produce
// BIT-IDENTICAL centroids after the identical applied insert/retrain sequence, even
// when two replicas have different frozen clocks that straddle an injected absolute
// TTL deadline — the train sample (liveSampleLocked) is tombstone-only (no
// isExpired/ix.now()), so wall-clock divergence cannot affect training.
//
// MUTATION GUARD: re-introducing TTL-awareness into liveSampleLocked (i.e. calling
// liveSlotLocked/isExpired) would make this test RED because replica A (clock=0) sees
// the TTL'd point live and replica B (clock=1000) sees it expired → different train
// samples → different centroids.
//
// The absolute deadline (500ms) is set via arena.SetExpires after the normal Insert
// so we bypass the relative-TTL arithmetic (deadline = now()+duration) that would
// make both replicas agree on liveness (see the comment inside buildReplica for
// why a relative 1ms TTL is insufficient as a mutation guard).
func TestIVFDriftTTLDeterminism(t *testing.T) {
	dim := 24
	const threshold = 400
	cfg := ivfDriftConfig(dim, threshold, 1.5, 1.2)

	rng := rand.New(rand.NewSource(88))
	base := clusteredVecs(rng, threshold, dim, 8)
	shift := driftedVecs(rng, 500, dim, 50)

	baseIDs := seqIDs(threshold)
	shiftIDs := make([]uint64, len(shift))
	for i := range shiftIDs {
		shiftIDs[i] = uint64(threshold + 1 + i)
	}
	ttlID := uint64(threshold + len(shift) + 100)

	// buildReplica constructs a replica with a frozen clock at `clockMs`. After
	// inserting the TTL'd point (with no TTL via the public API so deadline=0), we
	// directly set an ABSOLUTE deadline of 500ms via arena.SetExpires. With clockA=0
	// the point appears live (500 > 0 = false expired); with clockB=1000 it appears
	// expired (500 <= 1000 = true expired). If isExpired were consulted during
	// training the two replicas would train on different samples → different centroids.
	//
	// NOTE on why a relative TTL doesn't work as a mutation guard: Insert sets
	// deadline = now()+duration. For clock=0, ttl=1ms → deadline=1. For clock=1000,
	// ttl=1ms → deadline=1001. isExpired = deadline<=now → 1<=0 (false) and
	// 1001<=1000 (false) — BOTH see it live, the test cannot catch the regression.
	// The absolute deadline (set post-Insert) straddling the two clocks is the only
	// reliable mutation guard.
	buildReplica := func(clockMs int64) *ivf {
		ix, err := newIVF(cfg)
		if err != nil {
			t.Fatal(err)
		}
		ix.now = func() int64 { return clockMs }
		insertAll(t, ix, baseIDs, base)

		// Insert the TTL'd point WITHOUT a TTL (deadline=0) so the public deadline
		// arithmetic doesn't bind to the frozen clock.
		if _, _, err := ix.Insert(ttlID, base[0], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("TTL insert: %v", err)
		}
		// Directly set an ABSOLUTE deadline of 500ms so replica A (clock=0) sees the
		// point live and replica B (clock=1000) sees it expired at retrain time.
		slot, ok := ix.arena.Slot(ttlID)
		if !ok {
			t.Fatalf("ttlID slot not found after insert")
		}
		ix.arena.SetExpires(slot, 500) // absolute unix-ms deadline straddling 0 and 1000
		insertAll(t, ix, shiftIDs, shift)
		return ix
	}

	a := buildReplica(0)    // clock=0: deadline 500 > 0 → point appears LIVE
	b := buildReplica(1000) // clock=1000: deadline 500 <= 1000 → point appears EXPIRED

	if !a.trained || !b.trained {
		t.Fatalf("both replicas must be trained (a=%v b=%v)", a.trained, b.trained)
	}
	if len(a.centroids) != len(b.centroids) {
		t.Fatalf("nlist diverged: a=%d b=%d", len(a.centroids), len(b.centroids))
	}
	for c := range a.centroids {
		for d := range a.centroids[c] {
			if math.Float32bits(a.centroids[c][d]) != math.Float32bits(b.centroids[c][d]) {
				t.Fatalf("centroid[%d][%d] diverged under TTL clock skew (absolute deadline 500, clocks 0 vs 1000): "+
					"a=%x b=%x — train sample is NOT tombstone-only (isExpired leak)",
					c, d, math.Float32bits(a.centroids[c][d]), math.Float32bits(b.centroids[c][d]))
			}
		}
	}
	if a.lastTrainCount != b.lastTrainCount {
		t.Fatalf("lastTrainCount diverged: a=%d b=%d", a.lastTrainCount, b.lastTrainCount)
	}
	// Snapshot bytes must be bit-identical: if the train sample differed, centroids
	// would differ, and these bytes would diverge.
	if !bytes.Equal(snapshotBytes(t, a), snapshotBytes(t, b)) {
		t.Fatalf("replica snapshot bytes diverged under TTL clock skew — train sample is wall-clock-aware")
	}
}

// TestIVFDriftRestoreThenRetrain (#4): after a Snapshot+Restore of a drift-ON index,
// drift-retrain knobs are preserved in the restored cfg so a retrain still fires when
// the live count crosses the next growth checkpoint.
func TestIVFDriftRestoreThenRetrain(t *testing.T) {
	dim := 16
	const threshold = 200
	cfg := ivfDriftConfig(dim, threshold, 2.0, 1.2)

	rng := rand.New(rand.NewSource(55))
	base := clusteredVecs(rng, threshold, dim, 4)
	ids := seqIDs(threshold)

	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids, base)
	if !ix.trained {
		t.Fatalf("must be trained at threshold")
	}
	trainCountAfterFirstTrain := ix.lastTrainCount

	// Snapshot + Restore.
	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	restored, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	// Drift knobs must survive Restore.
	if !restored.cfg.IVFDriftRetrain {
		t.Fatalf("IVFDriftRetrain dropped by Restore")
	}

	// Insert a drifting cluster large enough to cross the 2x growth checkpoint AND
	// push the cost above the drift factor.
	shift := driftedVecs(rng, threshold*2, dim, 50)
	shiftIDs := make([]uint64, len(shift))
	for i := range shiftIDs {
		shiftIDs[i] = uint64(threshold + 1 + i)
	}
	insertAll(t, restored, shiftIDs, shift)

	// A retrain must have fired after Restore.
	if restored.lastTrainCount <= trainCountAfterFirstTrain {
		t.Fatalf("drift retrain did NOT fire after Restore: lastTrainCount=%d (first-train %d) — drift knobs were dropped",
			restored.lastTrainCount, trainCountAfterFirstTrain)
	}
}

// TestIVFDriftSnapshotVersionConditional (#3): a drift-ON index writes v4, a
// drift-OFF index writes v3. Both v4 and v3 snapshots restore correctly.
func TestIVFDriftSnapshotVersionConditional(t *testing.T) {
	dim := 16
	const threshold = 200
	rng := rand.New(rand.NewSource(66))
	base := clusteredVecs(rng, threshold, dim, 4)
	ids := seqIDs(threshold)

	readVersion := func(snap []byte) uint32 {
		const magicLen = 8
		if len(snap) < magicLen+4 {
			t.Fatalf("snapshot too short")
		}
		return uint32(snap[magicLen])<<24 | uint32(snap[magicLen+1])<<16 |
			uint32(snap[magicLen+2])<<8 | uint32(snap[magicLen+3])
	}

	// drift-ON → v4
	onCfg := ivfDriftConfig(dim, threshold, 2.0, 1.5)
	on, err := newIVF(onCfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, on, ids, base)
	snapOn := snapshotBytes(t, on)
	if v := readVersion(snapOn); v != 4 {
		t.Fatalf("drift-ON snapshot must be v4, got v%d", v)
	}
	// v4 must restore and preserve drift checkpoint.
	restoredOn, err := newIVF(onCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restoredOn.Restore(bytes.NewReader(snapOn)); err != nil {
		t.Fatalf("Restore v4: %v", err)
	}
	if restoredOn.lastTrainCount != on.lastTrainCount {
		t.Fatalf("v4 Restore dropped lastTrainCount: got %d want %d", restoredOn.lastTrainCount, on.lastTrainCount)
	}

	// drift-OFF → v3
	offCfg := ivfAutoTrainConfig(dim, threshold)
	off, err := newIVF(offCfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, off, ids, base)
	snapOff := snapshotBytes(t, off)
	if v := readVersion(snapOff); v != 3 {
		t.Fatalf("drift-OFF snapshot must be v3 (old-reader compatible), got v%d", v)
	}
	// v3 must restore (back-compat path, lastTrainCount=0).
	restoredOff, err := newIVF(offCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restoredOff.Restore(bytes.NewReader(snapOff)); err != nil {
		t.Fatalf("Restore v3 (off): %v", err)
	}
}

// --- test helpers -----------------------------------------------------------

// clusteredVecsFrom returns n vectors drawn from the SAME cluster centers that
// `existing` was clustered around (re-derives the blobs by sampling near random
// existing points), simulating stable-but-growing data.
func clusteredVecsFrom(rng *rand.Rand, n, dim int, existing [][]float32) [][]float32 {
	vs := make([][]float32, n)
	for i := range vs {
		anchor := existing[rng.Intn(len(existing))]
		v := make([]float32, dim)
		for d := range v {
			v[d] = anchor[d] + float32(rng.NormFloat64())*0.1 // tight around an existing point
		}
		vs[i] = v
	}
	return vs
}

// searchIDs runs a top-k search and returns the result ids in rank order.
func searchIDs(t *testing.T, ix *ivf, q []float32, k int) []uint64 {
	t.Helper()
	res, err := ix.Search(q, k)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// bruteForceTopK returns the ids of the k nearest (L2) vectors to q across base+shift,
// where ids are 1-based positions in the concatenated [base, shift] slice (matching
// the insert id scheme used by the tests).
func bruteForceTopK(base, shift [][]float32, q []float32, k int) []uint64 {
	type cand struct {
		id uint64
		d  float32
	}
	all := append(append([][]float32(nil), base...), shift...)
	cs := make([]cand, len(all))
	for i, v := range all {
		var s float32
		for d := range v {
			diff := v[d] - q[d]
			s += diff * diff
		}
		cs[i] = cand{id: uint64(i + 1), d: s}
	}
	// simple selection of k smallest
	for i := 0; i < k && i < len(cs); i++ {
		min := i
		for j := i + 1; j < len(cs); j++ {
			if cs[j].d < cs[min].d {
				min = j
			}
		}
		cs[i], cs[min] = cs[min], cs[i]
	}
	out := make([]uint64, 0, k)
	for i := 0; i < k && i < len(cs); i++ {
		out = append(out, cs[i].id)
	}
	return out
}

// recallAt returns |gt ∩ got| / |gt|.
func recallSet(gt, got []uint64) float64 {
	set := make(map[uint64]bool, len(gt))
	for _, id := range gt {
		set[id] = true
	}
	hit := 0
	for _, id := range got {
		if set[id] {
			hit++
		}
	}
	if len(gt) == 0 {
		return 1
	}
	return float64(hit) / float64(len(gt))
}

// writeIVFV3Snapshot serializes ix in the PRE-drift v3 format (no drift checkpoint in
// the IVF core block) so a back-compat restore can be exercised. Mirrors Snapshot but
// stamps version 3 and calls writeIVFCore(bw, false).
func writeIVFV3Snapshot(t *testing.T, ix *ivf) []byte {
	t.Helper()
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	var buf bytes.Buffer
	bw := bufio.NewWriter(&buf)
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := bw.WriteString(ivfSnapshotMagic); err != nil {
		t.Fatal(err)
	}
	must(writeU32(bw, 3)) // version 3 (pre-drift)
	must(writeU32(bw, uint32(ix.cfg.Dim)))
	must(bw.WriteByte(byte(ix.cfg.Metric)))
	must(writeU32(bw, uint32(ix.cfg.M)))
	must(writeU32(bw, uint32(ix.cfg.EfConstruction)))
	must(writeU32(bw, uint32(ix.cfg.EfSearch)))
	must(writeI64(bw, ix.cfg.Seed))
	must(ix.writeArena(bw, false))
	must(ix.writeIVFCore(bw, false, false)) // v3: NO drift trailer, NO SOAR block
	must(ix.writePQTrailer(bw))
	must(bw.Flush())
	return buf.Bytes()
}
