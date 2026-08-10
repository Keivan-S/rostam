// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These tests prove #4 Phase B / B3b: the logical-clock sweeper on a replicated
// shard reclaims expired pages DETERMINISTICALLY (against lastAppliedStampMs, never
// the wall clock), closing the B3a+B2 availability cliff where a near-full
// replicated shard crash-loops on cache.ErrFull under TTL churn. They exercise the
// cache layer directly: PutAt/GetAt are the stamped apply primitives (they advance
// the shard's logical clock), and the shard's sweepOnce is driven synchronously so
// the assertions are race-free.

// b3bReplicatedCache builds a single-shard replicated cache in reject-writes heap
// mode (the mode a cluster-replication shard is forced into by B2), with a small
// page budget so it fills quickly, and the background ticker OFF (sweepMs=0) so the
// test drives sweepOnce itself. clock0 seeds the injected wall clock; the apply
// path ignores it (that is the whole point), but skewing it across replicas proves
// the logical sweep is wall-clock-independent.
func b3bReplicatedCache(t *testing.T, clock0 uint64) *Cache {
	t.Helper()
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20          // 1 MiB pages
	cfg.MaxMemoryPerShard = 2 << 20 // 2 pages → fills fast, forces ErrFull
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.TTLSweepIntervalMs = 0 // ticker off; drive sweepOnce manually
	cfg.NowFn = func() uint64 { return clock0 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// fillWithTTL writes distinct ~200 KiB keys via stamped PutAt (stamp=stampMs,
// ttl→exp=stampMs+ttlMs) until the shard rejects with ErrFull, returning the keys
// written. Every entry therefore shares the same absolute expiry.
func fillWithTTL(t *testing.T, c *Cache, stampMs, ttlMs uint64) [][]byte {
	t.Helper()
	val := bytes.Repeat([]byte("x"), 200<<10) // 200 KiB
	ttl := time.Duration(ttlMs) * time.Millisecond
	var keys [][]byte
	for i := 0; i < 10_000; i++ {
		k := []byte(fmt.Sprintf("k%06d", i))
		err := c.PutAt(k, val, ttl, stampMs)
		if err == ErrFull {
			return keys
		}
		if err != nil {
			t.Fatalf("PutAt #%d: %v", i, err)
		}
		keys = append(keys, k)
	}
	t.Fatal("shard never filled to ErrFull — page budget too large for the test")
	return nil
}

// advanceLogicalClock folds `stamp` into every shard's logical clock, simulating a
// stamped apply landing at that stamp (getAtH/putAtH would do this in production).
func advanceLogicalClock(c *Cache, stamp uint64) {
	for _, s := range c.shards {
		s.advanceAppliedStamp(stamp)
	}
}

// TestB3bLogicalSweepReclaimsExpiredPages is the core reclamation proof: once the
// logical clock advances past the entries' expiry, the sweeper physically frees the
// pages and a write that previously hit ErrFull now succeeds. Before the clock
// advances (stamp still below exp) nothing is reclaimed — the contrast that proves
// the sweep keys off the logical clock, not merely "run the sweeper".
func TestB3bLogicalSweepReclaimsExpiredPages(t *testing.T) {
	const S, ttlMs = uint64(1_000_000), uint64(1_000)
	c := b3bReplicatedCache(t, 42) // wall clock 42 — irrelevant to the apply path
	keys := fillWithTTL(t, c, S, ttlMs)
	exp := S + ttlMs // every entry expires here

	// The overflow probe is a full-page-sized value: it needs a fresh page, so it is
	// rejected iff no page has real capacity (a tiny value could sneak into leftover
	// tail room and mask a "full" shard).
	bigVal := bytes.Repeat([]byte("y"), 200<<10)

	// The shard is full: another full-size stamped write at the SAME stamp is rejected.
	if err := c.PutAt([]byte("overflow"), bigVal, time.Second, S); err != ErrFull {
		t.Fatalf("pre-sweep PutAt: err=%v, want ErrFull (shard should be full)", err)
	}

	// CONTRAST — sweep with the logical clock still at S (< exp): nothing is
	// reclaimable, so the write stays rejected and no expirations are recorded.
	c.shards[0].sweepOnce()
	if got := c.Stats().Expirations; got != 0 {
		t.Fatalf("sweep with lastAppliedStampMs=%d (< exp=%d) reclaimed %d entries, want 0", S, exp, got)
	}
	if err := c.PutAt([]byte("overflow"), bigVal, time.Second, S); err != ErrFull {
		t.Fatalf("PutAt after no-op sweep: err=%v, want ErrFull", err)
	}

	// Advance the logical clock past exp with a stamped read (GetAt folds the stamp
	// into lastAppliedStampMs), then sweep: every page is now fully expired and gets
	// retired.
	if _, err := c.GetAt(keys[0], exp+5_000); err != ErrNotFound {
		t.Fatalf("GetAt of an expired key: err=%v, want ErrNotFound (filtered)", err)
	}
	if got := c.LastAppliedStampMs(); got < exp {
		t.Fatalf("logical clock = %d, want >= exp=%d after the advancing GetAt", got, exp)
	}
	c.shards[0].sweepOnce()

	if got := c.Stats().Expirations; got == 0 {
		t.Fatal("logical sweep past exp reclaimed nothing — the availability cliff is not closed")
	}
	// The decisive assertion: a full-size write that was ErrFull now succeeds,
	// because the sweeper physically freed page capacity.
	if err := c.PutAt([]byte("overflow"), bigVal, time.Second, exp+5_000); err != nil {
		t.Fatalf("post-sweep PutAt: err=%v, want success (reclamation must have freed a page)", err)
	}
	_ = keys
}

// TestB3bStampZeroSweepIsNoOp proves the lastAppliedStampMs==0 guard: with no
// stamped apply yet (entries installed via PutAbs, which does NOT advance the
// logical clock), a replicated sweep reclaims nothing and removes nothing — no
// crash, no wrong removal — even though the entries are long past their absolute
// expiry by any wall clock.
func TestB3bStampZeroSweepIsNoOp(t *testing.T) {
	c := b3bReplicatedCache(t, 9_999_999) // high wall clock, past every exp below
	// PutAbs installs an absolute expiry WITHOUT advancing lastAppliedStampMs.
	for i := 0; i < 5; i++ {
		k := []byte(fmt.Sprintf("abs%d", i))
		if err := c.PutAbs(k, []byte("v"), 1_000); err != nil { // exp=1000, long past
			t.Fatal(err)
		}
	}
	if got := c.LastAppliedStampMs(); got != 0 {
		t.Fatalf("lastAppliedStampMs=%d after PutAbs, want 0 (PutAbs must not advance the logical clock)", got)
	}

	c.shards[0].sweepOnce()

	if got := c.Stats().Expirations; got != 0 {
		t.Fatalf("stamp-0 replicated sweep reclaimed %d entries, want 0 (no logical clock ⇒ no-op)", got)
	}
	// The entries are still physically present (GetAt at a pinned-low clock hits).
	for i := 0; i < 5; i++ {
		k := []byte(fmt.Sprintf("abs%d", i))
		if _, err := c.GetAt(k, 1); err != nil {
			t.Fatalf("entry %s vanished after a no-op stamp-0 sweep: %v", k, err)
		}
	}
}

// TestB3bCrossReplicaDeterministicReclamation proves determinism: two replicas
// applying the IDENTICAL stamped sequence reclaim the SAME keys, despite wildly
// skewed wall clocks — because the sweep keys off the logical clock, which is
// identical on both. The live set (keys physically retained) and the expiration
// count must match exactly.
func TestB3bCrossReplicaDeterministicReclamation(t *testing.T) {
	const S, ttlShort, ttlLong = uint64(2_000_000), uint64(1_000), uint64(1_000_000)

	// Apply the same stamped writes on both replicas: some short-TTL keys (will be
	// swept), some long-TTL keys (must survive). Wall clocks are seconds/minutes
	// apart to prove they are irrelevant.
	apply := func(clock0 uint64) *Cache {
		c := b3bReplicatedCache(t, clock0)
		for i := 0; i < 3; i++ {
			short := []byte(fmt.Sprintf("short%d", i))
			long := []byte(fmt.Sprintf("long%d", i))
			if err := c.PutAt(short, []byte("s"), time.Duration(ttlShort)*time.Millisecond, S); err != nil {
				t.Fatal(err)
			}
			if err := c.PutAt(long, []byte("l"), time.Duration(ttlLong)*time.Millisecond, S); err != nil {
				t.Fatal(err)
			}
		}
		// Advance the logical clock to a point past the short TTL but before the long
		// one, identically on both, then sweep.
		advanceLogicalClock(c, S+5_000)
		c.shards[0].sweepOnce()
		return c
	}

	rA := apply(1_000)         // wall clock ~1s
	rB := apply(9_000_000_000) // wall clock ~104 days later

	if a, b := rA.Stats().Expirations, rB.Stats().Expirations; a != b {
		t.Fatalf("cross-replica expirations differ: A=%d B=%d — reclamation diverged under wall-clock skew", a, b)
	}
	// Every short key must be gone on BOTH; every long key must survive on BOTH.
	// GetAt at a pinned-low clock (1) reports physical presence only.
	for i := 0; i < 3; i++ {
		short := []byte(fmt.Sprintf("short%d", i))
		long := []byte(fmt.Sprintf("long%d", i))
		_, aShort := rA.GetAt(short, 1)
		_, bShort := rB.GetAt(short, 1)
		if aShort != bShort {
			t.Fatalf("short key %s: A err=%v B err=%v — physical reclamation diverged", short, aShort, bShort)
		}
		if aShort != ErrNotFound {
			t.Fatalf("short key %s should have been reclaimed on both replicas, got A=%v", short, aShort)
		}
		_, aLong := rA.GetAt(long, 1)
		_, bLong := rB.GetAt(long, 1)
		if aLong != nil || bLong != nil {
			t.Fatalf("long key %s must survive on both replicas: A=%v B=%v", long, aLong, bLong)
		}
	}
}

// TestB3bSweeperVsLaterWriteSafety proves the sweeper-vs-write invariant at the
// cache layer: once the sweeper removes K (exp <= lastAppliedStampMs), no later
// stamped access at a stamp >= the logical clock ever sees K live again — on any
// replica. (The leader-side monotonic clamp that guarantees no FUTURE write can
// carry a regressing stamp is proven at the store layer, see
// TestB3bLeaderStampMonotonicClamp.)
func TestB3bSweeperVsLaterWriteSafety(t *testing.T) {
	const S, ttlMs = uint64(3_000_000), uint64(1_000)
	exp := S + ttlMs

	check := func(clock0 uint64) {
		c := b3bReplicatedCache(t, clock0)
		if err := c.PutAt([]byte("K"), []byte("v"), time.Duration(ttlMs)*time.Millisecond, S); err != nil {
			t.Fatal(err)
		}
		// Advance the logical clock past exp and sweep K away.
		advanceLogicalClock(c, exp+1_000)
		c.shards[0].sweepOnce()

		// A later committed write W stamped at/after the logical clock reads K: it
		// must see K as absent/expired on every replica — never resurrect it.
		if _, err := c.GetAt([]byte("K"), exp+1_000); err != ErrNotFound {
			t.Fatalf("clock0=%d: post-sweep GetAt(K, %d) = %v, want ErrNotFound (K was reclaimed)", clock0, exp+1_000, err)
		}
		// And it is physically gone (pinned-low read also misses).
		if _, err := c.GetAt([]byte("K"), 1); err != ErrNotFound {
			t.Fatalf("clock0=%d: K still physically present after sweep", clock0)
		}
	}
	check(5)
	check(8_000_000_000) // skewed replica — identical outcome
}

// TestB3bAdvanceAppliedStampMonotonic proves the shard's logical clock never
// regresses: a lower stamp after a higher one is a no-op. This is the cache-side
// half of the monotonicity the leader clamp relies on.
func TestB3bAdvanceAppliedStampMonotonic(t *testing.T) {
	c := b3bReplicatedCache(t, 1)
	if err := c.PutAt([]byte("a"), []byte("v"), 0, 5_000); err != nil {
		t.Fatal(err)
	}
	if got := c.LastAppliedStampMs(); got != 5_000 {
		t.Fatalf("logical clock = %d, want 5000", got)
	}
	// A lower stamp must NOT pull the clock backward.
	if err := c.PutAt([]byte("b"), []byte("v"), 0, 3_000); err != nil {
		t.Fatal(err)
	}
	if got := c.LastAppliedStampMs(); got != 5_000 {
		t.Fatalf("logical clock regressed to %d after a lower stamp, want 5000 (must be monotonic)", got)
	}
	// A higher stamp advances it.
	if _, err := c.GetAt([]byte("a"), 7_000); err != nil && err != ErrNotFound {
		t.Fatal(err)
	}
	if got := c.LastAppliedStampMs(); got != 7_000 {
		t.Fatalf("logical clock = %d after stamp 7000, want 7000", got)
	}
}

// TestB3bSingleNodeWallClockSweeperUnchanged proves point 5: a non-replicated cache
// still runs the WALL-CLOCK sweeper (against s.now()) exactly as before — the
// logical clock plays no role, and the sweeper reclaims once the injected wall clock
// passes exp.
func TestB3bSingleNodeWallClockSweeperUnchanged(t *testing.T) {
	clk := uint64(1_000)
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0 // drive sweepOnce manually
	cfg.Replicated = false     // single-node / Direct
	cfg.NowFn = func() uint64 { return clk }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	// Wall-clock Put: exp = now(1000) + 1000 = 2000.
	if err := c.Put([]byte("k"), []byte("v"), time.Second); err != nil {
		t.Fatal(err)
	}
	// The logical clock stays 0 on a non-replicated cache — it is never consulted.
	if got := c.LastAppliedStampMs(); got != 0 {
		t.Fatalf("non-replicated logical clock = %d, want 0 (must be irrelevant here)", got)
	}
	// Sweep before the wall clock passes exp: nothing reclaimed.
	c.shards[0].sweepOnce()
	if got := c.Stats().Expirations; got != 0 {
		t.Fatalf("wall-clock sweep at now=1000 (< exp=2000) reclaimed %d, want 0", got)
	}
	// Advance the wall clock past exp and sweep: the entry is reclaimed.
	clk = 3_000
	c.shards[0].sweepOnce()
	if got := c.Stats().Expirations; got == 0 {
		t.Fatal("wall-clock sweep at now=3000 (> exp=2000) reclaimed nothing — single-node sweeper regressed")
	}
}

// TestB3bMmapReplicatedReclaimsIndexOnly covers the PRODUCTION (persistent) path,
// which the heap tests do not: a replicated MMAP shard (DataDir set ⇒ s.region !=
// nil). The logical sweep must reclaim INDEX SLOTS deterministically (expired keys
// become misses and are counted as expirations) but must NOT free page BYTES —
// mmap pages cannot be retired by frozen-swap, so a would-be-ErrFull write STILL
// fails after the sweep. This documents the mmap limitation behaviorally so the
// "MITIGATED, not closed" claim in the docs is backed by a test.
func TestB3bMmapReplicatedReclaimsIndexOnly(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	const S, ttlMs = uint64(1_000_000), uint64(1_000)
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 2 << 20 // 2 mmap pages, both preallocated
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir() // ⇒ mmap mode (s.region != nil)
	cfg.NowFn = func() uint64 { return 42 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	if c.shards[0].region == nil {
		t.Fatal("expected mmap mode (region != nil) with DataDir set")
	}

	keys := fillWithTTL(t, c, S, ttlMs)
	exp := S + ttlMs
	bigVal := bytes.Repeat([]byte("y"), 200<<10)

	// Full: a full-size stamped write is rejected.
	if err := c.PutAt([]byte("overflow"), bigVal, time.Second, S); err != ErrFull {
		t.Fatalf("pre-sweep PutAt: err=%v, want ErrFull", err)
	}

	// Advance the logical clock past exp and sweep.
	advanceLogicalClock(c, exp+5_000)
	c.shards[0].sweepOnce()

	// INDEX SLOTS were reclaimed: the sweep counted expirations, and the keys now
	// read as misses even at a pinned-low clock (physically dropped from the index).
	if got := c.Stats().Expirations; got == 0 {
		t.Fatal("mmap logical sweep reclaimed no index slots — index reclamation must still work on mmap")
	}
	if _, gErr := c.GetAt(keys[0], 1); gErr != ErrNotFound {
		t.Fatalf("mmap post-sweep GetAt(k,1) = %v, want ErrNotFound (index slot must be reclaimed)", gErr)
	}

	// But page BYTES were NOT freed: mmap pages can't be retired, so a full-size
	// write STILL fails. This is the documented mmap limitation.
	if err := c.PutAt([]byte("overflow2"), bigVal, time.Second, exp+5_000); err != ErrFull {
		t.Fatalf("mmap post-sweep full-size PutAt: err=%v, want ErrFull "+
			"(mmap must NOT free page bytes — page reclamation is heap-only)", err)
	}
	_ = keys
}

// TestB3bRetireReaderRaceConcurrent exercises the safety the whole frozen-page
// retirement argument rests on: a lock-free reject-writes reader running Get
// CONCURRENTLY with the sweeper retiring pages. Under `go test -race` this is the
// only test that actually drives the sweeper-vs-reader path (the others sweep
// synchronously with no concurrent reader). It asserts no torn/stale read (a hit
// always returns the exact written value) and no panic / use-after-free — the frozen
// old page object is immutable, and the fresh object is published atomically via
// pageSlots.Store, so a reader either reads the old immutable bytes or misses on the
// new generation.
func TestB3bRetireReaderRaceConcurrent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 8 << 20 // 8 heap pages of churn room
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.TTLSweepIntervalMs = 0
	// Wall clock pinned LOW so the client-read filter never itself expires a key —
	// we want reads to observe live values while the sweeper retires around them.
	cfg.NowFn = func() uint64 { return 1 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	const nKeys = 64
	valFor := func(i int) []byte { return bytes.Repeat([]byte{byte('a' + i%26)}, 8<<10) } // 8 KiB
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("rk%04d", i)) }

	var stop atomic.Bool
	var wg sync.WaitGroup

	// Readers: hammer Get on the key set. A hit MUST carry the exact value (no torn
	// read); a miss (key retired / not yet written) is fine.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				for i := 0; i < nKeys; i++ {
					v, gErr := c.Get(keyFor(i))
					if gErr == nil && !bytes.Equal(v, valFor(i)) {
						t.Errorf("torn read: key %d value mismatch (len got=%d want=%d)", i, len(v), len(valFor(i)))
						return
					}
				}
			}
		}()
	}

	// Writer+sweeper: keep (re)writing keys with a short logical TTL and retiring the
	// pages behind them, driving constant page churn under the readers.
	wg.Add(1)
	go func() {
		defer wg.Done()
		stamp := uint64(1_000)
		for iter := 0; iter < 300; iter++ {
			for i := 0; i < nKeys; i++ {
				// exp = stamp + 10; reject-writes ErrFull is expected under churn and
				// ignored — the sweeper frees pages the next round.
				_ = c.PutAt(keyFor(i), valFor(i), 10*time.Millisecond, stamp)
			}
			stamp += 1_000 // advance the logical clock past the just-written TTLs
			advanceLogicalClock(c, stamp)
			c.shards[0].sweepOnce() // retire the now-expired pages, racing the readers
		}
		stop.Store(true)
	}()

	wg.Wait()
	if t.Failed() {
		t.Fatal("concurrent reader observed a torn/stale read during page retirement")
	}
}

// TestB3bReclaimPerPageDecisionCorrectness proves the per-page retire DECISION is
// correct — a single live (non-expired, index-current) entry pins its whole page —
// and that correctness is unaffected by the per-page lock release (MAJOR #2's
// batching). It also runs a concurrent apply during the sweep to show a write can
// interleave (the lock is not held across the whole scan) without corrupting the
// outcome. One ~full-page key per page, alternating short/long TTL: only the
// short-TTL (fully-expired) pages may be retired; long-TTL keys must survive.
func TestB3bReclaimPerPageDecisionCorrectness(t *testing.T) {
	const S, ttlShort, ttlLong = uint64(2_000_000), uint64(1_000), uint64(10_000_000)
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 16 << 20 // room for many one-key pages
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.TTLSweepIntervalMs = 0
	cfg.NowFn = func() uint64 { return 1 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	// ~600 KiB values ⇒ one key per 1 MiB page (a second won't fit the tail).
	bigVal := bytes.Repeat([]byte("z"), 600<<10)
	const nPairs = 6
	var shortKeys, longKeys [][]byte
	for i := 0; i < nPairs; i++ {
		sk := []byte(fmt.Sprintf("short%02d", i))
		lk := []byte(fmt.Sprintf("long%02d", i))
		if err := c.PutAt(sk, bigVal, time.Duration(ttlShort)*time.Millisecond, S); err != nil {
			t.Fatalf("put short %d: %v", i, err)
		}
		if err := c.PutAt(lk, bigVal, time.Duration(ttlLong)*time.Millisecond, S); err != nil {
			t.Fatalf("put long %d: %v", i, err)
		}
		shortKeys = append(shortKeys, sk)
		longKeys = append(longKeys, lk)
	}

	// Advance the logical clock past the short TTL but before the long one, then run
	// the sweep while a concurrent apply interleaves (proving the reclaim scan does
	// not hold mu across the whole thing).
	advanceLogicalClock(c, S+5_000)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// A committed write landing during the sweep. It may ErrFull under pressure;
		// either way it must not deadlock or corrupt the sweep's decision.
		_ = c.PutAt([]byte("interleaved"), []byte("v"), 0, S+5_000)
	}()
	c.shards[0].sweepOnce()
	wg.Wait()

	// Every short (fully-expired) key must be gone; every long key must survive.
	for i := 0; i < nPairs; i++ {
		if _, gErr := c.GetAt(shortKeys[i], 1); gErr != ErrNotFound {
			t.Fatalf("short key %d survived the sweep: err=%v, want reclaimed", i, gErr)
		}
		if _, gErr := c.GetAt(longKeys[i], 1); gErr != nil {
			t.Fatalf("long key %d was reclaimed: err=%v, want it PINNED (non-expired) and retained", i, gErr)
		}
	}
}

// TestMmapOccupancyHighWaterWarns covers the #4 Option 3 observability step: a
// replicated MMAP shard cannot physically reclaim page bytes WHILE RUNNING (the
// reclaim pass early-returns for mmap — a file-backed page can't be
// frozen-swapped out from under a lock-free reader), so ghost bytes climb toward
// ErrFull between restarts. The sweep must latch a high-water WARNING before
// that halt, and re-arm once occupancy recovers. Heap shards (which DO reclaim
// bytes online) must never latch. The warning's remedy — cold compaction at the
// next shard open — is covered in cache/compact_test.go.
func TestMmapOccupancyHighWaterWarns(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	const S, ttlMs = uint64(1_000_000), uint64(1_000)
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 2 << 20 // 2 mmap pages
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir() // ⇒ mmap mode
	cfg.NowFn = func() uint64 { return 42 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	s := c.shards[0]
	if s.region == nil {
		t.Fatal("expected mmap mode (region != nil)")
	}

	// Empty shard: occupancy ~0, no warning latched.
	s.checkMmapOccupancy()
	if s.mmapHighWaterWarned.Load() {
		t.Fatal("empty mmap shard latched the high-water warning")
	}

	// Fill past the high-water with TTL'd entries (they become ghost bytes).
	fillWithTTL(t, c, S, ttlMs)
	if got := s.occupancyRatio(); got < mmapOccupancyWarnHigh {
		t.Fatalf("occupancy after fill = %.3f, want >= %.2f (test setup must fill past the high-water)", got, mmapOccupancyWarnHigh)
	}

	// The sweep (past expiry) reclaims index slots but NOT page bytes on mmap, so
	// occupancy stays high and the warning latches exactly once.
	advanceLogicalClock(c, S+ttlMs+5_000)
	s.sweepOnce()
	if !s.mmapHighWaterWarned.Load() {
		t.Fatal("mmap shard over the high-water did NOT latch the occupancy warning — the ErrFull cliff would arrive silently")
	}
	// Latched: a second sweep must not re-warn (rising-edge throttle). The latch
	// staying true across the call is the observable proof.
	s.sweepOnce()
	if !s.mmapHighWaterWarned.Load() {
		t.Fatal("warning latch cleared while still over the high-water")
	}

	// Hysteresis: a shard whose occupancy is below the low-water re-arms the latch.
	// bytesUsed can't drop on this shard (mmap frees no bytes), so exercise the
	// re-arm branch on a fresh empty mmap shard (occupancy ~0 < low-water).
	freshCfg := cfg
	freshCfg.DataDir = t.TempDir()
	fresh, ferr := New(freshCfg)
	if ferr != nil {
		t.Fatal(ferr)
	}
	t.Cleanup(func() { _ = fresh.Close() })
	fs := fresh.shards[0]
	fs.mmapHighWaterWarned.Store(true) // pretend it warned earlier
	fs.checkMmapOccupancy()
	if fs.mmapHighWaterWarned.Load() {
		t.Fatalf("occupancy %.3f below the low-water %.2f did not re-arm the warning latch",
			fs.occupancyRatio(), mmapOccupancyWarnLow)
	}
}

// TestHeapReplicatedNeverLatchesOccupancyWarning: heap shards DO reclaim page
// bytes, so the mmap-only alert must never fire for them (region == nil).
func TestHeapReplicatedNeverLatchesOccupancyWarning(t *testing.T) {
	const S, ttlMs = uint64(1_000_000), uint64(1_000)
	c := b3bReplicatedCache(t, 42) // heap (no DataDir)
	s := c.shards[0]
	fillWithTTL(t, c, S, ttlMs)
	advanceLogicalClock(c, S+ttlMs+5_000)
	s.sweepOnce()
	if s.mmapHighWaterWarned.Load() {
		t.Fatal("heap replicated shard latched the mmap occupancy warning; the alert is mmap-only")
	}
}
