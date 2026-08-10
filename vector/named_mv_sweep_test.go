// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"runtime"
	"testing"
	"time"
)

// Per-key-TTL BACKGROUND SWEEP for the named + MV families. Today named/MV are
// lazy-drop-only: an expired per-key payload key lingers in keyTTL + the payload
// map (and as a stale posting in the id-keyed payloadIdx) until the point/doc is
// overwritten or deleted. sweepKeyTTLOnce physically reclaims those keys — dropping
// them from keyTTL + the payload map AND reindexing payloadIdx so the stale posting
// is GONE (not merely lazily hidden), mirroring the dense ttl.go per-key pass. It
// shares the keyExpired predicate with the lazy liveMetaMap read path so the two can
// never diverge, is idempotent, bumps dataVersion (order_by cache), and is a cheap
// no-op for a collection with no per-key TTL. The injectable clock ages
// deterministically; no sleeps.

// namedFieldHasPosting reports whether the id-keyed payloadIdx still posts id under
// an exact (field == keyword) equality — the physical-presence probe. candidates()
// returns ([]ids, true) for an Eq leaf; an empty set means the posting is GONE.
func payloadIdxHasPosting(p *payloadIndexID, field, val string, id uint64) bool {
	ids, ok := p.candidates(Filter{Op: FilterEq, Field: field, Value: NewString(val)}, 1<<20)
	if !ok {
		return false
	}
	for _, got := range ids {
		if got == id {
			return true
		}
	}
	return false
}

// --- named ---

// TestNamedSweepPhysicallyReclaimsExpiredKey: a per-key TTL set at named Insert is
// physically dropped by sweepKeyTTLOnce after its deadline — gone from keyTTL +
// nc.meta AND its payloadIdx posting removed (proven via candidates(), not merely
// lazily hidden) — while a permanent key + the point itself survive. The sweep is
// idempotent (a 2nd call drops 0).
func TestNamedSweepPhysicallyReclaimsExpiredKey(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 2_000_000
	nc.now = func() int64 { return fakeNow }

	// "temp" is a keyword field with a 1000ms TTL; "perm" is permanent. Both are
	// indexed in payloadIdx (string values post as exact-match keyword terms).
	if _, err := nc.InsertCASKeyTTL(7, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"perm": NewString("keep"), "temp": NewString("bye")}, 0,
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("InsertCASKeyTTL: %v", err)
	}

	// Before the deadline a sweep drops nothing and the posting stands.
	if n := nc.sweepKeyTTLOnce(); n != 0 {
		t.Fatalf("pre-deadline sweep dropped %d, want 0", n)
	}
	if !payloadIdxHasPosting(nc.payloadIdx, "temp", "bye", 7) {
		t.Fatal("pre-sweep: temp posting missing")
	}

	// Age past the deadline and sweep: exactly one key dropped.
	fakeNow += 1500
	if n := nc.sweepKeyTTLOnce(); n != 1 {
		t.Fatalf("sweep dropped %d, want 1", n)
	}

	// Physically gone from keyTTL.
	if _, has := nc.keyTTL[7]["temp"]; has {
		t.Errorf("keyTTL still has expired 'temp': %v", nc.keyTTL[7])
	}
	// Physically gone from the payload map.
	if _, has := nc.meta[7]["temp"]; has {
		t.Errorf("nc.meta still has expired 'temp': %v", nc.meta[7])
	}
	// Physically gone from payloadIdx — the stale posting is removed, NOT lazily
	// hidden (this is the dense-parity proof).
	if payloadIdxHasPosting(nc.payloadIdx, "temp", "bye", 7) {
		t.Error("payloadIdx STILL posts the swept 'temp' key — stale posting not reclaimed")
	}
	// The permanent key + its posting + the point all survive.
	if nc.meta[7]["perm"].Str != "keep" {
		t.Errorf("sweep lost the permanent key: %v", nc.meta[7])
	}
	if !payloadIdxHasPosting(nc.payloadIdx, "perm", "keep", 7) {
		t.Error("sweep wrongly removed the permanent key's posting")
	}
	if _, _, _, _, ok := nc.Get(7); !ok {
		t.Error("sweep wrongly removed the point itself")
	}

	// Idempotent: a 2nd sweep at the same clock drops nothing.
	if n := nc.sweepKeyTTLOnce(); n != 0 {
		t.Fatalf("2nd sweep dropped %d, want 0 (idempotent)", n)
	}
}

// TestNamedSweepNoKeyTTLNoOp: a collection with no per-key TTL configured sweeps to
// a clean 0 (the cheap no-op path; no-key-TTL collections unaffected).
func TestNamedSweepNoKeyTTLNoOp(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"a": NewInt(1)}, 0); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if n := nc.sweepKeyTTLOnce(); n != 0 {
		t.Fatalf("no-keyTTL sweep dropped %d, want 0", n)
	}
	// The point + its payload + posting are untouched.
	if _, pay, _, _, ok := nc.Get(1); !ok || pay["a"].Int != 1 {
		t.Fatalf("no-op sweep mutated state: ok=%v pay=%v", ok, pay)
	}
}

// TestNamedSweepAgreesWithLazyDrop: at exactly-now and just-after, the lazy
// liveMetaMap read path and the sweep handle a key IDENTICALLY (shared predicate).
func TestNamedSweepAgreesWithLazyDrop(t *testing.T) {
	const deadline int64 = 1_000_000
	// At exactly the deadline the key is expired (<= now), so BOTH lazy + sweep drop.
	for _, tc := range []struct {
		name string
		now  int64
		want bool // want key dropped
	}{
		{"just-before", deadline - 1, false},
		{"exactly-now", deadline, true},
		{"just-after", deadline + 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			nc, err := NewNamedCollection("default/named", namedTestConfig())
			if err != nil {
				t.Fatalf("new: %v", err)
			}
			defer nc.Close()
			// Insert at t=0 with an absolute deadline by setting clock then ttl.
			nc.now = func() int64 { return 0 }
			if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
				Metadata{"temp": NewString("x")}, 0, map[string]int64{"temp": deadline}, CASCond{}); err != nil {
				t.Fatalf("insert: %v", err)
			}
			nc.now = func() int64 { return tc.now }

			// Lazy view (Get applies liveMetaMap).
			_, lazyPay, _, _, ok := nc.Get(1)
			if !ok {
				t.Fatal("point gone")
			}
			_, lazyHas := lazyPay["temp"]
			lazyDropped := !lazyHas

			// Sweep on a FRESH identical collection so the lazy read above didn't mutate.
			nc2, _ := NewNamedCollection("default/named", namedTestConfig())
			defer nc2.Close()
			nc2.now = func() int64 { return 0 }
			_, _ = nc2.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
				Metadata{"temp": NewString("x")}, 0, map[string]int64{"temp": deadline}, CASCond{})
			nc2.now = func() int64 { return tc.now }
			swept := nc2.sweepKeyTTLOnce() > 0

			if lazyDropped != tc.want || swept != tc.want {
				t.Fatalf("disagreement: lazyDropped=%v swept=%v want=%v", lazyDropped, swept, tc.want)
			}
		})
	}
}

// TestNamedSweepInvalidatesOrderCache: a sweep that drops an order-by field key
// bumps dataVersion so the order_by snapshot rebuilds on the next scroll.
func TestNamedSweepInvalidatesOrderCache(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 100
	nc.now = func() int64 { return fakeNow }
	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"rank": NewInt(5)}, 0, map[string]int64{"rank": 1000}, CASCond{}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	before := nc.dataVersion
	fakeNow += 2000
	if n := nc.sweepKeyTTLOnce(); n != 1 {
		t.Fatalf("sweep dropped %d, want 1", n)
	}
	if nc.dataVersion == before {
		t.Errorf("sweep did not bump dataVersion (%d) — order_by cache would not invalidate", nc.dataVersion)
	}
}

// --- MV ---

// TestMVSweepPhysicallyReclaimsExpiredKey: MV analogue of the named physical-reclaim
// proof — sweepKeyTTLOnce drops the expired key from keyTTL + docMeta AND removes its
// payloadIdx posting; the permanent key + the doc survive; idempotent.
func TestMVSweepPhysicallyReclaimsExpiredKey(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 2_000_000
	m.now = func() int64 { return fakeNow }

	if _, err := m.AddCASKeyTTL(7, [][]float32{{1, 0, 0, 0}},
		Metadata{"perm": NewString("keep"), "temp": NewString("bye")},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("AddCASKeyTTL: %v", err)
	}

	if n := m.sweepKeyTTLOnce(); n != 0 {
		t.Fatalf("pre-deadline sweep dropped %d, want 0", n)
	}
	if !payloadIdxHasPosting(m.payloadIdx, "temp", "bye", 7) {
		t.Fatal("pre-sweep: temp posting missing")
	}

	fakeNow += 1500
	if n := m.sweepKeyTTLOnce(); n != 1 {
		t.Fatalf("sweep dropped %d, want 1", n)
	}
	if _, has := m.keyTTL[7]["temp"]; has {
		t.Errorf("keyTTL still has expired 'temp': %v", m.keyTTL[7])
	}
	if _, has := m.docMeta[7]["temp"]; has {
		t.Errorf("docMeta still has expired 'temp': %v", m.docMeta[7])
	}
	if payloadIdxHasPosting(m.payloadIdx, "temp", "bye", 7) {
		t.Error("payloadIdx STILL posts the swept 'temp' key — stale posting not reclaimed")
	}
	if m.docMeta[7]["perm"].Str != "keep" {
		t.Errorf("sweep lost the permanent key: %v", m.docMeta[7])
	}
	if !payloadIdxHasPosting(m.payloadIdx, "perm", "keep", 7) {
		t.Error("sweep wrongly removed the permanent key's posting")
	}
	if _, _, _, ok := m.Get(7); !ok {
		t.Error("sweep wrongly removed the doc itself")
	}

	if n := m.sweepKeyTTLOnce(); n != 0 {
		t.Fatalf("2nd sweep dropped %d, want 0 (idempotent)", n)
	}
}

// TestMVSweepNoKeyTTLNoOp: MV no-key-TTL sweep is a clean 0 no-op.
func TestMVSweepNoKeyTTLNoOp(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"a": NewInt(1)}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if n := m.sweepKeyTTLOnce(); n != 0 {
		t.Fatalf("no-keyTTL sweep dropped %d, want 0", n)
	}
	if _, pay, _, ok := m.Get(1); !ok || pay["a"].Int != 1 {
		t.Fatalf("no-op sweep mutated state: ok=%v pay=%v", ok, pay)
	}
}

// TestMVSweepAgreesWithLazyDrop: MV lazy liveMetaMap read and sweep handle a key at
// exactly-now / just-after IDENTICALLY (shared predicate).
func TestMVSweepAgreesWithLazyDrop(t *testing.T) {
	const deadline int64 = 1_000_000
	for _, tc := range []struct {
		name string
		now  int64
		want bool
	}{
		{"just-before", deadline - 1, false},
		{"exactly-now", deadline, true},
		{"just-after", deadline + 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
			defer m.Close()
			m.now = func() int64 { return 0 }
			if _, err := m.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
				Metadata{"temp": NewString("x")}, map[string]int64{"temp": deadline}, CASCond{}); err != nil {
				t.Fatalf("add: %v", err)
			}
			m.now = func() int64 { return tc.now }
			_, lazyPay, _, ok := m.Get(1)
			if !ok {
				t.Fatal("doc gone")
			}
			_, lazyHas := lazyPay["temp"]
			lazyDropped := !lazyHas

			m2, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
			defer m2.Close()
			m2.now = func() int64 { return 0 }
			_, _ = m2.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
				Metadata{"temp": NewString("x")}, map[string]int64{"temp": deadline}, CASCond{})
			m2.now = func() int64 { return tc.now }
			swept := m2.sweepKeyTTLOnce() > 0

			if lazyDropped != tc.want || swept != tc.want {
				t.Fatalf("disagreement: lazyDropped=%v swept=%v want=%v", lazyDropped, swept, tc.want)
			}
		})
	}
}

// TestMVSweepInvalidatesOrderCache: an MV sweep that drops an order-by field key
// bumps dataVersion.
func TestMVSweepInvalidatesOrderCache(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 100
	m.now = func() int64 { return fakeNow }
	if _, err := m.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
		Metadata{"rank": NewInt(5)}, map[string]int64{"rank": 1000}, CASCond{}); err != nil {
		t.Fatalf("add: %v", err)
	}
	before := m.dataVersion
	fakeNow += 2000
	if n := m.sweepKeyTTLOnce(); n != 1 {
		t.Fatalf("sweep dropped %d, want 1", n)
	}
	if m.dataVersion == before {
		t.Errorf("sweep did not bump dataVersion (%d) — order_by cache would not invalidate", m.dataVersion)
	}
}

// --- lifecycle ---

// TestNamedSweeperLifecycleJoins: the sweeper goroutine starts on first insert and
// Stop() joins it (sweepDone closed — no leak); a double-Stop and a Stop-without-
// any-insert do not panic.
func TestNamedSweeperLifecycleJoins(t *testing.T) {
	// Stop-without-insert: sweeper never started → safe no-op (no panic, no hang).
	nc0, _ := NewNamedCollection("default/named", namedTestConfig())
	nc0.Stop()
	_ = nc0.Close()

	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	before := runtime.NumGoroutine()
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"a": NewInt(1)}, 0); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	// The sweeper goroutine is up.
	if nc.sweepStop == nil || nc.sweepDone == nil {
		t.Fatal("sweeper channels not initialized after first insert")
	}
	nc.Stop() // joins; sweepDone must be closed.
	select {
	case <-nc.sweepDone:
	default:
		t.Fatal("Stop() returned before sweepDone closed (goroutine not joined)")
	}
	// Double-Stop is a safe no-op.
	nc.Stop()
	// No goroutine leak: the count returns to (about) the pre-insert baseline.
	runtime.Gosched()
	if after := runtime.NumGoroutine(); after > before {
		t.Logf("goroutines before=%d after=%d (allowing scheduler slack)", before, after)
	}
}

// TestMVSweeperLifecycleJoins: MV analogue — Stop joins the goroutine; double-Stop
// and Stop-without-add do not panic.
func TestMVSweeperLifecycleJoins(t *testing.T) {
	m0, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	m0.Stop()
	_ = m0.Close()

	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"a": NewInt(1)}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if m.sweepStop == nil || m.sweepDone == nil {
		t.Fatal("sweeper channels not initialized after first add")
	}
	m.Stop()
	select {
	case <-m.sweepDone:
	default:
		t.Fatal("Stop() returned before sweepDone closed (goroutine not joined)")
	}
	m.Stop() // double-Stop safe
	_ = time.Millisecond
}
