// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

func TestFanOutMergesAllPartitions(t *testing.T) {
	// Fake caller returns one result per partition, distance = partition index.
	caller := func(physCol string, op string, args []byte, leaderOnly bool) ([]byte, error) {
		// physCol ends with "#<p>"; encode p as the single result's distance.
		p := physCol[len(physCol)-1] - '0'
		return []byte{p}, nil
	}
	decode := func(b []byte) ([]Scored, error) {
		return []Scored{{ID: uint64(b[0]), Dist: float32(b[0])}}, nil
	}
	res, fr, err := FanOut(FanArgs{
		Collection: "default/docs", P: 3, K: 10,
		Consistency: AnyReplica, OnUnavailable: Partial,
		Op:     "vector_search",
		Encode: func(physCol string) []byte { return []byte(physCol) },
	}, caller, decode, func(parts [][]Scored, k int) []Scored {
		return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 || fr.Degraded {
		t.Fatalf("got %d results degraded=%v, want 3 non-degraded", len(res), fr.Degraded)
	}
}

func TestFanOutUsesGeneration(t *testing.T) {
	// The fake caller records every physCol it is handed; FanOut must build each
	// physical name via PartitionKeyGen(collection, Generation, p) so a non-zero
	// generation routes to the "@g#p" physical partitions, not the legacy "#p".
	const coll, P, gen = "default/docs", 3, uint32(2)
	var (
		mu  sync.Mutex
		got []string
	)
	caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
		mu.Lock()
		got = append(got, physCol)
		mu.Unlock()
		return []byte{0}, nil
	}
	decode := func(b []byte) ([]Scored, error) { return []Scored{{ID: uint64(b[0])}}, nil }
	merge := func(parts [][]Scored, k int) []Scored {
		return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
	}
	_, _, err := FanOut(FanArgs{
		Collection: coll, P: P, K: 10, Generation: gen,
		Encode: func(physCol string) []byte { return []byte(physCol) },
	}, caller, decode, merge)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{}
	for p := 0; p < P; p++ {
		want[string(ops.PartitionKeyGen(coll, gen, p))] = true
	}
	if len(got) != P {
		t.Fatalf("caller invoked %d times, want %d", len(got), P)
	}
	for _, pc := range got {
		if !want[pc] {
			t.Fatalf("physCol %q not a gen-%d partition name of %q (want one of %v)", pc, gen, coll, want)
		}
	}
}

func TestFanOutForwardsLeaderOnly(t *testing.T) {
	for _, tc := range []struct {
		name string
		cons Consistency
		want bool
	}{
		{"any-replica", AnyReplica, false},
		{"leader-only", LeaderOnly, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
				if leaderOnly != tc.want {
					t.Errorf("leaderOnly=%v, want %v", leaderOnly, tc.want)
				}
				return []byte{0}, nil
			}
			decode := func(b []byte) ([]Scored, error) { return []Scored{{ID: uint64(b[0])}}, nil }
			merge := func(parts [][]Scored, k int) []Scored {
				return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
			}
			_, _, err := FanOut(FanArgs{
				Collection: "default/docs", P: 2, K: 10,
				Consistency: tc.cons,
				Encode:      func(physCol string) []byte { return []byte(physCol) },
			}, caller, decode, merge)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestFanOutPartialOnUnavailable(t *testing.T) {
	caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
		if physCol[len(physCol)-1] == '1' {
			return nil, errors.New("no leader")
		}
		return []byte{physCol[len(physCol)-1] - '0'}, nil
	}
	decode := func(b []byte) ([]Scored, error) {
		return []Scored{{ID: uint64(b[0]), Dist: float32(b[0])}}, nil
	}
	merge := func(parts [][]Scored, k int) []Scored {
		return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
	}
	// Partial: missing partition 1, return the other two, degraded.
	res, fr, err := FanOut(FanArgs{Collection: "default/docs", P: 3, K: 10, OnUnavailable: Partial,
		Encode: func(physCol string) []byte { return []byte(physCol) }}, caller, decode, merge)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || !fr.Degraded || len(fr.Missing) != 1 || fr.Missing[0] != 1 {
		t.Fatalf("got %d res degraded=%v missing=%v, want 2/true/[1]", len(res), fr.Degraded, fr.Missing)
	}
	// Fail: same failure errors out.
	_, _, err = FanOut(FanArgs{Collection: "default/docs", P: 3, K: 10, OnUnavailable: Fail,
		Encode: func(physCol string) []byte { return []byte(physCol) }}, caller, decode, merge)
	if err == nil {
		t.Fatal("OnUnavailable=Fail should error when a partition is unreachable")
	}
}

// TestFanOutInlinePartitionErrorPropagates pins that partition P-1 — which now
// runs on the CALLING goroutine instead of a spawned one — reports an error
// exactly like any spawned partition would (TestFanOutPartialOnUnavailable
// already covers a non-last, spawned partition failing).
func TestFanOutInlinePartitionErrorPropagates(t *testing.T) {
	const P = 3
	caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
		if int(physCol[len(physCol)-1]-'0') == P-1 {
			return nil, errors.New("inline partition unreachable")
		}
		return []byte{physCol[len(physCol)-1] - '0'}, nil
	}
	decode := func(b []byte) ([]Scored, error) {
		return []Scored{{ID: uint64(b[0]), Dist: float32(b[0])}}, nil
	}
	merge := func(parts [][]Scored, k int) []Scored {
		return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
	}
	// Partial: the inline partition's error is reported as missing, same as any
	// spawned partition's error would be.
	res, fr, err := FanOut(FanArgs{Collection: "default/docs", P: P, K: 10, OnUnavailable: Partial,
		Encode: func(physCol string) []byte { return []byte(physCol) }}, caller, decode, merge)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != P-1 || !fr.Degraded || len(fr.Missing) != 1 || fr.Missing[0] != P-1 {
		t.Fatalf("got %d res degraded=%v missing=%v, want %d/true/[%d]", len(res), fr.Degraded, fr.Missing, P-1, P-1)
	}
	// Fail: the inline partition's error still fails the whole call.
	_, _, err = FanOut(FanArgs{Collection: "default/docs", P: P, K: 10, OnUnavailable: Fail,
		Encode: func(physCol string) []byte { return []byte(physCol) }}, caller, decode, merge)
	if err == nil {
		t.Fatal("OnUnavailable=Fail should error when the inline (last) partition is unreachable")
	}
}

// TestFanOutZeroPartitionsNoop pins that P==0 is a no-op (no caller
// invocation, no panic from the inline runPartition(a.P-1) call) rather than
// indexing results[-1].
func TestFanOutZeroPartitionsNoop(t *testing.T) {
	called := false
	caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
		called = true
		return []byte{0}, nil
	}
	decode := func(b []byte) ([]Scored, error) { return []Scored{{ID: uint64(b[0])}}, nil }
	merge := func(parts [][]Scored, k int) []Scored {
		return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
	}
	res, fr, err := FanOut(FanArgs{Collection: "default/docs", P: 0, K: 10,
		Encode: func(physCol string) []byte { return []byte(physCol) }}, caller, decode, merge)
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("caller invoked with P=0, want no-op")
	}
	if len(res) != 0 || fr.Degraded || len(fr.Missing) != 0 {
		t.Fatalf("got %d res degraded=%v missing=%v, want 0/false/[]", len(res), fr.Degraded, fr.Missing)
	}
}
