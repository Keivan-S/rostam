// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestSlice2SingleShardMigration migrates shard 0 from its sole owner (n1) to a
// new sole owner (n2) as one coordinated gain-then-release operation, while a
// background writer keeps issuing puts to shard-0 keys. It asserts: every
// acknowledged write is readable afterwards (no lost writes), n2 ends up the
// sole owner/host, n1 no longer hosts the shard, and placement (local + meta)
// reflects the new owner.
func TestSlice2SingleShardMigration(t *testing.T) {
	// RF=1, 2 nodes, 2 shards: shard 0 owned only by n1.
	tc := newTestCluster(t, 2, 2, 1)
	n1, n2 := tc.nodes[0], tc.nodes[1]
	if n1.getShard(0) == nil || n2.getShard(0) != nil {
		t.Fatal("precondition: shard 0 should be owned only by n1")
	}
	mc := MigrationClusterFromPeers(map[string]*Node{"n1": n1, "n2": n2}, tc.peers)

	// shard0Key returns the i-th key (by probe order) that routes to shard 0.
	shard0Key := func(i int) []byte {
		found := 0
		for j := 0; ; j++ {
			k := fmt.Appendf(nil, "k-%d", j)
			if shardOf(k, 2) == 0 {
				if found == i {
					return k
				}
				found++
			}
		}
	}

	// --- Background writer: unique shard-0 keys, recording only acked writes. ---
	var (
		mu    sync.Mutex
		acked = map[string]string{}
	)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ctx := context.Background()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			key := shard0Key(i)
			val := fmt.Appendf(nil, "v-%d", i)
			// Retry transient errors (NotLeader / no-leader windows during the
			// membership change). Only record on a confirmed ack.
			var ok bool
			for attempt := 0; attempt < 50; attempt++ {
				if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(key, val, 0)); err == nil {
					ok = true
					break
				}
				time.Sleep(20 * time.Millisecond)
			}
			if ok {
				mu.Lock()
				acked[string(key)] = string(val)
				mu.Unlock()
			}
			time.Sleep(2 * time.Millisecond)
		}
	}()

	// Give the writer a moment to commit some writes before migrating.
	time.Sleep(100 * time.Millisecond)

	// --- The migration under test. ---
	if err := mc.MigrateShard(0, []string{"n2"}, 15*time.Second); err != nil {
		close(stop)
		wg.Wait()
		t.Fatalf("MigrateShard: %v", err)
	}

	// Let the writer run a bit longer post-migration, then stop it.
	time.Sleep(100 * time.Millisecond)
	close(stop)
	wg.Wait()

	// --- Final ownership / hosting. ---
	if n2.getShard(0) == nil {
		t.Fatal("n2 should host shard 0 after migration")
	}
	if n1.getShard(0) != nil {
		t.Fatal("n1 should no longer host shard 0 after migration")
	}
	if got := n2.ownersFor(0); len(got) != 1 || got[0] != "n2" {
		t.Fatalf("n2 placement[0] = %v, want [n2]", got)
	}
	if got := n1.ownersFor(0); len(got) != 1 || got[0] != "n2" {
		t.Fatalf("n1 placement[0] = %v, want [n2]", got)
	}

	// --- Meta-Raft placement converged to the new owner. ---
	deadline := time.Now().Add(5 * time.Second)
	for {
		st := n2.meta.FSM.State()
		if len(st.Placement) > 0 && len(st.Placement[0]) == 1 && st.Placement[0][0] == "n2" {
			break
		}
		if !time.Now().Before(deadline) {
			t.Fatalf("meta placement[0] = %v, want [n2]", st.Placement[0])
		}
		time.Sleep(50 * time.Millisecond)
	}

	// --- No lost writes: every acked key is readable with its acked value. ---
	mu.Lock()
	defer mu.Unlock()
	if len(acked) == 0 {
		t.Fatal("no writes were acknowledged; test exercised nothing")
	}
	ctx := context.Background()
	for key, want := range acked {
		var got []byte
		// Reads may briefly trail the just-completed migration; retry a few.
		for attempt := 0; attempt < 50; attempt++ {
			v, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs([]byte(key)))
			if err == nil {
				got = v
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if string(got) != want {
			t.Fatalf("lost write: key %q = %q, want %q", key, got, want)
		}
	}
	t.Logf("migration verified: %d acked writes all readable on new owner", len(acked))
}
