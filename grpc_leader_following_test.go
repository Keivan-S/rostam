// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi"
	"github.com/rostamlabs/rostam/ops"
	pb "github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestGRPCWritesFollowTheLeaderOnEveryNode closes a coverage gap that was, for a
// while, load-bearing and untested.
//
// The server-side leader redirect — a write addressed to a node that HOSTS the
// shard but is not its leader is forwarded rather than refused — is what makes
// a client able to talk to any node. It broke once already: pbReplicator's
// LeaderAddr returned a node ID where a raft address was expected, the hint was
// dropped, and LeaderFollowingDispatcher silently degraded to a no-op under PB
// (since fixed). That regression was caught and pinned for HTTP. gRPC was
// only ever ASSUMED correct, on the strength of `httpDisp, grpcDisp = fan, fan`
// in NewServer — one assignment, no test, and a failure mode invisible to every
// HTTP test in the tree.
//
// The assertion is deliberately blunt: with n=3 and RF=3 every node hosts the
// shard, so exactly ONE can be the leader. If a gRPC write succeeds through
// every node's own dispatcher, then at least two of those writes were served by
// a non-leader and were redirected. That makes the test non-vacuous WITHOUT
// having to identify the leader — which is worth having, because the leader can
// move between the check and the write.
func TestGRPCWritesFollowTheLeaderOnEveryNode(t *testing.T) {
	stores, servers := newInmemEmbeddedClusterServers(t, 3, 4, 3)
	ctx := context.Background()

	createCollectionTolerant(t, ctx, stores[0], "docs", VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1,
	})

	for i, srv := range servers {
		if srv.grpcDisp == nil {
			t.Fatalf("node %d has no gRPC dispatcher wired", i)
		}
		gs := grpcapi.NewServer(srv.grpcDisp, nil)
		id := uint64(100 + i)
		_, err := gs.Upsert(ctx, &pb.UpsertRequest{
			Collection: "docs",
			Id:         id,
			Vector:     []float32{float32(i), 1, 2, 3},
			Upsert:     true,
		})
		if err != nil {
			t.Errorf("gRPC Upsert through node %d failed: %v — a node that hosts the shard but is not its "+
				"leader must redirect the write, not refuse it", i, err)
		}
	}

	// The writes must actually be READABLE afterwards, or "success" above could be
	// a redirect that dropped the payload rather than one that delivered it.
	for i := range servers {
		id := uint64(100 + i)
		var found bool
		retryUntil(t, "get", func() error {
			ok, _, _, _, _, err := stores[0].VectorGet(ctx, "docs", id, true, false)
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("point %d not visible yet", id)
			}
			found = true
			return nil
		})
		if !found {
			t.Errorf("point %d, written through node %d's gRPC dispatcher, is not readable", id, i)
		}
	}
}

// TestGRPCAndHTTPDispatchersBothRedirect asserts the property directly, at the
// dispatcher seam, for BOTH client-facing transports.
//
// Pointer identity would be the tempting assertion — NewServer assigns both from
// the same `fan` — but it is not actually true and asserting it would be a trap:
// each transport gets its OWN keys-decorator instance wrapping that shared
// chain. What must hold is behavioural, so that is what is checked: a write
// dispatched through either transport's chain, on a node that hosts the shard
// but may not lead it, must succeed.
//
// TCP is deliberately excluded: a TCP client follows the leader hint itself, so
// its dispatcher stays unwrapped on purpose.
func TestGRPCAndHTTPDispatchersBothRedirect(t *testing.T) {
	stores, servers := newInmemEmbeddedClusterServers(t, 3, 4, 3)
	ctx := context.Background()

	createCollectionTolerant(t, ctx, stores[0], "docs", VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1,
	})

	// n=3, RF=3 ⇒ every node hosts the shard and at most one leads it, so each
	// loop below necessarily drives at least two redirects.
	for i, srv := range servers {
		for _, tr := range []struct {
			name string
			d    interface {
				Call(string, []byte) ([]byte, error)
			}
		}{
			{"http", srv.httpDisp},
			{"grpc", srv.grpcDisp},
		} {
			if tr.d == nil {
				t.Fatalf("node %d: %s transport has no dispatcher wired", i, tr.name)
			}
			id := uint64(1000 + i*10)
			if tr.name == "grpc" {
				id++
			}
			args := ops.EncodeVectorUpsertArgs("docs", id, []float32{float32(i), 1, 2, 3}, "", 0, nil, vector.SparseVector{})
			if _, err := tr.d.Call("vector_upsert", args); err != nil {
				t.Errorf("node %d: %s dispatcher refused a write (%v) — it is missing the server-side "+
					"leader redirect the other transport has", i, tr.name, err)
			}
		}
	}
}
