// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/rostamlabs/rostam/ops/wire"
	"github.com/rostamlabs/rostam/vtypes"
)

// decodeOpFrame strips the wire's [opNameLen:1][opName][argsLen:4][args]
// request frame (client/conn.go's doCall v1 wire, no auth) into the op name
// and the raw args bytes.
func decodeOpFrame(t *testing.T, body []byte) (opName string, args []byte) {
	t.Helper()
	if len(body) < 1 {
		t.Fatalf("frame too short: %d bytes", len(body))
	}
	opLen := int(body[0])
	if len(body) < 1+opLen+4 {
		t.Fatalf("frame too short for op name: %d bytes", len(body))
	}
	opName = string(body[1 : 1+opLen])
	off := 1 + opLen
	argsLen := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	if len(body) < off+argsLen {
		t.Fatalf("frame too short for args: %d bytes", len(body))
	}
	args = body[off : off+argsLen]
	return opName, args
}

// TestRecommendByExampleIDs proves Recommend sends the canonical vector_query
// spec shape — the SAME shape the HTTP /query handler builds for a recommend
// request (httpapi/vector.go's (queryLeafReq).toLeaf, appended into
// spec.Prefetch via vector.LeafSource — see also
// httpapi/httpapi_test.go's TestHTTPQueryRecommendRoundTrip, which verifies the
// identical spec shape at the HTTP edge): ModeFusion, no Root, a single
// Prefetch leaf of Kind LeafRecommend carrying the example ids. It captures
// the raw request frame via a fake single-shot server (mirroring
// client/client_test.go's startFakeServer/acceptAndRespond pattern) rather than
// running against startTestStack's coordinator-less single shard, because a
// ModeFusion vector_query result only decodes through
// wire.DecodeQueryResultDegraded once a fan-out coordinator has re-encoded the
// per-lane FUSION result as a flat RERANK-tagged wire (fanout_dispatcher.go's
// fanQuery) — a real deployment topology that startTestStack's raw
// coordinator-less shard.Store dispatcher does not have. The fake server here
// stands in for that coordinator: it decodes and asserts the outgoing spec,
// then returns a canned flat result via wire.EncodeQueryResultFusedDegraded so
// Recommend's decode path is exercised too.
func TestRecommendByExampleIDs(t *testing.T) {
	want := []vtypes.Result{{ID: 2, Distance: 0.1, Score: 0.9}, {ID: 3, Distance: 0.2, Score: 0.5}}

	var gotOp string
	var gotArgs []byte
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		gotOp, gotArgs = decodeOpFrame(t, body)
		return StatusOK, wire.EncodeQueryResultFusedDegraded(want, false, nil)
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")
	ctx := context.Background()

	resp, err := col.Recommend(ctx, RecommendRequest{Positive: []uint64{1}, K: 2})
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}

	if gotOp != "vector_query" {
		t.Fatalf("op = %q, want vector_query", gotOp)
	}
	_, _, spec, _, _, _, err := wire.DecodeQuerySpecArgs(gotArgs)
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	if spec.Mode != vtypes.ModeFusion {
		t.Fatalf("spec.Mode = %v, want ModeFusion", spec.Mode)
	}
	if len(spec.Prefetch) != 1 || spec.Prefetch[0].Leaf == nil {
		t.Fatalf("spec.Prefetch = %+v, want exactly one leaf source", spec.Prefetch)
	}
	leaf := spec.Prefetch[0].Leaf
	if leaf.Kind != vtypes.LeafRecommend {
		t.Fatalf("prefetch leaf.Kind = %v, want LeafRecommend", leaf.Kind)
	}
	if len(leaf.Positive) != 1 || leaf.Positive[0] != 1 {
		t.Fatalf("leaf.Positive = %v, want [1]", leaf.Positive)
	}
	if len(leaf.Negative) != 0 {
		t.Fatalf("leaf.Negative = %v, want empty", leaf.Negative)
	}
	// Root is unused by ModeFusion — the canonical shape carries no root leaf.
	if len(spec.Root.Positive) != 0 || len(spec.Root.Dense) != 0 || spec.Root.Kind != vtypes.LeafDense {
		t.Fatalf("spec.Root = %+v, want empty/unset", spec.Root)
	}

	if len(resp.Results) != 2 || resp.Results[0].ID != 2 || resp.Results[1].ID != 3 {
		t.Fatalf("Results = %v, want [2 3]", resp.Results)
	}
}
