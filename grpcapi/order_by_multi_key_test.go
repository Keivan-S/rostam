// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"errors"
	"testing"

	pb "github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestParsePBOrderByMultiKey asserts the gRPC OrderBy parser builds the MULTI-KEY Tail
// from tail_keys (each carrying its own key/desc/kind), keeps a no-tail request the
// byte-identical single-key path (empty Tail), and fails loud on a bad tail-key kind.
func TestParsePBOrderByMultiKey(t *testing.T) {
	// Primary price desc (numeric) + tail name asc (string) + tail ts desc (datetime).
	ob, err := parsePBOrderBy(&pb.OrderBy{
		Key: "price", Desc: true,
		TailKeys: []*pb.OrderBy{
			{Key: "name", IsString: true},
			{Key: "ts", Desc: true, IsDatetime: true},
		},
	})
	if err != nil {
		t.Fatalf("multi-key order_by: %v", err)
	}
	if ob == nil || ob.Key != "price" || !ob.Desc || ob.Kind != vector.OrderNumeric {
		t.Fatalf("primary parsed wrong: %+v", ob)
	}
	if len(ob.Tail) != 2 {
		t.Fatalf("Tail len = %d, want 2 (%+v)", len(ob.Tail), ob.Tail)
	}
	if ob.Tail[0].Key != "name" || ob.Tail[0].Kind != vector.OrderString || ob.Tail[0].Desc {
		t.Fatalf("tail[0] parsed wrong: %+v", ob.Tail[0])
	}
	if ob.Tail[1].Key != "ts" || ob.Tail[1].Kind != vector.OrderDatetime || !ob.Tail[1].Desc || !ob.Tail[1].IsDatetime {
		t.Fatalf("tail[1] parsed wrong: %+v", ob.Tail[1])
	}

	// No tail_keys ⇒ the single-key path: empty Tail, byte-identical.
	single, err := parsePBOrderBy(&pb.OrderBy{Key: "price", Desc: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(single.Tail) != 0 {
		t.Fatalf("single-key order_by leaked a Tail: %+v", single.Tail)
	}

	// A bad tail-key kind (is_string + is_datetime) ⇒ ErrBadOrderKind (loud at the edge).
	if _, err := parsePBOrderBy(&pb.OrderBy{
		Key: "price", TailKeys: []*pb.OrderBy{{Key: "x", IsString: true, IsDatetime: true}},
	}); !errors.Is(err, vector.ErrBadOrderKind) {
		t.Fatalf("bad tail-key kind: got err=%v, want ErrBadOrderKind", err)
	}
}
