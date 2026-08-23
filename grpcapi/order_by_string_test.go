// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"errors"
	"testing"

	pb "github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestParsePBOrderByString asserts the gRPC OrderBy parser maps is_string ⇒
// OrderString, keeps numeric/datetime byte/behaviour-identical, and fails loud on a bad
// order-kind combination (is_string + is_datetime, or is_string + start_from).
func TestParsePBOrderByString(t *testing.T) {
	// is_string ⇒ Kind=OrderString, Desc preserved.
	ob, err := parsePBOrderBy(&pb.OrderBy{Key: "city", Desc: true, IsString: true})
	if err != nil {
		t.Fatalf("string order_by: %v", err)
	}
	if ob == nil || ob.Kind != vector.OrderString || !ob.Desc || ob.Key != "city" {
		t.Fatalf("string order_by parsed wrong: %+v", ob)
	}

	// numeric (default) ⇒ OrderNumeric, IsDatetime false — unchanged.
	num, err := parsePBOrderBy(&pb.OrderBy{Key: "rank"})
	if err != nil {
		t.Fatal(err)
	}
	if num.Kind != vector.OrderNumeric || num.IsDatetime {
		t.Fatalf("numeric order_by regressed: %+v", num)
	}

	// datetime ⇒ OrderDatetime + IsDatetime true — unchanged.
	dt, err := parsePBOrderBy(&pb.OrderBy{Key: "ts", IsDatetime: true})
	if err != nil {
		t.Fatal(err)
	}
	if dt.Kind != vector.OrderDatetime || !dt.IsDatetime {
		t.Fatalf("datetime order_by regressed: %+v", dt)
	}

	// Bad combo: is_string + is_datetime ⇒ ErrBadOrderKind (InvalidArgument at the edge).
	if _, err := parsePBOrderBy(&pb.OrderBy{Key: "city", IsString: true, IsDatetime: true}); !errors.Is(err, vector.ErrBadOrderKind) {
		t.Fatalf("is_string+is_datetime: got err=%v, want ErrBadOrderKind", err)
	}

	// Bad combo: is_string + start_from ⇒ ErrBadOrderKind.
	start := 3.0
	if _, err := parsePBOrderBy(&pb.OrderBy{Key: "city", IsString: true, StartFrom: &start}); !errors.Is(err, vector.ErrBadOrderKind) {
		t.Fatalf("is_string+start_from: got err=%v, want ErrBadOrderKind", err)
	}

	// nil ⇒ (nil, nil) — no order_by.
	if got, err := parsePBOrderBy(nil); got != nil || err != nil {
		t.Fatalf("nil order_by: got (%v,%v), want (nil,nil)", got, err)
	}
}
