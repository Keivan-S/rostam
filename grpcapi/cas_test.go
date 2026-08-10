// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// conflictDispatcher returns ErrVersionConflict from Call so the error-mapping
// path (grpcError → FailedPrecondition) can be exercised. It also records the
// last args so the CAS encoder threading can be asserted.
type conflictDispatcher struct {
	lastOp  string
	lastArg []byte
}

func (d *conflictDispatcher) Call(name string, args []byte) ([]byte, error) {
	d.lastOp, d.lastArg = name, args
	return nil, vector.ErrVersionConflict
}
func (d *conflictDispatcher) LeaderAddr() string { return "" }

func TestGRPCVersionConflictMapsToFailedPrecondition(t *testing.T) {
	cases := []struct {
		name string
		call func(s *Server) error
	}{
		{"upsert", func(s *Server) error {
			_, err := s.Upsert(context.Background(), &pb.UpsertRequest{Collection: "docs", Id: 1, Vector: []float32{1}, ExpectedVersion: 5, HasExpectedVersion: true})
			return err
		}},
		{"delete", func(s *Server) error {
			_, err := s.Delete(context.Background(), &pb.DeleteRequest{Collection: "docs", Id: 1, ExpectedVersion: 5, HasExpectedVersion: true})
			return err
		}},
		{"set_payload", func(s *Server) error {
			_, err := s.SetPayload(context.Background(), &pb.SetPayloadRequest{Collection: "docs", Id: 1, ExpectedVersion: 5, HasExpectedVersion: true})
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := NewServer(&conflictDispatcher{}, nil)
			err := tc.call(s)
			if status.Code(err) != codes.FailedPrecondition {
				t.Fatalf("code = %v, want FailedPrecondition (err=%v)", status.Code(err), err)
			}
		})
	}
}

// versionDispatcher returns a Get result carrying a version so the Get handler's
// version decode + response field can be asserted.
type versionDispatcher struct{ version uint64 }

func (d *versionDispatcher) Call(name string, args []byte) ([]byte, error) {
	return ops.EncodeVectorGetResultV(true, []float32{1, 2}, nil, 0, nil, true, true, d.version), nil
}
func (d *versionDispatcher) LeaderAddr() string { return "" }

func TestGRPCGetCarriesVersion(t *testing.T) {
	s := NewServer(&versionDispatcher{version: 7}, nil)
	resp, err := s.Get(context.Background(), &pb.GetRequest{Collection: "docs", Id: 1})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if resp.GetVersion() != 7 {
		t.Fatalf("version = %d, want 7", resp.GetVersion())
	}
}

// TestGRPCNoCASWireUnchanged proves a request without the CAS fields produces the
// SAME op args as the pre-CAS encoder (no expected_version trailer).
func TestGRPCNoCASWireUnchanged(t *testing.T) {
	d := &conflictDispatcher{}
	s := NewServer(d, nil)
	_, _ = s.Delete(context.Background(), &pb.DeleteRequest{Collection: "docs", Id: 1}) // no CAS fields
	legacy := ops.EncodeVectorDeleteArgs("docs", 1)
	if string(d.lastArg) != string(legacy) {
		t.Fatalf("no-CAS delete args = %v, want legacy %v", d.lastArg, legacy)
	}
}
