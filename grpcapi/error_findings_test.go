// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"errors"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/vector"
)

// TestGrpcErrorConflictAndBackpressure covers finding 019 (gRPC side): the four
// collection-level outcomes that previously fell through to the Internal default.
// Create-conflicts map to AlreadyExists; quota/rate-limit refusals to
// ResourceExhausted. Both the sentinel path and the clustered/stringified fallback
// are exercised, mirroring the HTTP 409/429 mapping.
func TestGrpcErrorConflictAndBackpressure(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want codes.Code
	}{
		{"exists-sentinel", vector.ErrCollectionExists, codes.AlreadyExists},
		{"dupid-sentinel", vector.ErrDuplicateID, codes.AlreadyExists},
		{"ratelimited-sentinel", vector.ErrCollectionRateLimited, codes.ResourceExhausted},
		{"full-sentinel", vector.ErrCollectionFull, codes.ResourceExhausted},
		{"exists-string", errors.New("rostam: collection already exists"), codes.AlreadyExists},
		{"dupid-string", errors.New("rostam: id already present (delete first)"), codes.AlreadyExists},
		{"ratelimited-string", errors.New("rostam: collection insert rate limited"), codes.ResourceExhausted},
		{"full-string", errors.New("rostam: collection full (quota exceeded)"), codes.ResourceExhausted},
	}
	for _, tc := range cases {
		if got := status.Code(grpcError(tc.err)); got != tc.want {
			t.Errorf("%s: grpcError = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestGrpcErrorLeaderlessIsRetryable covers finding 021: a leaderless/ownership
// transient must map to the RETRYABLE codes.Unavailable (mirroring HTTP's 503), not
// the non-retryable codes.Internal that standard gRPC retry policies and service
// meshes ignore. The reachable divergent case is client.ErrNoLeaderKnown ("client:
// no leader known after retries") which contains "no leader", not "not leader"; the
// secondary case is cluster.ErrNoShardOwner.
func TestGrpcErrorLeaderlessIsRetryable(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"no-leader-known", errors.New("client: no leader known after retries")},
		{"not-leader", errors.New("shard: not leader for partition 3")},
		{"no-reachable-owner", errors.New("cluster: no reachable owner for shard")},
	}
	for _, tc := range cases {
		if got := status.Code(grpcError(tc.err)); got != codes.Unavailable {
			t.Errorf("%s: grpcError = %v, want Unavailable (retryable)", tc.name, got)
		}
	}
}
