// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"testing"
)

func TestNotLeaderErrorIs(t *testing.T) {
	wrapped := &NotLeaderError{LeaderAddr: "10.0.0.2:7102"}
	if !errors.Is(wrapped, ErrNotLeader) {
		t.Fatal("errors.Is(NotLeaderError, ErrNotLeader) must be true")
	}
}

func TestNotLeaderErrorAs(t *testing.T) {
	var wrapped error = &NotLeaderError{LeaderAddr: "10.0.0.2:7102"}
	var got *NotLeaderError
	if !errors.As(wrapped, &got) {
		t.Fatal("errors.As to *NotLeaderError must succeed")
	}
	if got.LeaderAddr != "10.0.0.2:7102" {
		t.Errorf("LeaderAddr = %q, want 10.0.0.2:7102", got.LeaderAddr)
	}
}

func TestNotLeaderErrorString(t *testing.T) {
	if (&NotLeaderError{}).Error() != "shard: not leader" {
		t.Error("Error() must return 'shard: not leader'")
	}
}

func TestSentinelErrNotLeaderIsNotLeaderError(t *testing.T) {
	var nle *NotLeaderError
	if !errors.As(ErrNotLeader, &nle) {
		t.Fatal("ErrNotLeader must be a *NotLeaderError")
	}
}
