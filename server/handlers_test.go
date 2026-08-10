// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"

	"github.com/rostamlabs/rostam/shard"
)

type fakeDispatcher struct{ leader string }

func (f *fakeDispatcher) Call(string, []byte) ([]byte, error) { return nil, nil }
func (f *fakeDispatcher) LeaderAddr() string                  { return f.leader }

func TestMapResultPrefersNotLeaderErrorHint(t *testing.T) {
	// Even though disp says "fallback", the wrapped error carries
	// a specific per-shard hint. mapResult must surface the hint.
	disp := &fakeDispatcher{leader: "fallback"}
	err := &shard.NotLeaderError{LeaderAddr: "10.0.0.2:7001"}
	status, payload := mapResult(disp, nil, err, "")
	if status != StatusNotLeader {
		t.Fatalf("status = %d, want StatusNotLeader", status)
	}
	got, derr := DecodeLeaderAddrPayload(payload)
	if derr != nil {
		t.Fatalf("DecodeLeaderAddrPayload: %v", derr)
	}
	if got != "10.0.0.2:7001" {
		t.Errorf("addr = %q, want 10.0.0.2:7001", got)
	}
}

func TestMapResultFallsBackToDispatcherLeaderAddrOnBareSentinel(t *testing.T) {
	disp := &fakeDispatcher{leader: "fallback"}
	status, payload := mapResult(disp, nil, shard.ErrNotLeader, "")
	if status != StatusNotLeader {
		t.Fatalf("status = %d, want StatusNotLeader", status)
	}
	got, derr := DecodeLeaderAddrPayload(payload)
	if derr != nil {
		t.Fatalf("DecodeLeaderAddrPayload: %v", derr)
	}
	if got != "fallback" {
		t.Errorf("addr = %q, want fallback (dispatcher fallback)", got)
	}
}

func TestMapResultFallsBackWhenHintIsEmpty(t *testing.T) {
	// Wrapped error with empty LeaderAddr should still fall back
	// to disp.LeaderAddr().
	disp := &fakeDispatcher{leader: "fallback"}
	err := &shard.NotLeaderError{LeaderAddr: ""}
	status, payload := mapResult(disp, nil, err, "")
	if status != StatusNotLeader {
		t.Fatalf("status = %d", status)
	}
	got, derr := DecodeLeaderAddrPayload(payload)
	if derr != nil {
		t.Fatalf("DecodeLeaderAddrPayload: %v", derr)
	}
	if got != "fallback" {
		t.Errorf("empty hint should fall back; got %q", got)
	}
}
