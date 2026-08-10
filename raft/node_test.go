// SPDX-License-Identifier: Apache-2.0

package raft

import (
	"bytes"
	"io"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"
)

// nopFSM is a no-op FSM for lifecycle smoke tests.
type nopFSM struct{}

func (nopFSM) Apply(*hraft.Log) any                 { return nil }
func (nopFSM) Snapshot() (hraft.FSMSnapshot, error) { return nopSnap{}, nil }
func (nopFSM) Restore(rc io.ReadCloser) error       { return rc.Close() }

type nopSnap struct{}

func (nopSnap) Persist(sink hraft.SnapshotSink) error { return sink.Close() }
func (nopSnap) Release()                              {}

// newSingleNodeLeader boots a single-node raft cluster and waits for it to win
// the (trivial, quorum==1) election. Mirrors TestNodeBootstrapShutdown's setup.
func newSingleNodeLeader(t *testing.T) *Node {
	t.Helper()
	cfg := Config{
		NodeID:      "n1",
		DataDir:     t.TempDir(),
		Bootstrap:   true,
		HeartbeatMs: 50,
		ElectionMs:  100,
		NoSync:      true,
	}
	n, err := NewNode(cfg, nopFSM{})
	if err != nil {
		t.Fatalf("NewNode: %v", err)
	}
	t.Cleanup(func() { _ = n.Shutdown() })
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && !n.IsLeader() {
		time.Sleep(20 * time.Millisecond)
	}
	if !n.IsLeader() {
		t.Fatalf("single-node never became leader; state=%v", n.raft.State())
	}
	return n
}

func TestVerifyLeaderSingleNodeReturnsNilQuickly(t *testing.T) {
	n := newSingleNodeLeader(t)
	// quorum==1 ⇒ VerifyLeader resolves immediately with no round-trip; it must
	// not hang and must report still-leader (nil).
	done := make(chan error, 1)
	go func() { done <- n.VerifyLeader() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("VerifyLeader on single-node leader = %v, want nil", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("VerifyLeader hung on a single-node leader (quorum==1 must be immediate)")
	}
}

func TestCommitIndexMonotonicAfterApply(t *testing.T) {
	n := newSingleNodeLeader(t)
	before := n.CommitIndex()
	_, idx, err := n.ApplyIndexed([]byte("entry"), 5*time.Second)
	if err != nil {
		t.Fatalf("ApplyIndexed: %v", err)
	}
	after := n.CommitIndex()
	if after < idx {
		t.Fatalf("CommitIndex()=%d after applying entry at index %d; must be >= it", after, idx)
	}
	if after < before {
		t.Fatalf("CommitIndex went backwards: before=%d after=%d", before, after)
	}
}

func TestNodeBootstrapShutdown(t *testing.T) {
	cfg := Config{
		NodeID:      "n1",
		DataDir:     t.TempDir(),
		Bootstrap:   true,
		HeartbeatMs: 50,
		ElectionMs:  100,
		NoSync:      true,
	}
	n, err := NewNode(cfg, nopFSM{})
	if err != nil {
		t.Fatalf("NewNode: %v", err)
	}
	defer func() { _ = n.Shutdown() }()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if n.IsLeader() {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !n.IsLeader() {
		t.Fatalf("single-node never became leader; state=%v", n.raft.State())
	}
}

// TestLogLevelAndOutputAreHonoured pins the two knobs that let an embedder
// quiet hashicorp/raft down.
//
// It exists because the absence of these was not a cosmetic problem. hashicorp's
// DefaultConfig ships LogLevel "DEBUG", and `go test` merges the test binary's
// stdout and stderr into one stream — so in any suite that builds clusters in a
// loop, every pre-vote and configuration change interleaved with the results.
// `make bench`, which this repo's own docs tell readers to run, printed
// benchmark rows that were unparseable. Three separate attempts to extract clean
// numbers from it failed before the cause was found.
//
// The assertion is deliberately coarse — that ERROR is quieter than DEBUG on the
// same startup path — rather than pinning any particular line, so it does not
// break when hashicorp changes its wording.
func TestLogLevelAndOutputAreHonoured(t *testing.T) {
	run := func(level string) int {
		var buf bytes.Buffer
		n, err := NewNode(Config{
			NodeID:      "n1",
			DataDir:     t.TempDir(),
			Bootstrap:   true,
			HeartbeatMs: 50,
			ElectionMs:  100,
			NoSync:      true,
			LogLevel:    level,
			LogOutput:   &buf,
		}, nopFSM{})
		if err != nil {
			t.Fatalf("NewNode(%q): %v", level, err)
		}
		// Let it elect, which is what produces the chatter in the first place.
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) && !n.IsLeader() {
			time.Sleep(10 * time.Millisecond)
		}
		_ = n.Shutdown()
		return buf.Len()
	}

	debug := run("DEBUG")
	quiet := run("ERROR")

	if debug == 0 {
		t.Fatal("DEBUG wrote nothing to the configured LogOutput — the knob is not wired at all")
	}
	if quiet >= debug {
		t.Errorf("ERROR wrote %d bytes and DEBUG wrote %d: the level is not being applied", quiet, debug)
	}
}
