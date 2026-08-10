// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"errors"
	"sync"
	"time"
)

// inMemTransport wires several engines' Receive methods together by peer ID for
// tests. It implements the async Transport contract (submit + completion
// callback) and supports per-peer fault injection so tests can exercise each
// hazard without a real network:
//
//   - Partition: done is NEVER invoked — the primary's Propose ctx times the
//     in-flight record out (the engine owns the deadline now, not the transport).
//   - Drop: the message is lost; done fires once with a negative ack (OK:false).
//   - Delay: done fires after a fixed delay, from a goroutine so Replicate stays
//     non-blocking (test-only).
//   - AckOverride: done fires with a crafted ack instead of the peer's real one
//     (used to forge a stale-epoch or wrong-seq ack for H6).
//
// The transport itself is not part of the correctness core; only engines are.
type inMemTransport struct {
	mu     sync.Mutex
	peers  map[string]*Engine
	faults map[string]peerFault
}

// peerFault describes injected misbehavior toward a single peer.
type peerFault struct {
	partition   bool
	drop        bool
	delay       time.Duration
	ackOverride func(msg ReplicateMsg) AckMsg
}

func newInMemTransport() *inMemTransport {
	return &inMemTransport{
		peers:  make(map[string]*Engine),
		faults: make(map[string]peerFault),
	}
}

// register makes eng reachable as peer under its node ID.
func (t *inMemTransport) register(nodeID string, eng *Engine) {
	t.mu.Lock()
	t.peers[nodeID] = eng
	t.mu.Unlock()
}

// setFault installs (or clears, with the zero value) fault injection for peer.
func (t *inMemTransport) setFault(peer string, f peerFault) {
	t.mu.Lock()
	t.faults[peer] = f
	t.mu.Unlock()
}

// clearFault removes any injected fault for peer.
func (t *inMemTransport) clearFault(peer string) {
	t.mu.Lock()
	delete(t.faults, peer)
	t.mu.Unlock()
}

// Replicate implements the async Transport by delivering msg to peer's Receive,
// applying any injected fault first, and reporting the outcome through done.
// It never returns a submission error (the in-memory path cannot fail to
// submit); completion is always via done, except under a partition where done
// is deliberately never invoked (see the type doc).
func (t *inMemTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	t.mu.Lock()
	eng := t.peers[peer]
	f := t.faults[peer]
	t.mu.Unlock()

	switch {
	case f.partition:
		// Never invoke done — the engine's Propose ctx times the record out.
		return nil
	case f.drop:
		done(AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}, nil)
		return nil
	}

	if f.delay > 0 {
		// Deliver after the delay from a goroutine so Replicate stays non-blocking.
		go func() {
			time.Sleep(f.delay)
			done(ackFor(eng, f, msg), nil)
		}()
		return nil
	}

	// Non-delay faults / normal delivery may complete synchronously: done is cheap
	// and non-blocking, and the engine's callback only takes e.mu briefly.
	done(ackFor(eng, f, msg), nil)
	return nil
}

// ReplicateGroup implements the optional GroupTransport capability so engine
// tests exercise the grouped submit path. It evaluates each record
// through the SAME per-message fault/ack machinery as Replicate — an override
// or fault applies per record, exactly as if the records had been shipped
// individually — and synthesizes the cumulative ack: the prefix that acked
// H6-exact counts, the first record that did not stops the group. Partition
// and drop apply to the whole frame (one frame on the wire), matching the net
// transport. msgs is copied before any asynchronous use (delay).
func (t *inMemTransport) ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error {
	t.mu.Lock()
	eng := t.peers[peer]
	f := t.faults[peer]
	t.mu.Unlock()

	switch {
	case f.partition:
		return nil // never invoke done — the Propose ctx times the records out
	case f.drop:
		done(AckMsg{Epoch: msgs[0].Epoch, Seq: msgs[0].Seq - 1, OK: false}, nil)
		return nil
	}

	if f.delay > 0 {
		cp := append([]ReplicateMsg(nil), msgs...)
		go func() {
			time.Sleep(f.delay)
			done(groupAckFor(eng, f, cp), nil)
		}()
		return nil
	}
	done(groupAckFor(eng, f, msgs), nil)
	return nil
}

var _ CatchupTransport = (*inMemTransport)(nil)

// CatchupRequest implements the grow handshake for engine unit tests: it
// answers with the target engine's CatchupInfo (its applied high-water). A
// partitioned target is unreachable (mirrors the net transport's dial failure);
// an unknown peer errors. epoch is informational — the growing primary fences on
// the returned AckMsg.Epoch itself.
func (t *inMemTransport) CatchupRequest(peer string, epoch uint64) (CatchupInfoMsg, error) {
	t.mu.Lock()
	eng := t.peers[peer]
	f := t.faults[peer]
	t.mu.Unlock()
	if f.partition {
		return CatchupInfoMsg{}, errInMemPeerUnreachable
	}
	if eng == nil {
		return CatchupInfoMsg{}, errInMemPeerUnreachable
	}
	return eng.CatchupInfo(), nil
}

var _ SnapshotTransport = (*inMemTransport)(nil)

// SendSnapshotChunk implements the snapshot transfer for engine unit
// tests: it delivers the chunk straight to the target engine's
// ReceiveSnapshotChunk. A partitioned or unknown target is unreachable, mirroring
// the net transport's dial failure. The chunk's Data is COPIED before delivery so
// the receiver's staging buffer can never alias the sender's blob (the net
// transport copies through the wire; an in-memory shortcut that aliased would
// hide a real bug class).
func (t *inMemTransport) SendSnapshotChunk(peer string, c SnapshotChunk) (AckMsg, error) {
	t.mu.Lock()
	eng := t.peers[peer]
	f := t.faults[peer]
	t.mu.Unlock()
	if f.partition || eng == nil {
		return AckMsg{}, errInMemPeerUnreachable
	}
	if f.drop {
		return AckMsg{Epoch: c.Epoch, Seq: c.FrontierSeq, OK: false}, nil
	}
	c.Data = append([]byte(nil), c.Data...)
	return eng.ReceiveSnapshotChunk(c), nil
}

// errInMemPeerUnreachable models a dial/RPC failure toward an unknown or
// partitioned peer on the in-memory transport (the net transport surfaces the
// underlying dial error here).
var errInMemPeerUnreachable = errors.New("pbisr: in-mem peer unreachable")

// groupAckFor folds per-record acks into the cumulative group ack.
func groupAckFor(eng *Engine, f peerFault, msgs []ReplicateMsg) AckMsg {
	lastOK := msgs[0].Seq - 1
	for i := range msgs {
		a := ackFor(eng, f, msgs[i])
		if !a.OK || a.Epoch != msgs[i].Epoch || a.Seq != msgs[i].Seq {
			return AckMsg{Epoch: msgs[0].Epoch, Seq: lastOK, OK: false}
		}
		lastOK = msgs[i].Seq
	}
	return AckMsg{Epoch: msgs[0].Epoch, Seq: lastOK, OK: true}
}

// ackFor computes the ack for a delivered message: an override wins; an unknown
// peer nacks (unreachable); otherwise the peer's Receive path decides.
func ackFor(eng *Engine, f peerFault, msg ReplicateMsg) AckMsg {
	if f.ackOverride != nil {
		return f.ackOverride(msg)
	}
	if eng == nil {
		return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}
	}
	return eng.Receive(msg)
}
