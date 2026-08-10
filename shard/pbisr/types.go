// SPDX-License-Identifier: Apache-2.0

package pbisr

// Wire messages for the primary-backup / ISR data-plane replication protocol.
//
// These are plain structs with no codec attached yet: the engine builds the
// correctness core against an in-memory transport, and the fabric-frame
// encoding is wired in a later step. The pair (Epoch, Seq) totally orders
// writes and replaces Raft's (term, index) — see shard/pbisr/DESIGN.md §Model.

// ReplicateMsg is one write shipped from the primary to an ISR backup.
//
//   - Epoch     — the primary's leadership generation. A backup that has adopted
//     a higher epoch rejects this message (H1/H5 fencing).
//   - Seq       — the primary-assigned monotonic sequence for this write.
//   - PrevSeq   — the seq of this write's PREDECESSOR (Seq-1).
//   - PrevEpoch — the EPOCH of that predecessor, i.e. the epoch under which the
//     write at PrevSeq was assigned. Together (PrevSeq, PrevEpoch) name the exact
//     history this write extends, and the receiver accepts only if it already
//     holds that exact predecessor — the LOG MATCHING property, the
//     direct analogue of Raft's (prevLogIndex, prevLogTerm).
//   - Data      — the opaque op payload handed to the Applier.
//
// WHY PrevSeq ALONE IS NOT ENOUGH. Promote resets the seq counter to the new
// primary's applied high-water, so seq numbers are REUSED across epochs with
// DIFFERENT content. A bare PrevSeq is therefore a position, not an identity: two
// nodes can both hold "seq 3" and mean different writes. Pairing it with the
// predecessor's epoch turns the chain link into a real history proof — the
// receiver can distinguish "I am a prefix of you" from "we forked".
type ReplicateMsg struct {
	Epoch     uint64
	Seq       uint64
	PrevSeq   uint64
	PrevEpoch uint64
	Data      []byte
}

// CatchupInfoMsg answers a learner-catch-up handshake (ISR grow) with the
// responder's log identity. It is deliberately a SEPARATE type from AckMsg,
// because the handshake answers two DIFFERENT questions that used to be conflated
// into AckMsg.Seq:
//
//   - AppliedSeq is the node's applied high-water AS A BACKUP (lastApplied). It is
//     what the failover promotion gate reads (cluster's pbCandidateHighWater): "how
//     much of the committed tail did you receive as a replica".
//   - FrontierSeq/FrontierEpoch are the node's LOG IDENTITY: the (seq, epoch) of
//     the newest write it holds in EITHER role — max(lastSeq, lastApplied) and that
//     write's epoch. This is what a grow must resume from, because a node that
//     PROPOSED as primary holds writes its lastApplied never counted.
//
// Epoch is the responder's fencing watermark, unchanged in meaning: the growing
// primary aborts if it exceeds the epoch it is growing at.
type CatchupInfoMsg struct {
	Epoch         uint64
	AppliedSeq    uint64
	FrontierSeq   uint64
	FrontierEpoch uint64
	OK            bool
}

// AckMsg is a backup's response to a single ReplicateMsg.
//
// An ack counts toward a write's durability floor ONLY when it is OK and its
// (Epoch, Seq) match that exact write — a liveness signal or an ack for a
// different seq is never a durability signal (H6). See Engine.Propose.
type AckMsg struct {
	Epoch uint64
	Seq   uint64
	OK    bool
}
