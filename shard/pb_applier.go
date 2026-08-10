// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"encoding/binary"
	"errors"
)

// shardApplier adapts the shard FSM's op dispatch to the pbisr.Applier interface.
// The primary-backup engine calls Apply on both the primary (authoritative apply)
// and each backup (materialize the replicated write). It runs through the exact
// same op path as the Raft FSM (fsm.applyEntryData), so an op behaves identically
// under either replication mode.
type shardApplier struct {
	f *fsm
}

func newShardApplier(f *fsm) *shardApplier { return &shardApplier{f: f} }

// errPBInfra distinguishes an infrastructure failure (decode / unregistered /
// read-only-through-write) — which must abort the write before a seq is assigned
// — from an op-level business error, which is encoded into the result bytes and
// still replicated (matching Raft, where such errors ride in ApplyResponse.Err
// on a committed entry).
var errPBInfra = errors.New("pbisr apply: infrastructure failure")

// Apply runs the encoded entry through the FSM. It returns encodePBResult bytes
// and a nil error for a successful apply OR a DETERMINISTIC op-level error (which
// rides in the result frame and is still replicated, matching Raft). It returns a
// non-nil error — the errPBInfra sentinel — for either an infrastructure failure
// OR a classFatal NON-DETERMINISTIC apply error (cache.ErrFull / ErrCannotEvict),
// so the PB engine aborts instead of falsely committing (production-readiness #4).
//
// The engine treats this non-nil return identically in both roles, which is the
// fail-closed behavior we want — trading availability for consistency:
//   - Primary (engine.go proposeSequenced step (d)): apply runs BEFORE the seq is
//     assigned; a non-nil error unwinds with NO seq burned, so the client sees the
//     error and no phantom seq gap-wedges future writes.
//   - Backup (engine.go receiveLocked step (c)): a non-nil error NACKs (OK:false)
//     WITHOUT advancing lastApplied. The DIRECTION is correct — a NACK-and-stall is
//     strictly better than advancing into a silent divergence. But note: dynamic
//     ISR membership is not yet implemented (ISR shrink / drop-out / catch-up /
//     snapshot reconvergence are future work — see ErrPBUnimplemented in
//     pb_replicator.go), so TODAY a persistently-full backup WEDGES the shard's
//     writes under full-ISR commit (the write cannot reach the full ISR) rather
//     than auto-dropping-out and reconverging. That auto-reconvergence is the
//     intended end state; the fail-closed NACK is the safe interim behavior.
//
// Note: unlike the Raft FSM path, there is no isReplicated gate here — the PB
// engine IS the replication path, and aborting on ErrFull is correct even for a
// single-node PB store (burn no seq, surface the error to the client; no os.Exit).
//
// ############ classRetry IS A RETRY HERE, NOT A NACK. STATE BOTH ROLES. ######
//
// A classRetry outcome (ops.ErrWASMModuleNotResident — this node holds a correct
// binding to a module version whose bytes have not arrived) must NOT become
// errPBInfra, and the reason is narrow enough to be worth writing down: a NACK
// wedges the shard ANYWAY under full-ISR commit, but does so from INSIDE the
// engine's receiveLocked, i.e. while holding the engine lock. It converts a
// condition that resolves itself into one that also blocks every other engine
// operation on the way. Retrying here wedges no more than the NACK would and
// releases the moment the blob lands.
//
// ON A BACKUP the consequence is exactly the NACK's, minus the lock: the
// replication stream for this shard stalls at this seq, and because dynamic ISR
// membership is not implemented (see the note above), a full-ISR commit cannot be
// reached, so the shard's writes wedge until the bytes arrive. That is the same
// outcome today's NACK produces, and it is accepted for the same reason — a stall
// is strictly better than advancing into a silent divergence — with the
// difference that this one clears without operator action, and clears IMMEDIATELY
// if an operator hands the bytes over with __wasm_blob_put__.
//
// ON A PRIMARY it is SHOULD-NOT-HAPPEN. The primary is the proposer: it accepted
// the registration, which means it stored the module locally and compile-verified
// it before anything was sequenced (cluster.pushWASMBlob), and the blob store is
// never pruned (retirement is not built). So a primary reaching this line has lost
// bytes it is required to hold — a data-dir problem, not a replication one. It
// retries rather than failing the write because there is no better answer
// available on this path: the write cannot be executed, so it cannot be
// acknowledged, and the block is at least visible and self-healing. The
// observability hook fires identically in both roles, and a block reported on a
// primary should be read as a local storage fault.
func (a *shardApplier) Apply(data []byte) ([]byte, error) {
	// The retry loop lives in the FSM (applyWithRetry) so both replication modes
	// share ONE waiting/backoff/observability path — an op must behave identically
	// under either, and a second, subtly-different block here is exactly how that
	// stops being true. There is no prefix to record: the PB engine applies one
	// entry at a time, so nothing precedes this one to lose.
	resp, applied := a.f.applyWithRetry(data, nil)
	if !applied {
		// Store closing while still blocked. Abort rather than commit: on a primary
		// this unwinds with no seq burned, on a backup it NACKs without advancing
		// lastApplied. Either way nothing is recorded for an entry that never ran.
		return nil, errPBInfra
	}
	if resp.Err != nil && (isInfraError(resp.Err) || classifyApplyErr(resp.Err) == classFatal) {
		return nil, errPBInfra
	}
	return encodePBResult(resp.Result, resp.Err), nil
}

// isInfraError reports whether an ApplyResponse.Err is an infrastructure failure
// (must abort) rather than an op-level business error (must still commit). The
// FSM path produces only ErrOpNotRegistered and wrapped decode/read-only errors
// for infra failures; op handlers return their own domain errors.
func isInfraError(err error) bool {
	return errors.Is(err, ErrOpNotRegistered) ||
		errors.Is(err, errPBApplyDecode) ||
		errors.Is(err, errPBApplyReadOnly)
}

// Sentinels mirrored from fsm.applyEntryData's infra-error wraps so isInfraError
// can classify without string matching. fsm.applyEntryData must wrap with these.
var (
	errPBApplyDecode   = errors.New("apply decode")
	errPBApplyReadOnly = errors.New("apply: read-only op through write path")
)

// encodePBResult serializes (result, opErr) into a self-describing frame:
//
//	[1 byte hasErr][4 byte len][payload]
//
// where payload is the result bytes (hasErr=0) or the error string (hasErr=1).
func encodePBResult(result []byte, opErr error) []byte {
	if opErr != nil {
		msg := []byte(opErr.Error())
		out := make([]byte, 5+len(msg))
		out[0] = 1
		binary.LittleEndian.PutUint32(out[1:5], uint32(len(msg)))
		copy(out[5:], msg)
		return out
	}
	out := make([]byte, 5+len(result))
	out[0] = 0
	binary.LittleEndian.PutUint32(out[1:5], uint32(len(result)))
	copy(out[5:], result)
	return out
}

// decodePBResult is the inverse of encodePBResult, rebuilding the *ApplyResponse
// the Store's applyOpIndexed expects.
//
// Note: an op error's IDENTITY is NOT preserved across this codec — it is
// rebuilt as errors.New(string(...)), so a typed/errors.Is-able sentinel a
// handler returns becomes an opaque string on the PB path (unlike Raft mode,
// where ApplyResponse.Err is the live object). Fine for current ops
// (Put/Del/Expire don't errors.Is the handler error), but revisit before
// adding an op that relies on error identity.
func decodePBResult(b []byte) *ApplyResponse {
	if len(b) < 5 {
		return &ApplyResponse{Err: errors.New("pbisr: short result frame")}
	}
	n := binary.LittleEndian.Uint32(b[1:5])
	if int(n) > len(b)-5 {
		return &ApplyResponse{Err: errors.New("pbisr: truncated result frame")}
	}
	payload := b[5 : 5+n]
	if b[0] == 1 {
		return &ApplyResponse{Err: errors.New(string(payload))}
	}
	out := make([]byte, len(payload))
	copy(out, payload)
	return &ApplyResponse{Result: out}
}
