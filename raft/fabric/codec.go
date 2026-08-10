// SPDX-License-Identifier: Apache-2.0

// Package fabric is a multiplexed, batching Raft transport for Rostam. It
// replaces the per-group hashicorp/raft NetworkTransport (one TCP socket per
// Raft group per peer, msgpack RPC codec) with a shared, batching,
// zero-reflection transport that carries every group's traffic to a peer over a
// single long-lived connection. It attacks the two costs the CPU profile found
// on the replicated-write hot path: inter-node network syscalls (per-group
// sockets, tiny writes) and msgpack encode/decode.
//
// The package is flag-gated behind cluster.Config.RaftTransport=="fabric"; the
// default ("mux") path is unchanged. See DESIGN.md in this directory.
package fabric

import (
	"encoding/binary"
	"errors"
	"time"

	hraft "github.com/hashicorp/raft"
)

// errShortPayload marks a truncated / malformed RPC payload.
var errShortPayload = errors.New("fabric: short payload")

// maxEntriesGuard bounds the AppendEntries entry count decoded from a frame so a
// corrupt length cannot drive an unbounded allocation. Raft batches are far
// smaller than this in practice.
const maxEntriesGuard = 1 << 20

// zeroTime is the canonical zero Log.AppendedAt used on decode.
var zeroTime = time.Time{}

// unixNano mirrors raft/logstore's timestamp decode so a Log round-trips
// bit-for-bit through both the WAL and the wire.
func unixNano(n int64) time.Time { return time.Unix(0, n) }

// decbuf is a big-endian cursor decoder over a payload slice. Every read is
// bounds-checked; the first failure latches err and all subsequent reads are
// no-ops returning zero. Callers check err once at the end.
type decbuf struct {
	b   []byte
	off int
	err error
}

func (d *decbuf) need(n int) bool {
	if d.err != nil {
		return false
	}
	if n < 0 || d.off+n > len(d.b) {
		d.err = errShortPayload
		return false
	}
	return true
}

func (d *decbuf) u8() uint8 {
	if !d.need(1) {
		return 0
	}
	v := d.b[d.off]
	d.off++
	return v
}

func (d *decbuf) u32() uint32 {
	if !d.need(4) {
		return 0
	}
	v := binary.BigEndian.Uint32(d.b[d.off:])
	d.off += 4
	return v
}

func (d *decbuf) u64() uint64 {
	if !d.need(8) {
		return 0
	}
	v := binary.BigEndian.Uint64(d.b[d.off:])
	d.off += 8
	return v
}

func (d *decbuf) i64() int64 { return int64(d.u64()) } //nolint:gosec // round-trips

// bytes reads a [u32 len][bytes] slice, copying it into fresh memory (the
// underlying frame buffer is reused). A zero length decodes to nil so a nil
// input round-trips (an empty slice is indistinguishable on the wire).
func (d *decbuf) bytes() []byte {
	n := d.u32()
	if d.err != nil {
		return nil
	}
	if !d.need(int(n)) {
		return nil
	}
	if n == 0 {
		return nil
	}
	out := make([]byte, n)
	copy(out, d.b[d.off:d.off+int(n)])
	d.off += int(n)
	return out
}

// putBytes appends [u32 len][bytes].
func putBytes(b []byte, v []byte) []byte {
	b = binary.BigEndian.AppendUint32(b, uint32(len(v))) //nolint:gosec // lengths bounded upstream
	return append(b, v...)
}

// --- RPCHeader ---

// [u8 ProtocolVersion][bytes ID][bytes Addr]
func encodeRPCHeader(b []byte, h hraft.RPCHeader) []byte {
	b = append(b, uint8(h.ProtocolVersion)) //nolint:gosec // small enum
	b = putBytes(b, h.ID)
	b = putBytes(b, h.Addr)
	return b
}

func (d *decbuf) rpcHeader() hraft.RPCHeader {
	var h hraft.RPCHeader
	h.ProtocolVersion = hraft.ProtocolVersion(d.u8())
	h.ID = d.bytes()
	h.Addr = d.bytes()
	return h
}

// --- Log ---

// [u64 Index][u64 Term][u8 Type][bytes Data][bytes Extensions][i64 AppendedAtUnixNano]
func encodeLog(b []byte, l *hraft.Log) []byte {
	b = binary.BigEndian.AppendUint64(b, l.Index)
	b = binary.BigEndian.AppendUint64(b, l.Term)
	b = append(b, byte(l.Type))
	b = putBytes(b, l.Data)
	b = putBytes(b, l.Extensions)
	var nano int64
	if !l.AppendedAt.IsZero() {
		nano = l.AppendedAt.UnixNano()
	}
	b = binary.BigEndian.AppendUint64(b, uint64(nano)) //nolint:gosec // round-trips
	return b
}

func (d *decbuf) log(out *hraft.Log) {
	out.Index = d.u64()
	out.Term = d.u64()
	out.Type = hraft.LogType(d.u8())
	out.Data = d.bytes()
	out.Extensions = d.bytes()
	nano := d.i64()
	if nano == 0 {
		out.AppendedAt = zeroTime
	} else {
		out.AppendedAt = unixNano(nano)
	}
}

// --- response payload prefix: [u8 appErr][bytes err] ---

// putRespPrefix writes the application-error prefix. appErr=="" is the success
// case (flag 0, empty error). A non-empty appErr sets the flag and carries the
// error string; the waiter surfaces it instead of decoding a struct.
func putRespPrefix(b []byte, appErr string) []byte {
	if appErr == "" {
		b = append(b, 0)
		return putBytes(b, nil)
	}
	b = append(b, 1)
	return putBytes(b, []byte(appErr))
}

// respPrefix reads the prefix and reports the carried application error string
// ("" when none). The cursor is left positioned at the struct payload.
func (d *decbuf) respPrefix() string {
	flag := d.u8()
	msg := d.bytes()
	if flag == 0 {
		return ""
	}
	return string(msg)
}

// --- AppendEntries ---

func encodeAppendEntriesRequest(b []byte, r *hraft.AppendEntriesRequest) []byte {
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = putBytes(b, r.Leader) //nolint:staticcheck // SA1019: codec must round-trip what the raft lib populates
	b = binary.BigEndian.AppendUint64(b, r.PrevLogEntry)
	b = binary.BigEndian.AppendUint64(b, r.PrevLogTerm)
	b = binary.BigEndian.AppendUint32(b, uint32(len(r.Entries))) //nolint:gosec // bounded upstream
	for _, e := range r.Entries {
		b = encodeLog(b, e)
	}
	b = binary.BigEndian.AppendUint64(b, r.LeaderCommitIndex)
	return b
}

func decodeAppendEntriesRequest(payload []byte, out *hraft.AppendEntriesRequest) error {
	d := decbuf{b: payload}
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.Leader = d.bytes() //nolint:staticcheck // SA1019: codec must round-trip what the raft lib populates
	out.PrevLogEntry = d.u64()
	out.PrevLogTerm = d.u64()
	n := d.u32()
	if d.err != nil {
		return d.err
	}
	if !d.need(0) || n > maxEntriesGuard {
		return errShortPayload
	}
	if n == 0 {
		out.Entries = nil
	} else {
		out.Entries = make([]*hraft.Log, n)
		for i := range out.Entries {
			var l hraft.Log
			d.log(&l)
			out.Entries[i] = &l
		}
	}
	out.LeaderCommitIndex = d.u64()
	return d.err
}

func encodeAppendEntriesResponse(b []byte, appErr string, r *hraft.AppendEntriesResponse) []byte {
	b = putRespPrefix(b, appErr)
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = binary.BigEndian.AppendUint64(b, r.LastLog)
	b = append(b, boolByte(r.Success), boolByte(r.NoRetryBackoff))
	return b
}

func decodeAppendEntriesResponse(payload []byte, out *hraft.AppendEntriesResponse) (appErr string, err error) {
	d := decbuf{b: payload}
	appErr = d.respPrefix()
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.LastLog = d.u64()
	out.Success = d.u8() != 0
	out.NoRetryBackoff = d.u8() != 0
	return appErr, d.err
}

// --- RequestVote ---

func encodeRequestVoteRequest(b []byte, r *hraft.RequestVoteRequest) []byte {
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = putBytes(b, r.Candidate) //nolint:staticcheck // SA1019: codec must round-trip what the raft lib populates
	b = binary.BigEndian.AppendUint64(b, r.LastLogIndex)
	b = binary.BigEndian.AppendUint64(b, r.LastLogTerm)
	b = append(b, boolByte(r.LeadershipTransfer))
	return b
}

func decodeRequestVoteRequest(payload []byte, out *hraft.RequestVoteRequest) error {
	d := decbuf{b: payload}
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.Candidate = d.bytes() //nolint:staticcheck // SA1019: codec must round-trip what the raft lib populates
	out.LastLogIndex = d.u64()
	out.LastLogTerm = d.u64()
	out.LeadershipTransfer = d.u8() != 0
	return d.err
}

func encodeRequestVoteResponse(b []byte, appErr string, r *hraft.RequestVoteResponse) []byte {
	b = putRespPrefix(b, appErr)
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = putBytes(b, r.Peers)
	b = append(b, boolByte(r.Granted))
	return b
}

func decodeRequestVoteResponse(payload []byte, out *hraft.RequestVoteResponse) (appErr string, err error) {
	d := decbuf{b: payload}
	appErr = d.respPrefix()
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.Peers = d.bytes()
	out.Granted = d.u8() != 0
	return appErr, d.err
}

// --- RequestPreVote ---

func encodeRequestPreVoteRequest(b []byte, r *hraft.RequestPreVoteRequest) []byte {
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = binary.BigEndian.AppendUint64(b, r.LastLogIndex)
	b = binary.BigEndian.AppendUint64(b, r.LastLogTerm)
	return b
}

func decodeRequestPreVoteRequest(payload []byte, out *hraft.RequestPreVoteRequest) error {
	d := decbuf{b: payload}
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.LastLogIndex = d.u64()
	out.LastLogTerm = d.u64()
	return d.err
}

func encodeRequestPreVoteResponse(b []byte, appErr string, r *hraft.RequestPreVoteResponse) []byte {
	b = putRespPrefix(b, appErr)
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = append(b, boolByte(r.Granted))
	return b
}

func decodeRequestPreVoteResponse(payload []byte, out *hraft.RequestPreVoteResponse) (appErr string, err error) {
	d := decbuf{b: payload}
	appErr = d.respPrefix()
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.Granted = d.u8() != 0
	return appErr, d.err
}

// --- TimeoutNow ---

func encodeTimeoutNowRequest(b []byte, r *hraft.TimeoutNowRequest) []byte {
	return encodeRPCHeader(b, r.RPCHeader)
}

func decodeTimeoutNowRequest(payload []byte, out *hraft.TimeoutNowRequest) error {
	d := decbuf{b: payload}
	out.RPCHeader = d.rpcHeader()
	return d.err
}

func encodeTimeoutNowResponse(b []byte, appErr string, r *hraft.TimeoutNowResponse) []byte {
	b = putRespPrefix(b, appErr)
	return encodeRPCHeader(b, r.RPCHeader)
}

func decodeTimeoutNowResponse(payload []byte, out *hraft.TimeoutNowResponse) (appErr string, err error) {
	d := decbuf{b: payload}
	appErr = d.respPrefix()
	out.RPCHeader = d.rpcHeader()
	return appErr, d.err
}

// --- InstallSnapshot (dedicated conn; snapshot bytes streamed separately) ---

func encodeInstallSnapshotRequest(b []byte, r *hraft.InstallSnapshotRequest) []byte {
	b = encodeRPCHeader(b, r.RPCHeader)
	b = append(b, uint8(r.SnapshotVersion)) //nolint:gosec // small enum
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = putBytes(b, r.Leader)
	b = binary.BigEndian.AppendUint64(b, r.LastLogIndex)
	b = binary.BigEndian.AppendUint64(b, r.LastLogTerm)
	b = putBytes(b, r.Peers)
	b = putBytes(b, r.Configuration)
	b = binary.BigEndian.AppendUint64(b, r.ConfigurationIndex)
	b = binary.BigEndian.AppendUint64(b, uint64(r.Size)) //nolint:gosec // round-trips
	return b
}

func decodeInstallSnapshotRequest(payload []byte, out *hraft.InstallSnapshotRequest) error {
	d := decbuf{b: payload}
	out.RPCHeader = d.rpcHeader()
	out.SnapshotVersion = hraft.SnapshotVersion(d.u8())
	out.Term = d.u64()
	out.Leader = d.bytes()
	out.LastLogIndex = d.u64()
	out.LastLogTerm = d.u64()
	out.Peers = d.bytes()
	out.Configuration = d.bytes()
	out.ConfigurationIndex = d.u64()
	out.Size = d.i64()
	return d.err
}

func encodeInstallSnapshotResponse(b []byte, appErr string, r *hraft.InstallSnapshotResponse) []byte {
	b = putRespPrefix(b, appErr)
	b = encodeRPCHeader(b, r.RPCHeader)
	b = binary.BigEndian.AppendUint64(b, r.Term)
	b = append(b, boolByte(r.Success))
	return b
}

func decodeInstallSnapshotResponse(payload []byte, out *hraft.InstallSnapshotResponse) (appErr string, err error) {
	d := decbuf{b: payload}
	appErr = d.respPrefix()
	out.RPCHeader = d.rpcHeader()
	out.Term = d.u64()
	out.Success = d.u8() != 0
	return appErr, d.err
}

func boolByte(v bool) byte {
	if v {
		return 1
	}
	return 0
}
