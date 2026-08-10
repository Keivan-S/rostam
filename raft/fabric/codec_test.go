// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"reflect"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"
)

// norm mirrors the decoder's empty→nil convention so expected structs compare
// equal to decoded ones under reflect.DeepEqual (a zero-length slice is
// indistinguishable from nil on the wire).
func norm(b []byte) []byte {
	if len(b) == 0 {
		return nil
	}
	return b
}

// mkTime builds an AppendedAt that survives the round-trip: 0 → the zero Time
// (encoded as 0), otherwise time.Unix(0, nano) matching the decoder exactly.
func mkTime(nano int64) time.Time {
	if nano == 0 {
		return time.Time{}
	}
	return time.Unix(0, nano)
}

func hdr(pv uint8, id, addr []byte) hraft.RPCHeader {
	return hraft.RPCHeader{
		ProtocolVersion: hraft.ProtocolVersion(pv),
		ID:              norm(id),
		Addr:            norm(addr),
	}
}

func mkLog(index, term uint64, typ uint8, data, ext []byte, nano int64) *hraft.Log {
	return &hraft.Log{
		Index:      index,
		Term:       term,
		Type:       hraft.LogType(typ),
		Data:       norm(data),
		Extensions: norm(ext),
		AppendedAt: mkTime(nano),
	}
}

// --- AppendEntries ---

func TestAppendEntriesRequestRoundTrip(t *testing.T) {
	unicodeAddr := []byte("réseau-nœud-Ω:7400")
	cases := []hraft.AppendEntriesRequest{
		{}, // fully zero
		{
			RPCHeader:         hdr(3, []byte("node-1"), unicodeAddr),
			Term:              9,
			Leader:            unicodeAddr,
			PrevLogEntry:      100,
			PrevLogTerm:       8,
			Entries:           nil,
			LeaderCommitIndex: 99,
		},
		{ // heartbeat-shaped
			RPCHeader: hdr(3, []byte("n"), []byte("a")),
			Term:      5,
			Leader:    []byte("a"),
		},
		{ // multi-entry incl. empty and large Data
			RPCHeader:    hdr(2, []byte("leader"), unicodeAddr),
			Term:         12,
			Leader:       []byte("leader"),
			PrevLogEntry: 7,
			PrevLogTerm:  11,
			Entries: []*hraft.Log{
				mkLog(8, 11, uint8(hraft.LogCommand), nil, nil, 0),
				mkLog(9, 11, uint8(hraft.LogCommand), []byte("hello"), []byte("ext"), 1_700_000_000_000_000_000),
				mkLog(10, 12, uint8(hraft.LogNoop), make([]byte, 64<<10), nil, 42),
			},
			LeaderCommitIndex: 6,
		},
	}
	for i, in := range cases {
		var out hraft.AppendEntriesRequest
		if err := decodeAppendEntriesRequest(encodeAppendEntriesRequest(nil, &in), &out); err != nil {
			t.Fatalf("case %d: decode: %v", i, err)
		}
		if !reflect.DeepEqual(in, out) {
			t.Fatalf("case %d: mismatch\n in=%#v\nout=%#v", i, in, out)
		}
	}
}

func TestAppendEntriesResponseRoundTrip(t *testing.T) {
	in := hraft.AppendEntriesResponse{
		RPCHeader:      hdr(3, []byte("id"), []byte("addr")),
		Term:           7,
		LastLog:        42,
		Success:        true,
		NoRetryBackoff: true,
	}
	var out hraft.AppendEntriesResponse
	appErr, err := decodeAppendEntriesResponse(encodeAppendEntriesResponse(nil, "", &in), &out)
	if err != nil || appErr != "" {
		t.Fatalf("decode: err=%v appErr=%q", err, appErr)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("mismatch\n in=%#v\nout=%#v", in, out)
	}
	// Application-error path: the struct is ignored, the error string surfaces.
	appErr, err = decodeAppendEntriesResponse(encodeAppendEntriesResponse(nil, "not the leader", &hraft.AppendEntriesResponse{}), &out)
	if err != nil {
		t.Fatalf("decode err: %v", err)
	}
	if appErr != "not the leader" {
		t.Fatalf("appErr = %q, want %q", appErr, "not the leader")
	}
}

// --- RequestVote ---

func TestRequestVoteRoundTrip(t *testing.T) {
	req := hraft.RequestVoteRequest{
		RPCHeader:          hdr(3, []byte("cand"), []byte("cand-addr")),
		Term:               4,
		Candidate:          []byte("cand-addr"),
		LastLogIndex:       10,
		LastLogTerm:        3,
		LeadershipTransfer: true,
	}
	var rout hraft.RequestVoteRequest
	if err := decodeRequestVoteRequest(encodeRequestVoteRequest(nil, &req), &rout); err != nil {
		t.Fatalf("decode req: %v", err)
	}
	if !reflect.DeepEqual(req, rout) {
		t.Fatalf("req mismatch\n in=%#v\nout=%#v", req, rout)
	}

	resp := hraft.RequestVoteResponse{
		RPCHeader: hdr(0, nil, nil),
		Term:      4,
		Peers:     []byte("peerblob"),
		Granted:   true,
	}
	var pout hraft.RequestVoteResponse
	appErr, err := decodeRequestVoteResponse(encodeRequestVoteResponse(nil, "", &resp), &pout)
	if err != nil || appErr != "" {
		t.Fatalf("decode resp: err=%v appErr=%q", err, appErr)
	}
	if !reflect.DeepEqual(resp, pout) {
		t.Fatalf("resp mismatch\n in=%#v\nout=%#v", resp, pout)
	}
}

// --- RequestPreVote ---

func TestRequestPreVoteRoundTrip(t *testing.T) {
	req := hraft.RequestPreVoteRequest{
		RPCHeader:    hdr(3, []byte("id"), []byte("addr")),
		Term:         6,
		LastLogIndex: 20,
		LastLogTerm:  5,
	}
	var rout hraft.RequestPreVoteRequest
	if err := decodeRequestPreVoteRequest(encodeRequestPreVoteRequest(nil, &req), &rout); err != nil {
		t.Fatalf("decode req: %v", err)
	}
	if !reflect.DeepEqual(req, rout) {
		t.Fatalf("req mismatch\n in=%#v\nout=%#v", req, rout)
	}

	resp := hraft.RequestPreVoteResponse{
		RPCHeader: hdr(3, []byte("id"), []byte("addr")),
		Term:      6,
		Granted:   false,
	}
	var pout hraft.RequestPreVoteResponse
	appErr, err := decodeRequestPreVoteResponse(encodeRequestPreVoteResponse(nil, "", &resp), &pout)
	if err != nil || appErr != "" {
		t.Fatalf("decode resp: err=%v appErr=%q", err, appErr)
	}
	if !reflect.DeepEqual(resp, pout) {
		t.Fatalf("resp mismatch\n in=%#v\nout=%#v", resp, pout)
	}
}

// --- TimeoutNow ---

func TestTimeoutNowRoundTrip(t *testing.T) {
	req := hraft.TimeoutNowRequest{RPCHeader: hdr(3, []byte("id"), []byte("addr"))}
	var rout hraft.TimeoutNowRequest
	if err := decodeTimeoutNowRequest(encodeTimeoutNowRequest(nil, &req), &rout); err != nil {
		t.Fatalf("decode req: %v", err)
	}
	if !reflect.DeepEqual(req, rout) {
		t.Fatalf("req mismatch\n in=%#v\nout=%#v", req, rout)
	}

	resp := hraft.TimeoutNowResponse{RPCHeader: hdr(3, []byte("id"), []byte("addr"))}
	var pout hraft.TimeoutNowResponse
	appErr, err := decodeTimeoutNowResponse(encodeTimeoutNowResponse(nil, "", &resp), &pout)
	if err != nil || appErr != "" {
		t.Fatalf("decode resp: err=%v appErr=%q", err, appErr)
	}
	if !reflect.DeepEqual(resp, pout) {
		t.Fatalf("resp mismatch\n in=%#v\nout=%#v", resp, pout)
	}
}

// --- InstallSnapshot ---

func TestInstallSnapshotRoundTrip(t *testing.T) {
	req := hraft.InstallSnapshotRequest{
		RPCHeader:          hdr(3, []byte("id"), []byte("addr")),
		SnapshotVersion:    1,
		Term:               8,
		Leader:             []byte("leader-addr"),
		LastLogIndex:       500,
		LastLogTerm:        7,
		Peers:              []byte("peers"),
		Configuration:      []byte("config-blob"),
		ConfigurationIndex: 123,
		Size:               1 << 20,
	}
	var rout hraft.InstallSnapshotRequest
	if err := decodeInstallSnapshotRequest(encodeInstallSnapshotRequest(nil, &req), &rout); err != nil {
		t.Fatalf("decode req: %v", err)
	}
	if !reflect.DeepEqual(req, rout) {
		t.Fatalf("req mismatch\n in=%#v\nout=%#v", req, rout)
	}

	resp := hraft.InstallSnapshotResponse{
		RPCHeader: hdr(3, []byte("id"), []byte("addr")),
		Term:      8,
		Success:   true,
	}
	var pout hraft.InstallSnapshotResponse
	appErr, err := decodeInstallSnapshotResponse(encodeInstallSnapshotResponse(nil, "", &resp), &pout)
	if err != nil || appErr != "" {
		t.Fatalf("decode resp: err=%v appErr=%q", err, appErr)
	}
	if !reflect.DeepEqual(resp, pout) {
		t.Fatalf("resp mismatch\n in=%#v\nout=%#v", resp, pout)
	}
}

// --- truncation safety ---

func TestDecodeTruncatedIsError(t *testing.T) {
	full := encodeAppendEntriesRequest(nil, &hraft.AppendEntriesRequest{
		RPCHeader: hdr(3, []byte("id"), []byte("addr")),
		Term:      1,
		Leader:    []byte("addr"),
		Entries:   []*hraft.Log{mkLog(1, 1, 0, []byte("d"), nil, 1)},
	})
	for n := 0; n < len(full); n++ {
		var out hraft.AppendEntriesRequest
		if err := decodeAppendEntriesRequest(full[:n], &out); err == nil {
			t.Fatalf("truncation to %d bytes decoded without error", n)
		}
	}
}

// --- fuzz round-trips ---

func FuzzAppendEntriesRequest(f *testing.F) {
	f.Add(uint8(3), []byte("id"), []byte("réseau:7400"), uint64(9), uint64(100), uint64(8), uint64(99), uint8(2), []byte("data"), []byte("ext"), int64(42))
	f.Fuzz(func(t *testing.T, pv uint8, id, addr []byte, term, prevIdx, prevTerm, commit uint64, nEntries uint8, data, ext []byte, nano int64) {
		in := hraft.AppendEntriesRequest{
			RPCHeader:         hdr(pv, id, addr),
			Term:              term,
			Leader:            norm(addr),
			PrevLogEntry:      prevIdx,
			PrevLogTerm:       prevTerm,
			LeaderCommitIndex: commit,
		}
		for i := 0; i < int(nEntries%4); i++ {
			in.Entries = append(in.Entries, mkLog(uint64(i), term, uint8(i)%6, data, ext, nano))
		}
		var out hraft.AppendEntriesRequest
		if err := decodeAppendEntriesRequest(encodeAppendEntriesRequest(nil, &in), &out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !reflect.DeepEqual(in, out) {
			t.Fatalf("mismatch\n in=%#v\nout=%#v", in, out)
		}
	})
}

func FuzzAppendEntriesResponse(f *testing.F) {
	f.Add([]byte("id"), []byte("addr"), uint64(7), uint64(42), true, false, "")
	f.Fuzz(func(t *testing.T, id, addr []byte, term, lastLog uint64, success, noRetry bool, appErrIn string) {
		in := hraft.AppendEntriesResponse{
			RPCHeader:      hdr(3, id, addr),
			Term:           term,
			LastLog:        lastLog,
			Success:        success,
			NoRetryBackoff: noRetry,
		}
		var out hraft.AppendEntriesResponse
		appErr, err := decodeAppendEntriesResponse(encodeAppendEntriesResponse(nil, appErrIn, &in), &out)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if appErr != appErrIn {
			t.Fatalf("appErr = %q, want %q", appErr, appErrIn)
		}
		if appErrIn == "" && !reflect.DeepEqual(in, out) {
			t.Fatalf("mismatch\n in=%#v\nout=%#v", in, out)
		}
	})
}

func FuzzRequestVoteRequest(f *testing.F) {
	f.Add(uint8(3), []byte("cand"), []byte("addr"), uint64(4), uint64(10), uint64(3), true)
	f.Fuzz(func(t *testing.T, pv uint8, cand, addr []byte, term, lli, llt uint64, transfer bool) {
		in := hraft.RequestVoteRequest{
			RPCHeader:          hdr(pv, addr, addr),
			Term:               term,
			Candidate:          norm(cand),
			LastLogIndex:       lli,
			LastLogTerm:        llt,
			LeadershipTransfer: transfer,
		}
		var out hraft.RequestVoteRequest
		if err := decodeRequestVoteRequest(encodeRequestVoteRequest(nil, &in), &out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !reflect.DeepEqual(in, out) {
			t.Fatalf("mismatch\n in=%#v\nout=%#v", in, out)
		}
	})
}

func FuzzInstallSnapshotRequest(f *testing.F) {
	f.Add(uint8(1), []byte("id"), []byte("addr"), uint64(8), []byte("peers"), []byte("cfg"), uint64(123), int64(1<<20))
	f.Fuzz(func(t *testing.T, sv uint8, id, addr []byte, term uint64, peers, cfg []byte, cfgIdx uint64, size int64) {
		in := hraft.InstallSnapshotRequest{
			RPCHeader:          hdr(3, id, addr),
			SnapshotVersion:    hraft.SnapshotVersion(sv),
			Term:               term,
			Leader:             norm(addr),
			LastLogIndex:       cfgIdx,
			LastLogTerm:        term,
			Peers:              norm(peers),
			Configuration:      norm(cfg),
			ConfigurationIndex: cfgIdx,
			Size:               size,
		}
		var out hraft.InstallSnapshotRequest
		if err := decodeInstallSnapshotRequest(encodeInstallSnapshotRequest(nil, &in), &out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !reflect.DeepEqual(in, out) {
			t.Fatalf("mismatch\n in=%#v\nout=%#v", in, out)
		}
	})
}
