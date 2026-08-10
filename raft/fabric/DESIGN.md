# raft/fabric — multiplexed batching Raft transport

Goal: replace the per-group `hraft.NewNetworkTransport` (one TCP socket per Raft
group per peer, msgpack RPC codec) with a shared, batching, zero-reflection
transport. Attacks the two costs the CPU profile found on the replicated-write
hot path: ~30% inter-node network syscalls (per-group sockets, tiny writes) and
~10% msgpack encode/decode.

Flag-gated (`-raft-transport=fabric`, default `mux` = current path). Old path
stays byte-identical until fabric is proven and promoted.

## Connection model

Per node, one `Fabric` owns a single TCP listener (the node's `-raft-addr`).

- **One shared multiplexed link per peer, per direction.** When this node first
  needs to SEND to a peer, it dials one long-lived TCP conn and runs a frame
  loop over it: this node's requests to that peer flow out, that peer's
  responses flow back. The peer independently dials THIS node for its own
  requests. So a pair has up to 2 conns (one per initiator); each carries
  requests one way + their responses the other way. Symmetric, like hashicorp
  but shared across ALL groups.
- **Dedicated one-shot conn per InstallSnapshot.** Snapshots stream an arbitrary
  `io.Reader` and can be large; multiplexing them over the shared link would
  head-of-line-block every other group. So a snapshot dials its own conn, does a
  single request→stream→response, and closes. Only AppendEntries / RequestVote /
  RequestPreVote / TimeoutNow / heartbeat ride the shared link.

Handshake: after dial, initiator writes 1 conn-type byte — `connMux` (0x01) or
`connSnapshot` (0x02) — then the group-agnostic frame loop (mux) or the one-shot
snapshot exchange runs.

## Wire frame (shared mux link)

Little-endian, length-delimited:

```
[magic u8=0xR][ver u8=1][kind u8][groupID u32][reqID u64][payloadLen u32][payload…]
```

- `kind`: high bit = isResponse. Low 7 bits = rpcType (matches hraft iota:
  0 AppendEntries, 1 RequestVote, 2 InstallSnapshot(only over dedicated conn),
  3 TimeoutNow, 4 RequestPreVote). A response echoes the request's rpcType with
  the isResponse bit set, plus a 1-byte app-error flag in the payload prefix.
- `groupID`: routes a REQUEST to that group's Consumer (or heartbeat fast-path).
  On a response it is informational; correlation is by `reqID`.
- `reqID`: link-global monotonic (atomic uint64) chosen by the requester;
  responses echo it. Because groups interleave on the shared conn, responses
  can arrive out of group-order, so correlation is by reqID, NOT conn order
  (this is the key difference from hashicorp's per-conn in-order pipeline).

## Payload codecs (zero-reflection, big-endian)

Helpers: `putU64/getU64`, `putBytes` = `[u32 len][bytes]`, `getBytes`.
`RPCHeader` = `[u8 ProtocolVersion][bytes ID][bytes Addr]`.
`Log` = `[u64 Index][u64 Term][u8 Type][bytes Data][bytes Extensions][i64 AppendedAtUnixNano]`
(AppendedAt zero → 0). Reuse the field layout already proven in
`raft/logstore/codec.go`.

- **AppendEntriesRequest**: header, `[u64 Term]`, `putBytes Leader`,
  `[u64 PrevLogEntry][u64 PrevLogTerm]`, `[u32 nEntries]` then each Log,
  `[u64 LeaderCommitIndex]`.
- **AppendEntriesResponse**: header, `[u64 Term][u64 LastLog][u8 Success][u8 NoRetryBackoff]`.
- **RequestVoteRequest**: header, `[u64 Term]`, `putBytes Candidate`,
  `[u64 LastLogIndex][u64 LastLogTerm][u8 LeadershipTransfer]`.
- **RequestVoteResponse**: header, `[u64 Term]`, `putBytes Peers`, `[u8 Granted]`.
- **RequestPreVoteRequest**: header, `[u64 Term][u64 LastLogIndex][u64 LastLogTerm]`.
- **RequestPreVoteResponse**: header, `[u64 Term][u8 Granted]`.
- **TimeoutNowRequest/Response**: header only.
- **InstallSnapshotRequest** (dedicated conn): header, `[u8 SnapshotVersion][u64 Term]`,
  `putBytes Leader`, `[u64 LastLogIndex][u64 LastLogTerm]`, `putBytes Peers`,
  `[u64 Configuration? via putBytes][u64 ConfigurationIndex][i64 Size]`, then the
  raw snapshot bytes streamed (Size bytes). Response: header, `[u64 Term][u8 Success]`.

Each response payload is prefixed with `[u8 appErr][u32 errLen][errbytes]`; if
appErr=1 the waiter gets that error instead of a decoded struct.

## Outbound path (send)

- **Sync RPC** (`AppendEntries`/`RequestVote`/`RequestPreVote`/`TimeoutNow`):
  reqID = atomic add; register `pending[reqID] = chan result`; encode frame,
  hand to the peer link's write channel; block on the chan with the transport
  timeout; on timeout/err delete pending + return error.
- **Pipeline** (`AppendEntriesPipeline`): returns a `*pipeline`. Its
  `AppendEntries(args,resp)` allocates reqID, registers an `AppendFuture`,
  enqueues the frame, and returns the future (no block — back-pressure via the
  write channel capacity). A `Consumer()` chan yields futures as their responses
  land. `maxInFlight` = DefaultMaxRPCsInFlight semantics.
- **Batched writer** (the syscall win): each peer link has a `sendCh chan
  frame` and one writer goroutine. Loop: block for one frame, then greedily
  drain `sendCh` non-blocking into a `bufio.Writer`; when the channel is
  momentarily empty, `Flush()` once. So a burst of frames from many groups
  becomes ONE `write()`. Optional tiny (~50µs) linger timer to widen batches
  under steady load. No per-frame syscall.

## Inbound path (receive)

Per accepted mux conn, one reader goroutine:
- Read frame. If **request**: decode by rpcType into the arg struct; build
  `hraft.RPC{Command, RespChan: make(chan RPCResponse,1)}`; heartbeat fast-path
  (AppendEntries with Term>0, PrevLogEntry==0, PrevLogTerm==0, no entries,
  LeaderCommit==0 — mirror hraft `isHeartbeat`) → call the group's registered
  heartbeat handler if set, else push to the group's Consumer. Spawn/att a
  small responder that waits on RespChan, encodes the response frame (same
  reqID, isResponse), enqueues to this conn's writer.
- If **response**: look up `pending[reqID]`, deliver decoded resp/err, delete.

Consumer channels are per-group, buffered (like hashicorp's rpcChan). Fabric
routes by groupID.

## Per-group Transport facade

`fabric.For(groupID) hraft.Transport` returns a `groupTransport` implementing:
`Consumer`, `LocalAddr`, `AppendEntries`, `AppendEntriesPipeline`, `RequestVote`,
`RequestPreVote` (WithPreVote), `InstallSnapshot`, `TimeoutNow`, `EncodePeer`,
`DecodePeer`, `SetHeartbeatHandler`, `Close` (WithClose). All share the Fabric's
link manager, listener, reqID counter, and pending map, tagging outbound frames
with `groupID` and receiving inbound via the demux.

`EncodePeer/DecodePeer`: address as raw bytes (`[]byte(addr)` / `ServerAddress(b)`),
same as hashicorp.

## Errors / reconnect

On any conn read/write error: close the conn, fail every `pending` waiting on it
with the error, and drop the link entry. The next send re-dials lazily. Raft
tolerates transport errors (it retries AppendEntries and re-elects). No data
durability impact — the transport is best-effort; safety lives in the log +
majority commit.

## Wiring

- `raft.Config`: add `Transport hraft.Transport` (highest precedence in
  `buildTransport`; if set, return it directly).
- `cluster.Config`: add `RaftTransport string` ("mux"|"fabric").
- `cluster/node.go`: when fabric, build ONE `fabric.New(raftAddr, groupIDs)`
  per node instead of `mux.New`; pass `fabric.For(shardID)` as
  `subCfg.RaftTransport`-equivalent (`raft.Config.Transport`). Meta group uses
  `fabric.For(metaGroupID)`.
- `cmd/rostam-server`: `-raft-transport` flag → `cluster.Config.RaftTransport`.
- Fold in the proven `BatchApplyCh=true` (already staged in raft/node.go).

## Test plan

- Codec: table + fuzz round-trip for every RPC struct (encode→decode→equal),
  including empty/large Data, multi-entry AppendEntries, unicode addrs.
- Frame: partial-read reassembly, bad magic/version rejection, oversized len bound.
- Integration: run the EXISTING `cluster` multinode / write-from-any-node /
  leader-kill / partition tests with `RaftTransport=fabric`.
- A/B bench: shards=8, nosync, 256 conns — fabric vs 134k baseline.

Promote to default only if fabric wins AND raft+shard+cluster suites stay green.
