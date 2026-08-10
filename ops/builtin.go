// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// ErrShortArgs indicates the args byte slice is shorter than expected.
var ErrShortArgs = errors.New("ops: args too short")

// stdKeyExtractor reads [keyLen u16][key] from the start of args.
// It is the canonical extractor for all five built-in routable ops
// (get/put/del/expire/incr), whose args always start with this layout.
func stdKeyExtractor(args []byte) ([]byte, bool) {
	if len(args) < 2 {
		return nil, false
	}
	n := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+n {
		return nil, false
	}
	return args[2 : 2+n], true
}

// RegisterBuiltins adds the standard set of ops to the registry:
//   - "get"    (read-only)   args: [keyLen u16][key]                    → value bytes (or ErrNotFound)
//   - "put"    (read-write)  args: [keyLen u16][key][valLen u32][val][ttlMs u64]
//   - "del"    (read-write)  args: [keyLen u16][key]                    → 1-byte 0/1
//   - "expire" (read-write)  args: [keyLen u16][key][ttlMs u64]
//   - "incr"   (read-write)  args: [keyLen u16][key][delta i64]         → new value as i64 BE
//   - "__ping__" (read-only) args: (ignored)                             → empty
func RegisterBuiltins(r *Registry) error {
	routables := []struct {
		name string
		kind OpKind
		fn   Handler
	}{
		{"get", OpReadOnly, handleGet},
		{"put", OpReadWrite, handlePut},
		{"del", OpReadWrite, handleDel},
		{"expire", OpReadWrite, handleExpire},
		{"incr", OpReadWrite, handleIncr},
	}
	for _, x := range routables {
		if err := r.RegisterRoutable(x.name, x.kind, x.fn, stdKeyExtractor); err != nil {
			return err
		}
	}
	// put_batch packs N puts into one Raft log entry (one fsync / round-trip /
	// apply for the whole batch). It routes by its FIRST key, so every key in a
	// batch must hash to the same shard — the cluster fan-out (Node.PutBatch)
	// guarantees that by grouping before it calls.
	if err := r.RegisterRoutable("put_batch", OpReadWrite, handlePutBatch, putBatchKeyExtractor); err != nil {
		return err
	}
	// __ping__ is shardless — registered via Register.
	if err := r.Register("__ping__", OpReadOnly, handlePing); err != nil {
		return err
	}
	// __ready__ is a shardless READINESS probe. The default handler here always
	// reports ready (correct for single-node / Direct, which is always its own
	// leader). In cluster mode cluster.Node intercepts __ready__ in its adminOps
	// BEFORE this registry lookup and runs a real per-hosted-shard leader check
	// (see cluster/admin_ops.go handleReady). Distinct from __ping__ (liveness):
	// readiness reflects whether this node can actually serve its shards.
	if err := r.Register(ReadyOp, OpReadOnly, handleReady); err != nil {
		return err
	}
	// __metrics__ is shardless: it renders the local node's per-collection
	// Prometheus stats. follow-up: a clustered scrape would gather + concatenate
	// each shard's exposition; today it serves the node it is dispatched to.
	if err := r.Register(MetricsOp, OpReadOnly, handleMetrics); err != nil {
		return err
	}
	// __repl_metrics__ is a shardless REPLICATION-observability op. The default
	// handler here reports no replicated shards (correct for single-node / Direct,
	// which replicates nothing). In cluster mode cluster.Node intercepts it in its
	// adminOps BEFORE this registry lookup and renders the real per-hosted-shard
	// ISR / lag view (see cluster/repl_metrics.go handleReplMetrics), mirroring how
	// __ready__ is overridden. Read-only, no args; result is a JSON body served
	// as-is by the HTTP /v1/replication handler.
	if err := r.Register(ReplMetricsOp, OpReadOnly, handleReplMetrics); err != nil {
		return err
	}
	// Vector ops are routed by collection name (vectorKeyColAt1/At2) so each
	// collection lives on one shard's Raft group and collections distribute
	// across shards. Args that lead with [flags][colLen] use At2; those that lead
	// with [nameLen] use At1.
	type vop struct {
		name string
		kind OpKind
		fn   Handler
		ke   routeExtractor
	}
	for _, o := range []vop{
		{"vector_create_collection", OpReadWrite, handleVectorCreateCollection, routeAt1},
		{"vector_drop_collection", OpReadWrite, handleVectorDropCollection, routeAt1},
		{"vector_insert", OpReadWrite, handleVectorInsert, routeAt2},
		{"vector_insert_if_absent", OpReadWrite, handleVectorInsertIfAbsent, routeAt2},
		{"vector_exists", OpReadOnly, handleVectorExists, routeAt1},
		{"vector_delete", OpReadWrite, handleVectorDelete, routeAt1},
		{"vector_get", OpReadOnly, handleVectorGet, routeAt1},
		{"vector_get_batch", OpReadOnly, handleVectorGetBatch, routeAt1},
		{"vector_set_payload", OpReadWrite, handleVectorSetPayload, routeAt1},
		{"vector_overwrite_payload", OpReadWrite, handleVectorOverwritePayload, routeAt1},
		{"vector_delete_payload_keys", OpReadWrite, handleVectorDeletePayloadKeys, routeAt1},
		{"vector_clear_payload", OpReadWrite, handleVectorClearPayload, routeAt1},
		{"vector_search", OpReadOnly, handleVectorSearch, routeAt2},
		{"vector_hybrid_search", OpReadOnly, handleVectorHybridSearch, routeAt2},
		{"vector_hybrid_lanes", OpReadOnly, handleVectorHybridLanes, routeAt2},
		// Full-text (BM25) ops. Both lead with [flags:u8][colLen:u8][col]... (the
		// At2 layout, IDENTICAL to vector_hybrid_search), so they route via
		// vectorKeyColAt2 — name at offset 1, behind the flags byte. vector_hybrid_text_lanes
		// is the per-partition fan-out leaf (shares the vector_hybrid_text wire).
		{"vector_search_text", OpReadOnly, handleVectorSearchText, routeAt2},
		{"vector_hybrid_text", OpReadOnly, handleVectorHybridText, routeAt2},
		{"vector_hybrid_text_lanes", OpReadOnly, handleVectorHybridTextLanes, routeAt2},
		// vector_bm25_stats is phase 0 of the global-DF (dfs) text fan-out. Its args
		// lead with [colLen:u8][col]... (NO flags byte), so it routes via
		// vectorKeyColAt1 — name at offset 0, NOT At2 like the scoring ops.
		{"vector_bm25_stats", OpReadOnly, handleVectorBM25Stats, routeAt1},
		// vector_query is the unified Query API op. Its args lead with [colLen:u8]
		// [col] (the QuerySpec blob is opaque to routing), so it routes via
		// vectorKeyColAt1 like the rest of the At1 family.
		{"vector_query", OpReadOnly, handleVectorQuery, routeAt1},
		{"vector_upsert", OpReadWrite, handleVectorUpsert, routeAt2},
		{"vector_bulk_stage", OpReadWrite, handleVectorBulkStage, routeAt1},
		{"vector_bulk_stage_payload", OpReadWrite, handleVectorBulkStagePayload, routeAt1},
		{"vector_bulk_build", OpReadWrite, handleVectorBulkBuild, routeAt1},
		{"vector_search_docs", OpReadOnly, handleVectorSearchDocs, routeAt2},
		{"vector_delete_by_filter", OpReadWrite, handleVectorDeleteByFilter, routeAt1},
		{"vector_search_groups", OpReadOnly, handleVectorSearchGroups, routeAt1},
		{"vector_group_candidates", OpReadOnly, handleVectorGroupCandidates, routeAt1},
		{"vector_scroll", OpReadOnly, handleVectorScroll, routeAt1},
		{"vector_scan_vectors", OpReadOnly, handleVectorScanVectors, routeAt1},
		{"vector_get_config", OpReadOnly, handleVectorGetConfig, routeAt1},
		{"vector_mv_create_collection", OpReadWrite, handleMVCreate, routeAt1},
		{"vector_mv_drop_collection", OpReadWrite, handleMVDrop, routeAt1},
		{"vector_mv_add", OpReadWrite, handleMVAdd, routeAt1},
		{"vector_mv_add_if_absent", OpReadWrite, handleMVAddIfAbsent, routeAt1},
		{"vector_mv_add_versioned", OpReadWrite, handleMVAddVersioned, routeAt1},
		{"vector_mv_add_batch", OpReadWrite, handleMVAddBatch, routeAt1},
		{"vector_mv_exists", OpReadOnly, handleMVExists, routeAt1},
		{"vector_mv_search", OpReadOnly, handleMVSearch, routeAt1},
		// MV-hybrid ops use the At2 (flags-first) layout — [flags:u8][colLen:u8][col]...
		// (EncodeMVHybridArgs), IDENTICAL to the named/dense hybrid wire, so they route
		// via vectorKeyColAt2 (NOT At1 like the rest of the mv_* family). The lanes op
		// is the partition fan-out leaf.
		{"vector_mv_hybrid_search", OpReadOnly, handleMVHybridSearch, routeAt2},
		{"vector_mv_hybrid_lanes", OpReadOnly, handleMVHybridLanes, routeAt2},
		{"vector_mv_delete", OpReadWrite, handleMVDelete, routeAt1},
		{"vector_mv_get", OpReadOnly, handleMVGet, routeAt1},
		{"vector_mv_get_batch", OpReadOnly, handleMVGetBatch, routeAt1},
		{"vector_mv_set_payload", OpReadWrite, handleMVSetPayload, routeAt1},
		{"vector_mv_overwrite_payload", OpReadWrite, handleMVOverwritePayload, routeAt1},
		{"vector_mv_delete_payload_keys", OpReadWrite, handleMVDeletePayloadKeys, routeAt1},
		{"vector_mv_clear_payload", OpReadWrite, handleMVClearPayload, routeAt1},
		{"vector_mv_get_config", OpReadOnly, handleMVGetConfig, routeAt1},
		{"vector_mv_scan_vectors", OpReadOnly, handleMVScanVectors, routeAt1},
		{"vector_mv_scroll", OpReadOnly, handleMVScroll, routeAt1},
		// vector_mv_query is the multi-vector Query API op (MaxSim + doc-sparse
		// FUSION / RERANK). It shares the vector_query arg wire ([colLen:u8][col]
		// [specLen:u32][spec][optsTrailer]) and result codec, so it routes by
		// collection name at offset 0 (vectorKeyColAt1) like the rest of the At1 family.
		{"vector_mv_query", OpReadOnly, handleMVQuery, routeAt1},
		{"vector_named_create_collection", OpReadWrite, handleNamedCreate, routeAt1},
		{"vector_named_drop_collection", OpReadWrite, handleNamedDrop, routeAt1},
		{"vector_named_insert", OpReadWrite, handleNamedInsert, routeAt1},
		{"vector_named_delete", OpReadWrite, handleNamedDelete, routeAt1},
		{"vector_named_get", OpReadOnly, handleNamedGet, routeAt1},
		{"vector_named_get_batch", OpReadOnly, handleNamedGetBatch, routeAt1},
		{"vector_named_set_payload", OpReadWrite, handleNamedSetPayload, routeAt1},
		{"vector_named_overwrite_payload", OpReadWrite, handleNamedOverwritePayload, routeAt1},
		{"vector_named_delete_payload_keys", OpReadWrite, handleNamedDeletePayloadKeys, routeAt1},
		{"vector_named_clear_payload", OpReadWrite, handleNamedClearPayload, routeAt1},
		{"vector_named_search", OpReadOnly, handleNamedSearch, routeAt1},
		{"vector_named_sparse_search", OpReadOnly, handleNamedSparseSearch, routeAt1},
		{"vector_named_hybrid_search", OpReadOnly, handleNamedHybridSearch, routeAt2},
		{"vector_named_hybrid_lanes", OpReadOnly, handleNamedHybridLanes, routeAt2},
		{"vector_named_search_docs", OpReadOnly, handleNamedSearchDocs, routeAt1},
		{"vector_named_scroll", OpReadOnly, handleNamedScroll, routeAt1},
		{"vector_named_get_config", OpReadOnly, handleNamedGetConfig, routeAt1},
		// vector_named_query is the named-collection Query API op (multi-space N-lane
		// FUSION / RERANK). It shares the vector_query arg wire ([colLen:u8][col]
		// [specLen:u32][spec][optsTrailer]) and result codec, so it routes by
		// collection name at offset 0 (vectorKeyColAt1) like the rest of the At1 family.
		{"vector_named_query", OpReadOnly, handleNamedQuery, routeAt1},
	} {
		if err := r.registerRoutableInto(o.name, o.kind, o.fn, o.ke.ke, o.ke.layout); err != nil {
			return err
		}
	}
	return nil
}

// --- handler implementations ---

func handleGet(tx *TxContext, args []byte) ([]byte, error) {
	key, err := decodeKeyArgs(args)
	if err != nil {
		return nil, err
	}
	return tx.Get(key)
}

func handlePut(tx *TxContext, args []byte) ([]byte, error) {
	key, val, ttl, err := decodePutArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.Put(key, val, ttl)
}

func handleDel(tx *TxContext, args []byte) ([]byte, error) {
	key, err := decodeKeyArgs(args)
	if err != nil {
		return nil, err
	}
	existed, err := tx.Del(key)
	if err != nil {
		return nil, err
	}
	if existed {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

func handleExpire(tx *TxContext, args []byte) ([]byte, error) {
	key, ttl, err := decodeExpireArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.Expire(key, ttl)
}

func handleIncr(tx *TxContext, args []byte) ([]byte, error) {
	key, delta, err := decodeIncrArgs(args)
	if err != nil {
		return nil, err
	}
	var current int64
	v, err := tx.Get(key)
	switch {
	case err == cache.ErrNotFound:
		current = 0
	case err != nil:
		return nil, err
	case len(v) != 8:
		return nil, errors.New("ops: incr value is not 8 bytes")
	default:
		current = int64(binary.BigEndian.Uint64(v)) //nolint:gosec // safe: reinterpret stored i64 as u64 for binary read
	}
	next := current + delta
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(next)) //nolint:gosec // safe: store i64 as u64 for binary write
	if err := tx.Put(key, buf, 0); err != nil {
		return nil, err
	}
	return EncodeIncrResult(next), nil
}

// handlePing is a no-op heartbeat used by the client pool's stale-conn check.
// It does not touch the cache, must be cheap, and tolerates non-empty args.
func handlePing(_ *TxContext, _ []byte) ([]byte, error) {
	return nil, nil
}

// ReadyOp is the shardless readiness-probe op name (see the __ready__
// registration). A nil error means ready; a non-nil error means not ready.
const ReadyOp = "__ready__"

// handleReady is the DEFAULT (single-node / Direct) readiness handler: always
// ready. Cluster mode overrides it with a real hosted-shard leader check.
func handleReady(_ *TxContext, _ []byte) ([]byte, error) {
	return nil, nil
}

// MetricsOp renders the Prometheus text exposition for all dense collections on
// this node. The result bytes are the exposition body (text/plain), served as-is
// by the HTTP /metrics handler.
const MetricsOp = "__metrics__"

// handleMetrics renders every dense collection's stats into the Prometheus text
// exposition format and returns it as the op result. It is read-only and takes
// no args. A node with no vector store (KV-only) returns ErrVectorsNotAvailable.
func handleMetrics(tx *TxContext, _ []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	var buf bytes.Buffer
	if err := tx.vectors.WritePrometheusAll(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ReplMetricsOp is the shardless replication-observability op name (see the
// __repl_metrics__ registration). Its result is a JSON body describing the
// per-hosted-shard replication state (mode / primary / ISR / min-ISR / lag).
const ReplMetricsOp = "__repl_metrics__"

// handleReplMetrics is the DEFAULT (single-node / Direct) replication-metrics
// handler: no shards are replicated, so it reports an empty shard list. Cluster
// mode overrides it with a real per-hosted-shard ISR/lag view. The empty JSON is
// a valid body the /v1/replication handler serves verbatim.
func handleReplMetrics(_ *TxContext, _ []byte) ([]byte, error) {
	return []byte(`{"shards":[]}`), nil
}

// --- args codecs (Encode public, decode private) ---

// EncodeKeyArgs encodes "{keyLen u16}{key}" used by get and del.
func EncodeKeyArgs(key []byte) []byte {
	buf := make([]byte, 2+len(key))
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	return buf
}

func decodeKeyArgs(args []byte) ([]byte, error) {
	if len(args) < 2 {
		return nil, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen {
		return nil, ErrShortArgs
	}
	return args[2 : 2+klen], nil
}

// EncodePutArgs encodes "{keyLen u16}{key}{valLen u32}{val}{ttlMs u64}".
func EncodePutArgs(key, val []byte, ttl time.Duration) []byte {
	buf := make([]byte, 2+len(key)+4+len(val)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(val))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[off+4:], val)
	off += 4 + len(val)
	binary.BigEndian.PutUint64(buf[off:off+8], uint64(ttl/time.Millisecond)) //nolint:gosec // safe: duration to milliseconds always positive
	return buf
}

func decodePutArgs(args []byte) (key, val []byte, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen+4 {
		return nil, nil, 0, ErrShortArgs
	}
	key = args[2 : 2+klen]
	off := 2 + klen
	vlen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	if len(args) < off+vlen+8 {
		return nil, nil, 0, ErrShortArgs
	}
	val = args[off : off+vlen]
	off += vlen
	ttl = time.Duration(binary.BigEndian.Uint64(args[off:off+8])) * time.Millisecond //nolint:gosec // safe: u64 from wire is milliseconds, always positive
	return key, val, ttl, nil
}

// EncodeExpireArgs encodes "{keyLen u16}{key}{ttlMs u64}".
func EncodeExpireArgs(key []byte, ttl time.Duration) []byte {
	buf := make([]byte, 2+len(key)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	binary.BigEndian.PutUint64(buf[2+len(key):], uint64(ttl/time.Millisecond)) //nolint:gosec // safe: duration to milliseconds always positive
	return buf
}

func decodeExpireArgs(args []byte) (key []byte, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen+8 {
		return nil, 0, ErrShortArgs
	}
	key = args[2 : 2+klen]
	ttl = time.Duration(binary.BigEndian.Uint64(args[2+klen:2+klen+8])) * time.Millisecond //nolint:gosec // safe: u64 from wire is milliseconds, always positive
	return key, ttl, nil
}

// EncodeIncrArgs encodes "{keyLen u16}{key}{delta i64}".
func EncodeIncrArgs(key []byte, delta int64) []byte {
	buf := make([]byte, 2+len(key)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	binary.BigEndian.PutUint64(buf[2+len(key):], uint64(delta)) //nolint:gosec // safe: reinterpret i64 as u64 for binary write
	return buf
}

func decodeIncrArgs(args []byte) (key []byte, delta int64, err error) {
	if len(args) < 2 {
		return nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen+8 {
		return nil, 0, ErrShortArgs
	}
	key = args[2 : 2+klen]
	delta = int64(binary.BigEndian.Uint64(args[2+klen : 2+klen+8])) //nolint:gosec // safe: reinterpret stored u64 as i64 for binary read
	return key, delta, nil
}

// EncodeIncrResult encodes the int64 result of an incr op.
func EncodeIncrResult(v int64) []byte {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(v)) //nolint:gosec // safe: reinterpret i64 as u64 for binary write
	return buf
}

// DecodeIncrResult parses an incr result back into int64.
func DecodeIncrResult(b []byte) (int64, error) {
	if len(b) != 8 {
		return 0, ErrShortArgs
	}
	return int64(binary.BigEndian.Uint64(b)), nil //nolint:gosec // safe: reinterpret stored u64 as i64 for binary read
}

// ErrVectorsNotAvailable is returned by vector op handlers when the
// dispatcher was constructed without a CollectionStore.
var ErrVectorsNotAvailable = errors.New("ops: vector store not available")

func handleVectorCreateCollection(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, cfg, err := DecodeCreateCollectionArgs(args)
	if err != nil {
		return nil, err
	}
	if err := tx.vectors.CreateCollection(name, cfg); err != nil {
		return nil, err
	}
	return nil, nil
}

func handleVectorDropCollection(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := DecodeDropCollectionArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.DropCollection(name)
}

func handleVectorInsert(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// Decode the dense vector into a pooled scratch backing instead of allocating a
	// fresh []float32 per insert. Both downstream paths (RestoreInsert and
	// InsertCASKeyTTL → hnsw.Insert → arena.Insert) COPY the vector into the arena
	// before this handler returns, so the scratch is never retained past the op and
	// is safe to recycle via the defer. (The BULK staging path retains its decoded
	// vecs and is deliberately left on the allocating path.)
	bufp := vectorDenseBufPool.Get().(*[]float32)
	defer vectorDenseBufPool.Put(bufp)
	name, id, vec, ttl, meta, sparse, version, expected, hasExpected, keyTTLMs, keyExpiresAbs, err := DecodeVectorInsertArgsKeyExpiresInto((*bufp)[:0], args)
	if err != nil {
		return nil, err
	}
	*bufp = vec // retain the (possibly grown/reallocated) backing for the next op
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	if version != 0 {
		// Version-preserving reinsert (reshard/resplit backfill): restore the exact
		// per-point CAS version verbatim instead of bumping to 1. keyExpiresAbs is the
		// copied point's ABSOLUTE per-key payload deadline map (from the scan trailer),
		// applied VERBATIM by RestoreInsert (NOT recomputed now+ttl) so resharded
		// per-key TTLs survive time-stable; nil when the point has no per-key TTL.
		//
		// Under a replicated apply stamp the POINT ttl deadline is stamped against the
		// leader clock (RestoreInsertAt) so every replica records the identical absolute
		// point expiry; unstamped (single-node/Direct) keeps the wall-clock path
		// byte-identical (#4 vector TTL determinism).
		if tx.applyStamped {
			return nil, c.RestoreInsertAt(id, vec, ttl, meta, sparse, keyExpiresAbs, version, int64(tx.applyNowMs)) //nolint:gosec // stamped unix-millis fits int64
		}
		return nil, c.RestoreInsert(id, vec, ttl, meta, sparse, keyExpiresAbs, version)
	}
	// keyTTLMs (relative ms) → the engine computes the absolute deadline now+ttl at
	// insert and the WAL logs it (replay restores verbatim). Empty/nil = no per-key
	// TTL (zero-overhead).
	//
	// Under a replicated apply stamp EVERY deadline computation and liveness check
	// (point ttl, per-key ttl, CAS/reclaim) is judged against the leader-stamped
	// clock via InsertCASKeyTTLAt, so replicas at skewed wall clocks store
	// byte-identical committed state; unstamped keeps the wall-clock path
	// byte-identical (branch on applyStamped, NOT applyNowMs != 0 — see TxContext).
	if tx.applyStamped {
		_, err = c.InsertCASKeyTTLAt(id, vec, ttl, meta, sparse, keyTTLMs, vector.CASCond{Expected: expected, Has: hasExpected}, int64(tx.applyNowMs)) //nolint:gosec // stamped unix-millis fits int64
	} else {
		_, err = c.InsertCASKeyTTL(id, vec, ttl, meta, sparse, keyTTLMs, vector.CASCond{Expected: expected, Has: hasExpected})
	}
	return nil, err
}

// handleVectorInsertIfAbsent runs the atomic insert-if-absent engine op (reuses
// the insert-args wire shape). It is registered OpReadWrite: Raft serialization
// is the cross-op atomicity guarantee that closes Race A. Result: [inserted:u8].
func handleVectorInsertIfAbsent(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, vec, ttl, meta, sparse, version, _, _, _, keyExpiresAbs, err := DecodeVectorInsertArgsKeyExpires(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// version!=0 → version-PRESERVING if-absent (the online reshard copy pass,
	// EncodeVectorInsertArgsVersionedKeyExpires): carry the copied point's exact
	// per-point CAS version instead of resetting it to 1, while still never
	// clobbering a concurrent live dual-write (Race A). version==0 is the plain
	// if-absent. keyExpiresAbs is the copied point's ABSOLUTE per-key payload
	// deadline map (from the scan trailer), set VERBATIM on a real insert (NOT
	// recomputed) so resharded per-key TTLs survive time-stable; nil otherwise.
	// Under a replicated apply stamp the liveness OUTCOME (resurrect an expired id
	// vs no-op) and the point-TTL deadline are judged against the leader-stamped
	// clock, so skewed replicas agree on insert-vs-noop and stamp identical
	// deadlines; unstamped keeps the wall-clock path byte-identical (#4 vector TTL
	// determinism).
	var inserted bool
	if tx.applyStamped {
		inserted, err = c.InsertIfAbsentVersionAt(id, vec, ttl, meta, sparse, keyExpiresAbs, version, int64(tx.applyNowMs)) //nolint:gosec // stamped unix-millis fits int64
	} else {
		inserted, err = c.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, keyExpiresAbs, version)
	}
	if err != nil {
		return nil, err
	}
	return EncodeIfAbsentResult(inserted), nil
}

// handleVectorExists is the cheap dense liveness probe (OpReadOnly) the copy's
// resurrection guard uses (Race B). Result: [exists:u8].
func handleVectorExists(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, err := DecodeExistsArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	return EncodeExistsResult(c.Exists(id)), nil
}

// handleVectorUpsert reuses the insert-args wire shape; the caller embeds
// document content in the metadata ($content field), so Upsert is called with an
// empty content string and the content rides in meta.
func handleVectorUpsert(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, vec, ttl, meta, sparse, _, expected, hasExpected, keyTTLMs, err := DecodeVectorInsertArgsKeyTTL(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	if ms, stamped := tx.applyStamp(); stamped {
		_, err = c.UpsertCASKeyTTLAt(id, vec, "", ttl, meta, sparse, keyTTLMs, cas, ms)
	} else {
		_, err = c.UpsertCASKeyTTL(id, vec, "", ttl, meta, sparse, keyTTLMs, cas)
	}
	return nil, err
}

// handleVectorBulkStage appends a batch of (id, vector) pairs to a collection's
// bulk-load staging buffer (nothing is indexed until vector_bulk_build).
func handleVectorBulkStage(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// NOTE: the decoded vecs are RETAINED in the collection's stage buffer
	// (StageBulk → c.stageVecs) until vector_bulk_build, so they are NOT poolable
	// the way the single-insert decode buffer is — the per-vector []float32 here is
	// kept live past this op and must stay independently owned.
	name, ids, vecs, err := DecodeBulkStageArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.StageBulk(name, ids, vecs)
}

// handleVectorBulkStagePayload is handleVectorBulkStage for a batch that also
// carries a per-point payload — the shape a filtered workload needs, and the one
// the vectors-only staging op has no room for. The payloads are applied by the
// build's placement pass (see hnsw.BuildConcurrentMeta), so a filter case gets
// the multi-core build instead of one indexed insert per point.
func handleVectorBulkStagePayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// Like handleVectorBulkStage, the decoded vecs (and now the decoded metadata
	// maps) are RETAINED in the collection's stage buffer until vector_bulk_build,
	// so neither is poolable.
	name, ids, vecs, metas, err := DecodeBulkStagePayloadArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.StageBulkPayloads(name, ids, vecs, metas)
}

// handleVectorBulkBuild builds a collection's staged vectors into the (empty)
// index in one concurrent pass — the multi-core initial-load path.
func handleVectorBulkBuild(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, workers, err := DecodeBulkBuildArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.BuildStaged(name, workers)
}

func handleVectorSearchDocs(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, k, query, filter, err := DecodeVectorSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	docs, err := c.SearchDocs(query, k, filter)
	if err != nil {
		return nil, err
	}
	return EncodeVectorDocs(docs), nil
}

func handleVectorSearchGroups(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, k, query, opts, err := DecodeGroupSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	groups, err := c.SearchGroups(query, k, opts)
	if err != nil {
		return nil, err
	}
	return EncodeGroups(groups), nil
}

func handleVectorGroupCandidates(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, _, query, opts, err := DecodeGroupSearchArgs(args) // k unused; coordinator groups
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	cands, err := c.GroupCandidates(query, opts)
	if err != nil {
		return nil, err
	}
	return EncodeVectorDocs(cands), nil
}

func handleVectorScroll(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, filter, limit, _, _, afterID, hasAfter, order, err := DecodeScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// Cursor-aware page (the partition fan-out passes the SAME global cursor to
	// every partition; the coordinator derives next_cursor from the merged docs, so
	// this handler returns only the per-partition docs). With no cursor
	// (hasAfter=false) and no order_by this is the deterministic id-ascending
	// smallest-id `limit` scroll.
	if order != nil {
		// order_by page: sort the live admitted set by the (value, id) order, EXCLUDE
		// missing-field points, seek past the (resumeKey, afterID) cursor / start_from.
		ob := scrollOrderToVector(order)
		var afterKey float64
		if order.HasResume {
			afterKey = order.ResumeKey
		}
		docs, _, _, derr := c.ScrollDocsPageOrder(filter, ob, afterID, afterKey, hasAfter, limit)
		if derr != nil {
			return nil, derr
		}
		return EncodeVectorDocs(docs), nil
	}
	docs, _, _, err := c.ScrollDocsPage(filter, afterID, hasAfter, limit)
	if err != nil {
		return nil, err
	}
	return EncodeVectorDocs(docs), nil
}

// scrollOrderToVector maps the ops args ScrollOrder onto the engine's vector.OrderBy
// (the resume value/id are passed to ScrollDocsPageOrder separately, not in OrderBy).
// It delegates to the exported ScrollOrderToOrderBy so the leaf engine and the
// coordinator fan-out (rostam.scrollOrderByFromOps) share ONE mapping — including the
// multi-key Tail + v4 resume tuple.
func scrollOrderToVector(o *ScrollOrder) *vector.OrderBy {
	return ScrollOrderToOrderBy(o)
}

// ScrollOrderToOrderBy maps the ops args ScrollOrder onto vector.OrderBy, including the
// MULTI-KEY Tail (the secondary key specs) and the v4 resume TUPLE (ResumeKeys). A nil
// or single-key ScrollOrder maps to the byte/behaviour-identical single-key vector.OrderBy
// (empty Tail / no ResumeKeys); a multi-key ScrollOrder fills OrderBy.Tail + ResumeKeys so
// the engine's tuple comparator + v4 seek run. Shared by the leaf (scrollOrderToVector)
// and the coordinator fan-out (rostam.scrollOrderByFromOps) so they agree on the order.
func ScrollOrderToOrderBy(o *ScrollOrder) *vector.OrderBy {
	if o == nil {
		return nil
	}
	ob := &vector.OrderBy{
		Key:          o.Key,
		Desc:         o.Desc,
		IsDatetime:   o.IsDatetime,
		Kind:         o.Kind,
		StartFrom:    o.StartFrom,
		HasStart:     o.HasStart,
		ResumeStr:    o.ResumeStr,
		HasResumeStr: o.HasResumeStr,
	}
	if len(o.Tail) > 0 {
		ob.Tail = make([]vector.OrderBy, len(o.Tail))
		for i, tk := range o.Tail {
			ob.Tail[i] = vector.OrderBy{Key: tk.Key, Desc: tk.Desc, IsDatetime: tk.IsDatetime, Kind: tk.Kind}
		}
		if o.HasResumeKeys {
			ob.ResumeKeys = make([]vector.OrderVal, len(o.ResumeKeys))
			for i, rv := range o.ResumeKeys {
				ob.ResumeKeys[i] = vector.OrderVal{Num: rv.Num, Str: rv.Str, Kind: rv.Kind}
			}
			ob.HasResumeKeys = true
		}
	}
	return ob
}

// OrderByToScrollOrder maps a vector.OrderBy's MULTI-KEY Tail onto the ops args
// ScrollOrder.Tail (the per-key specs) — the inverse direction of ScrollOrderToOrderBy
// for the Tail only. The primary fields (Key/Desc/Kind/Start/Resume) are set by the
// caller (each transport builds the primary + resume per its cursor path); this fills the
// Tail so every transport's ScrollOrder construction shares ONE multi-key mapping. A
// single-key OrderBy (empty Tail) yields an empty Tail (byte-identical single-key path).
func OrderByToScrollOrderTail(ob *vector.OrderBy) []ScrollOrderKey {
	if ob == nil || len(ob.Tail) == 0 {
		return nil
	}
	tail := make([]ScrollOrderKey, len(ob.Tail))
	for i, tk := range ob.Tail {
		tail[i] = ScrollOrderKey{Key: tk.Key, Desc: tk.Desc, IsDatetime: tk.IsDatetime, Kind: tk.Kind}
	}
	return tail
}

// handleVectorScanVectors enumerates every live record of a (physical
// partition) collection — the read primitive an offline resplit uses to
// re-insert each vector into a re-hashed generation. Read-only.
func handleVectorScanVectors(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := DecodeScanVectorsArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	return EncodeScanVectorsResult(c.ScanVectors()), nil
}

// handleVectorGetConfig returns a collection's Config so resplit can create the
// new-generation partitions with the same configuration. Read-only.
func handleVectorGetConfig(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := DecodeGetConfigArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	return EncodeGetConfigResult(c.Config()), nil
}

func handleVectorDeleteByFilter(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, filter, err := DecodeDeleteByFilterArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	var n int
	if ms, stamped := tx.applyStamp(); stamped {
		n, err = c.DeleteByFilterAt(filter, ms)
	} else {
		n, err = c.DeleteByFilter(filter)
	}
	if err != nil {
		return nil, err
	}
	out := make([]byte, 4)
	binary.BigEndian.PutUint32(out, uint32(n)) //nolint:gosec // count >= 0
	return out, nil
}

func handleVectorHybridSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, k, sparse, opts, err := DecodeHybridSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	results, err := c.HybridSearch(dense, sparse, k, opts)
	if err != nil {
		return nil, err
	}
	return EncodeHybridResults(results), nil
}

func handleVectorHybridLanes(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, k, sparse, opts, err := DecodeHybridSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	denseRes, sparseRes, err := c.HybridLanes(dense, sparse, k, opts)
	if err != nil {
		return nil, err
	}
	return EncodeHybridLanesResult(denseRes, sparseRes), nil
}

// handleVectorSearchText runs a BM25 full-text search: the raw query text is
// tokenized + scored server-side (the SDK ships no tokens). Returns Documents
// (content + metadata), like vector_search_docs. The collection must have been
// created with FullText (else the engine returns ErrFullTextDisabled).
func handleVectorSearchText(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, query, k, filter, _, _, _, _, g, err := DecodeSearchTextArgsGlobal(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// Phase 1 of the global-DF (dfs) fan-out: when the coordinator supplied global
	// stats, score this shard's LOCAL postings with the injected global N/avgdl/df
	// so the returned scores are globally comparable. When absent, the EXISTING
	// per-shard-local path runs unchanged.
	if g != nil {
		docs, err := c.SearchTextGlobalDocs(query, k, filter, *g)
		if err != nil {
			return nil, err
		}
		return EncodeVectorDocs(docs), nil
	}
	docs, err := c.SearchText(query, k, filter)
	if err != nil {
		return nil, err
	}
	return EncodeVectorDocs(docs), nil
}

// handleVectorBM25Stats is phase 0 of the global-DF (dfs_query_then_fetch) text
// fan-out: it returns this partition's CORPUS-WIDE BM25 stats (n, tokenTotal, and
// per-query-term df) for the query's analyzed terms. The coordinator sums these
// across all partitions into the global N/avgdl/df injected into phase 1. A
// collection without a BM25 lane contributes zero/empty (no error), so a mixed
// fleet sums cleanly.
func handleVectorBM25Stats(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, query, _, _, _, err := DecodeBM25StatsArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	n, tokenTotal, df := c.CorpusStats(query)
	return EncodeBM25StatsResult(n, tokenTotal, df), nil
}

// handleVectorHybridText fuses a dense KNN lane with a BM25 full-text lane. The
// raw query text is analyzed server-side. Returns fused hybrid results.
func handleVectorHybridText(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, query, k, opts, err := DecodeHybridTextArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	results, err := c.HybridText(dense, query, k, opts)
	if err != nil {
		return nil, err
	}
	return EncodeHybridResults(results), nil
}

// handleVectorHybridTextLanes returns the UNFUSED dense + BM25-text candidate
// lanes for the partitioned hybrid-text fan-out (text_fanout.go), mirroring
// vector_hybrid_lanes. It shares the vector_hybrid_text wire (decoded with
// DecodeHybridTextArgs).
func handleVectorHybridTextLanes(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, query, k, opts, _, _, _, _, g, err := DecodeHybridTextArgsGlobal(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// Phase 1 of the global-DF (dfs) hybrid-text fan-out: score the text lane with
	// the injected global stats when supplied; otherwise the EXISTING per-shard-local
	// lane builder runs unchanged. The dense lane is identical either way.
	var (
		denseRes, textRes []vector.Result
	)
	if g != nil {
		denseRes, textRes, err = c.HybridTextLanesGlobal(dense, query, k, opts, *g)
	} else {
		denseRes, textRes, err = c.HybridTextLanes(dense, query, k, opts)
	}
	if err != nil {
		return nil, err
	}
	return EncodeHybridLanesResult(denseRes, textRes), nil
}

func handleVectorDelete(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, expected, hasExpected, err := DecodeVectorDeleteArgsCAS(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var removed bool
	if ms, stamped := tx.applyStamp(); stamped {
		removed, err = c.DeleteCASAt(id, cas, ms)
	} else {
		removed, err = c.DeleteCAS(id, cas)
	}
	if err != nil {
		return nil, err
	}
	if removed {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// handleVectorGet retrieves a dense point by id: vector + payload + remaining TTL
// + sparse, gated by the with_vector/with_payload flags. A missing point returns
// the found=0 FLAG (NEVER an op error), so a point-op fan-out treats "absent in
// this partition" as expected. Read-only.
func handleVectorGet(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, flags, err := DecodeVectorGetArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&getFlagWithVector != 0
	withPayload := flags&getFlagWithPayload != 0
	// Read the dense vector into a pooled scratch backing instead of allocating a
	// fresh []float32 per get. EncodeVectorGetResultV serializes the vector into the
	// response bytes (a full copy) before this handler returns, so the scratch is
	// never retained past the op and is recycled via the defer. (The BATCH get
	// handler retains each row's vec until the batch encode and stays unpooled.)
	bufp := vectorDenseBufPool.Get().(*[]float32)
	defer vectorDenseBufPool.Put(bufp)
	vec, meta, ttl, sparse, version, ok, err := tx.vectors.GetPointVersionInto((*bufp)[:0], name, id)
	if err != nil {
		return nil, err
	}
	if vec != nil {
		*bufp = vec // retain the grown backing; a miss returns nil, leave the buffer intact
	}
	return EncodeVectorGetResultV(ok, vec, meta, ttl, sparse, withVec, withPayload, version), nil
}

// handleVectorGetBatch retrieves a subset of dense points by id in one op: for each
// requested id it runs the same GetPoint lookup as handleVectorGet and emits a row.
// A missing id is a Found=false row (NEVER an op error) so the coordinator can derive
// the global missing set from absent ids. Rows preserve the input id order (this is
// the per-partition handler — it returns rows for ITS id-subset in the order given).
// The with_vector/with_payload flags gate the vec and the meta+sparse projections,
// applied here at fetch time exactly as single get. Read-only.
func handleVectorGetBatch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, ids, flags, err := DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&getFlagWithVector != 0
	withPayload := flags&getFlagWithPayload != 0
	// Acquire the collection ONCE for the whole batch and fetch each id through the
	// projection-aware getter, so a with_vector=false / with_payload=false batch pays
	// one Acquire/Release and copies NOTHING per point — versus the old per-id
	// GetPointVersion, which re-Acquired and deep-copied the dense vector + meta map +
	// sparse vector for every id regardless of the requested projection (a
	// vector_get_batch of 1000 dim-768 ids with with_vector=false copied ~3 MB of
	// float32 garbage per op). The callback owns each row's vec/meta/sparse and
	// retains them in the row slice until the single EncodeVectorGetBatchResult below.
	rows := make([]GetBatchRow, 0, len(ids))
	if err := tx.vectors.GetPointsProjected(name, ids, withVec, withPayload,
		func(id uint64, vec []float32, meta vector.Metadata, ttl time.Duration, sparse *vector.SparseVector, version uint64, ok bool) {
			row := GetBatchRow{ID: id, Found: ok}
			if ok {
				row.Vec = vec                          // nil when !withVec (getter skipped the copy)
				row.Meta = meta                        // nil when !withPayload
				row.Sparse = sparse                    // nil when !withPayload
				row.TTLMs = uint64(ttl.Milliseconds()) //nolint:gosec // TTL >= 0
				row.Version = version
			}
			rows = append(rows, row)
		}); err != nil {
		return nil, err
	}
	return EncodeVectorGetBatchResult(rows), nil
}

// handleVectorSetPayload merges the provided payload into the point's existing
// payload (reindexing + WAL-logging on a dense WAL-mode collection). A missing
// point returns applied=0 (the not-found FLAG, not an op error); a bad payload
// JSON is a hard decode error (fail-loud). Read-write.
func handleVectorSetPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, meta, keyTTLMs, expected, hasExpected, err := DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.SetPayloadCASAt(name, id, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.SetPayloadCAS(name, id, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err
	}
	return EncodePayloadResult(applied), nil
}

// handleVectorOverwritePayload replaces the point's entire payload. applied=0 for a
// missing point (not-found flag); bad JSON is a hard error. Read-write.
func handleVectorOverwritePayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, meta, keyTTLMs, expected, hasExpected, err := DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.OverwritePayloadCASAt(name, id, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.OverwritePayloadCAS(name, id, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err
	}
	return EncodePayloadResult(applied), nil
}

// handleVectorDeletePayloadKeys removes the listed keys from the point's payload.
// applied=0 for a missing point (not-found flag). Read-write.
func handleVectorDeletePayloadKeys(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, keys, expected, hasExpected, err := DecodeDeletePayloadKeysArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.DeletePayloadKeysCASAt(name, id, keys, cas, ms)
	} else {
		applied, _, err = tx.vectors.DeletePayloadKeysCAS(name, id, keys, cas)
	}
	if err != nil {
		return nil, err
	}
	return EncodePayloadResult(applied), nil
}

// handleVectorClearPayload clears the point's payload. applied=0 for a missing
// point (not-found flag). Read-write.
func handleVectorClearPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, expected, hasExpected, err := DecodeClearPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.ClearPayloadCASAt(name, id, cas, ms)
	} else {
		applied, _, err = tx.vectors.ClearPayloadCAS(name, id, cas)
	}
	if err != nil {
		return nil, err
	}
	return EncodePayloadResult(applied), nil
}

func handleVectorSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	sc := vectorSearchPool.Get().(*vectorSearchScratch)
	defer vectorSearchPool.Put(sc)

	name, k, query, filter, err := DecodeVectorSearchArgsInto(args, sc.query)
	if err != nil {
		return nil, err
	}
	sc.query = query // retain the (possibly regrown) buffer for reuse
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// SearchInto writes into the pooled result slice (zero-alloc engine path);
	// EncodeVectorSearchResults copies it into the response buffer before the
	// deferred Put recycles the scratch.
	results, err := c.SearchInto(sc.results[:0], query, k, filter)
	if err != nil {
		return nil, err
	}
	sc.results = results
	return EncodeVectorSearchResults(results), nil
}
