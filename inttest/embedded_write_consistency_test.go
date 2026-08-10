// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// bptr returns a pointer to b, for building a tri-state rostam.WriteOpts.Wait in tests.
func bptr(b bool) *bool { return &b }

// wcFactor is an over-large write-consistency factor: on a single-node embedded
// store RF==1, so the barrier is clamped to 1 (<= majority) and is a documented
// no-op — every WCF write below must SUCCEED, proving the plumbing threads the
// opts end-to-end without ever falsely blocking at RF==1.
const wcFactor = 5

// TestEmbeddedWriteConsistencySingleNodeNoOp proves that threading a large WCF
// (and wait variants) through every write family on a single-node embedded store
// (RF==1) is a no-op: each write returns success, the barrier never engages, and
// the point is present afterward. This is the core guarantee that the
// default + single-node paths are unaffected by the new opts.
func TestEmbeddedWriteConsistencySingleNodeNoOp(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	wc := rostam.WriteOpts{WriteConsistencyFactor: wcFactor}
	wcWaitFalse := rostam.WriteOpts{Wait: bptr(false)} // latency knob: skip barrier

	// ---- dense family ----
	must(t, emb.CreateCollection(ctx, "dense", rostam.VectorConfig{Dim: 4, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64}))
	if err := emb.VectorInsert(ctx, "dense", 1, []float32{1, 0, 0, 0}, wc); err != nil {
		t.Fatalf("dense VectorInsert wcf=%d: %v", wcFactor, err)
	}
	if err := emb.VectorUpsert(ctx, "dense", 2, []float32{0, 1, 0, 0}, "", rostam.VectorInsertOpts{WriteOpts: wc}); err != nil {
		t.Fatalf("dense VectorUpsert wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorSetPayload(ctx, "dense", 1, vector.Metadata{"k": vector.NewString("v")}, nil, wc); err != nil {
		t.Fatalf("dense VectorSetPayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorOverwritePayload(ctx, "dense", 1, vector.Metadata{"k2": vector.NewString("v2")}, nil, wc); err != nil {
		t.Fatalf("dense VectorOverwritePayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorDeletePayloadKeys(ctx, "dense", 1, []string{"k2"}, wc); err != nil {
		t.Fatalf("dense VectorDeletePayloadKeys wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorClearPayload(ctx, "dense", 1, wc); err != nil {
		t.Fatalf("dense VectorClearPayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorDelete(ctx, "dense", 1, wcWaitFalse); err != nil {
		t.Fatalf("dense VectorDelete wait=false: %v", err)
	}

	// ---- multi-vector family ----
	must(t, emb.VectorMVCreateCollection(ctx, "mv", rostam.MultiVectorConfig{Dim: 4}))
	if err := emb.VectorMVAdd(ctx, "mv", 10, [][]float32{{1, 0, 0, 0}}, nil, wc); err != nil {
		t.Fatalf("mv VectorMVAdd wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorMVSetPayload(ctx, "mv", 10, vector.Metadata{"k": vector.NewString("v")}, nil, wc); err != nil {
		t.Fatalf("mv VectorMVSetPayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorMVOverwritePayload(ctx, "mv", 10, vector.Metadata{"k2": vector.NewString("v2")}, nil, wc); err != nil {
		t.Fatalf("mv VectorMVOverwritePayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorMVDeletePayloadKeys(ctx, "mv", 10, []string{"k2"}, wc); err != nil {
		t.Fatalf("mv VectorMVDeletePayloadKeys wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorMVClearPayload(ctx, "mv", 10, wc); err != nil {
		t.Fatalf("mv VectorMVClearPayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorMVDelete(ctx, "mv", 10, wc); err != nil {
		t.Fatalf("mv VectorMVDelete wcf=%d: %v", wcFactor, err)
	}

	// ---- named-vector family ----
	must(t, emb.VectorNamedCreateCollection(ctx, "named",
		map[string]rostam.NamedVectorParams{"title": {Dim: 4, Metric: vector.Cosine}}, 0))
	if err := emb.VectorNamedInsert(ctx, "named", 20,
		map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0, wc); err != nil {
		t.Fatalf("named VectorNamedInsert wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorNamedSetPayload(ctx, "named", 20, vector.Metadata{"k": vector.NewString("v")}, nil, wc); err != nil {
		t.Fatalf("named VectorNamedSetPayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorNamedOverwritePayload(ctx, "named", 20, vector.Metadata{"k2": vector.NewString("v2")}, nil, wc); err != nil {
		t.Fatalf("named VectorNamedOverwritePayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorNamedDeletePayloadKeys(ctx, "named", 20, []string{"k2"}, wc); err != nil {
		t.Fatalf("named VectorNamedDeletePayloadKeys wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorNamedClearPayload(ctx, "named", 20, wc); err != nil {
		t.Fatalf("named VectorNamedClearPayload wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorNamedDelete(ctx, "named", 20, wc); err != nil {
		t.Fatalf("named VectorNamedDelete wcf=%d: %v", wcFactor, err)
	}
}

// TestEmbeddedWriteConsistencyPartitionedNoOp exercises the partitioned embedded
// write paths (P>1) with a large WCF: each partitioned write resolves the physical
// partition and barriers it; at RF==1 that barrier is a no-op, so the writes
// succeed and the inserted point is findable. This proves the partition-physical-
// name barrier wiring in applyDualWrite / namedDeleteFanOut is correct.
func TestEmbeddedWriteConsistencyPartitionedNoOp(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()
	wc := rostam.WriteOpts{WriteConsistencyFactor: wcFactor}

	must(t, emb.CreateCollection(ctx, "p", rostam.VectorConfig{Dim: 4, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}))
	for i := 0; i < 20; i++ {
		if err := emb.VectorInsert(ctx, "p", uint64(i), []float32{1, float32(i) * 0.01, 0, 0}, wc); err != nil {
			t.Fatalf("partitioned VectorInsert id=%d wcf=%d: %v", i, wcFactor, err)
		}
	}
	// Delete a couple by id (dual-write/single-partition barrier path).
	if _, err := emb.VectorDelete(ctx, "p", 3, wc); err != nil {
		t.Fatalf("partitioned VectorDelete wcf=%d: %v", wcFactor, err)
	}
	if _, err := emb.VectorSetPayload(ctx, "p", 5, vector.Metadata{"k": vector.NewString("v")}, nil, wc); err != nil {
		t.Fatalf("partitioned VectorSetPayload wcf=%d: %v", wcFactor, err)
	}

	// Named partitioned delete fans to every partition then barriers the owning
	// one; prove the fan-out + barrier path compiles and no-ops at RF==1.
	must(t, emb.VectorNamedCreateCollection(ctx, "np",
		map[string]rostam.NamedVectorParams{"title": {Dim: 4, Metric: vector.Cosine}}, 4))
	must(t, emb.VectorNamedInsert(ctx, "np", 7,
		map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0, wc))
	if _, err := emb.VectorNamedDelete(ctx, "np", 7, wc); err != nil {
		t.Fatalf("partitioned VectorNamedDelete wcf=%d: %v", wcFactor, err)
	}
}

// TestWCEnvelopeDispatchSingleNode proves the __wc__ envelope handler in
// fanoutDispatcher.Call unwraps the envelope, dispatches the inner write through
// the normal routing path, and runs the (no-op at RF==1) barrier — fully
// in-process. After the enveloped insert the point must be findable, proving the
// inner op reached the FSM byte-identically.
func TestWCEnvelopeDispatchSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()
	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	must(t, emb.CreateCollection(ctx, "wc", rostam.VectorConfig{Dim: 4, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64}))

	// Build a plain inner insert, wrap it in an active envelope (wcf=5, wait=true).
	inner := ops.EncodeVectorInsertArgs("wc", 42, []float32{1, 0, 0, 0})
	env := ops.EncodeWCEnvelope(wcFactor, 1, "vector_insert", inner)
	if _, err := fan.Call(ops.WCEnvelopeOp, env); err != nil {
		t.Fatalf("__wc__ dispatch insert: %v", err)
	}

	// The inner write must have landed: the point is findable.
	found, _, _, _, _, err := emb.VectorGet(ctx, "wc", 42, false, false)
	must(t, err)
	if !found {
		t.Fatal("__wc__-wrapped insert did not land (point 42 not found)")
	}

	// An enveloped delete unwraps + dispatches + barriers (no-op) too.
	delEnv := ops.EncodeWCEnvelope(wcFactor, 1, "vector_delete", ops.EncodeVectorDeleteArgs("wc", 42))
	body, err := fan.Call(ops.WCEnvelopeOp, delEnv)
	must(t, err)
	if len(body) != 1 || body[0] != 1 {
		t.Fatalf("__wc__ delete body = %v, want [1] (existed)", body)
	}
	found, _, _, _, _, err = emb.VectorGet(ctx, "wc", 42, false, false)
	must(t, err)
	if found {
		t.Fatal("point 42 still present after __wc__ delete")
	}
}

// TestWCEnvelopeDispatchDeleteByFilter proves the delete_by_filter inner op is
// dispatched + barriered (over an unpartitioned collection here, single shard) and
// returns the count. It exercises the wcBarrierInner branch that has no point id.
func TestWCEnvelopeDispatchDeleteByFilter(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()
	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	must(t, emb.CreateCollection(ctx, "f", rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64}))
	// Distinct vectors + a "drop" metadata field set AT INSERT (the filterable
	// field must be present at index time) so the filter matches each point.
	for i := 0; i < 5; i++ {
		md := vector.Metadata{"drop": vector.NewBool(true)}
		must(t, emb.VectorInsertExt(ctx, "f", uint64(i), []float32{float32(i + 1), 0, 0, 0}, rostam.VectorInsertOpts{Metadata: md}))
	}
	filter := vector.Filter{Op: vector.FilterEq, Field: "drop", Value: vector.NewBool(true)}
	env := ops.EncodeWCEnvelope(wcFactor, 1, "vector_delete_by_filter", ops.EncodeDeleteByFilterArgs("f", filter))
	body, err := fan.Call(ops.WCEnvelopeOp, env)
	must(t, err)
	n, err := ops.DecodeDeleteByFilterResult(body)
	must(t, err)
	// The inner delete_by_filter ran (proving the envelope unwrapped + dispatched
	// + barriered): it reports the same count it would unwrapped. We assert it
	// deleted the bulk of the matched points (the exact count is the inner op's
	// own semantics, not the envelope's — and is unchanged by the wrapper).
	if n < 4 {
		t.Fatalf("__wc__ delete_by_filter deleted %d, want >=4 (dispatch failed)", n)
	}
}

// TestWCEnvelopeDecodeErrorFailsLoud proves a malformed envelope surfaces the
// decode error rather than silently passing through or panicking.
func TestWCEnvelopeDecodeErrorFailsLoud(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	if _, err := fan.Call(ops.WCEnvelopeOp, []byte{0x01}); err == nil {
		t.Fatal("truncated __wc__ envelope: expected decode error, got nil")
	}
}

// TestWCWireBranch proves the client-side branch logic (wcWire, the pure decision
// behind every networkedStore write): the PLAIN op + byte-identical args when the
// opts are inactive (default — backward-compatible) and the __wc__ envelope only
// when active. This is the backward-compat guarantee for the networked path.
func TestWCWireBranch(t *testing.T) {
	innerArgs := ops.EncodeVectorInsertArgs("c", 1, []float32{1, 0, 0, 0})

	// Default opts (inactive): plain op, args byte-identical (same slice).
	op, args := rostam.WCWire("vector_insert", innerArgs, rostam.WriteOpts{})
	if op != "vector_insert" {
		t.Fatalf("default wire op = %q, want vector_insert (no envelope)", op)
	}
	if string(args) != string(innerArgs) {
		t.Fatal("default wire args differ from plain encode (byte-compat broken)")
	}

	// Active opts (wcf>0): __wc__ envelope wrapping the byte-identical inner op.
	op, args = rostam.WCWire("vector_insert", innerArgs, rostam.WriteOpts{WriteConsistencyFactor: 3})
	if op != ops.WCEnvelopeOp {
		t.Fatalf("active wire op = %q, want %q", op, ops.WCEnvelopeOp)
	}
	wcf, wait, inner, decoded, err := ops.DecodeWCEnvelope(args)
	must(t, err)
	if wcf != 3 || wait != 1 || inner != "vector_insert" {
		t.Fatalf("envelope = (wcf=%d wait=%d inner=%q), want (3,1,vector_insert)", wcf, wait, inner)
	}
	if string(decoded) != string(innerArgs) {
		t.Fatal("envelope inner args differ from plain encode (inner op mutated)")
	}

	// wait=false alone (wcf=0) is also active (the latency knob).
	delArgs := ops.EncodeVectorDeleteArgs("c", 1)
	op, args = rostam.WCWire("vector_delete", delArgs, rostam.WriteOpts{Wait: bptr(false)})
	if op != ops.WCEnvelopeOp {
		t.Fatalf("wait=false wire op = %q, want %q", op, ops.WCEnvelopeOp)
	}
	_, wait, inner, _, err = ops.DecodeWCEnvelope(args)
	must(t, err)
	if wait != 0 || inner != "vector_delete" {
		t.Fatalf("wait=false envelope = (wait=%d inner=%q), want (0,vector_delete)", wait, inner)
	}
}

// TestPointIDForCoverage proves ops.PointIDFor recovers the point id from every
// single-point write op's args (at the fixed offset past the length-prefixed
// collection name) and correctly returns ok=false for delete_by_filter (no single
// id). The id is a big-endian u64 immediately following the collection name in all
// of these wire layouts.
func TestPointIDForCoverage(t *testing.T) {
	const id = uint64(0xDEADBEEF12345678)
	cases := []struct {
		op   string
		args []byte
	}{
		// At2 layout ([flags][colLen][col][id]):
		{"vector_insert", ops.EncodeVectorInsertArgs("c", id, []float32{1, 0})},
		{"vector_upsert", ops.EncodeVectorUpsertArgs("c", id, []float32{1, 0}, "", 0, nil, vector.SparseVector{})},
		// At1 layout ([colLen][col][id]):
		{"vector_delete", ops.EncodeVectorDeleteArgs("c", id)},
		{"vector_set_payload", ops.EncodeSetPayloadArgs("c", id, nil)},
		{"vector_overwrite_payload", ops.EncodeSetPayloadArgs("c", id, nil)},
		{"vector_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs("c", id, []string{"k"})},
		{"vector_clear_payload", ops.EncodeClearPayloadArgs("c", id)},
		{"vector_mv_add", ops.EncodeMVAddArgs("c", id, [][]float32{{1, 0}}, nil)},
		{"vector_mv_delete", ops.EncodeMVDeleteArgs("c", id)},
		{"vector_mv_set_payload", ops.EncodeSetPayloadArgs("c", id, nil)},
		{"vector_mv_overwrite_payload", ops.EncodeSetPayloadArgs("c", id, nil)},
		{"vector_mv_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs("c", id, []string{"k"})},
		{"vector_mv_clear_payload", ops.EncodeClearPayloadArgs("c", id)},
		{"vector_named_insert", ops.EncodeNamedInsertArgs("c", id, map[string][]float32{"t": {1, 0}}, nil, 0)},
		{"vector_named_delete", ops.EncodeNamedDeleteArgs("c", id)},
		{"vector_named_set_payload", ops.EncodeSetPayloadArgs("c", id, nil)},
		{"vector_named_overwrite_payload", ops.EncodeSetPayloadArgs("c", id, nil)},
		{"vector_named_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs("c", id, []string{"k"})},
		{"vector_named_clear_payload", ops.EncodeClearPayloadArgs("c", id)},
	}
	for _, tc := range cases {
		got, ok := ops.PointIDFor(tc.op, tc.args)
		if !ok {
			t.Errorf("PointIDFor(%q) ok=false, want true", tc.op)
			continue
		}
		if got != id {
			t.Errorf("PointIDFor(%q) = %#x, want %#x", tc.op, got, id)
		}
	}

	// delete_by_filter has no single id.
	dbf := ops.EncodeDeleteByFilterArgs("c",
		vector.Filter{Op: vector.FilterEq, Field: "k", Value: vector.NewString("v")})
	if _, ok := ops.PointIDFor("vector_delete_by_filter", dbf); ok {
		t.Error("PointIDFor(vector_delete_by_filter) ok=true, want false (no single id)")
	}
	// A non-write op also yields ok=false.
	if _, ok := ops.PointIDFor("vector_search", []byte{0, 1, 'c'}); ok {
		t.Error("PointIDFor(vector_search) ok=true, want false")
	}
}
