// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"fmt"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops/wire"
	"github.com/rostamlabs/rostam/vector"
)

// handleVectorQuery is the vector_query op handler: decode the args, acquire the
// collection, proto-unmarshal the spec blob, convert proto→engine struct, run
// (*Collection).Query, and encode the mode-tagged result.
//
// GROUPED query (spec.GroupBy != ""): the handler is a PARTITION LEAF in the grouped
// fan-out (the coordinator re-fans the spec verbatim with GroupBy set), so it runs the
// flat dense pipeline over the WIDE candidate pool via QueryGroupedFanOut and returns
// the UNGROUPED flat result + per-id group-key map (wire.EncodeQueryResultGroupedFanOut).
// Grouping happens ONCE at the coordinator (both P>1 and the single-shard P==1 path),
// never on the partition — mirroring how FUSION returns unfused lanes and the
// coordinator fuses. A non-grouped query is byte-identical to before.
func handleVectorQuery(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, specBytes, _, _, _, err := wire.DecodeQueryArgs(args)
	if err != nil {
		return nil, err
	}
	var pbSpec pb.QuerySpec
	if uerr := proto.Unmarshal(specBytes, &pbSpec); uerr != nil {
		return nil, fmt.Errorf("ops: decode query spec: %w", uerr)
	}
	spec, cerr := wire.QuerySpecFromProto(&pbSpec, 0)
	if cerr != nil {
		return nil, cerr
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	if spec.GroupBy != "" {
		qr, keys, gerr := c.QueryGroupedFanOut(spec)
		if gerr != nil {
			return nil, gerr
		}
		return wire.EncodeQueryResultGroupedFanOut(qr, keys), nil
	}
	// NESTED MULTI-lane FUSION: ship the per-partition UNFUSED tree-lanes (the flat
	// pre-order spec-tree lane list) instead of pre-fusing nested FUSION nodes, so the
	// coordinator folds EVERY FUSION node over the cross-partition GLOBAL union ⇒
	// P>1==P1 EXACT. The codec choice is the PURE spec shape (SpecHasNestedFusion),
	// evaluated identically on the coordinator (which re-walks the SAME spec) — no wire
	// flag. A spec WITHOUT a nested multi-lane FUSION node falls through to the flat
	// Query path BYTE-IDENTICALLY.
	if vector.SpecHasNestedFusion(spec) {
		lanes, lerr := c.QueryTreeLanes(spec)
		if lerr != nil {
			return nil, lerr
		}
		return wire.EncodeQueryTreeLanes(lanes), nil
	}
	qr, qerr := c.Query(spec)
	if qerr != nil {
		return nil, qerr
	}
	return wire.EncodeQueryResult(qr), nil
}

// handleNamedQuery is the vector_named_query op handler: the NAMED-collection
// analogue of handleVectorQuery. It decodes the args, proto-unmarshals + converts
// the spec (QuerySpecFromProto now handles the NamedDense / NamedSparse oneof
// arms), then runs the query against the NAMED collection via
// CollectionStore.NamedQuery (which acquires the *NamedCollection with a ref held,
// mirroring how handleNamedHybridSearch reaches the named collection) and encodes
// the SAME mode-tagged result wire.EncodeQueryResult produces — so the named result is
// decoded by the exact dense reader (wire.DecodeQueryResult), and the Task-2 fan-out
// coordinator reuses the v1 merge verbatim. Every leaf must carry a named space;
// the engine fails loud otherwise.
func handleNamedQuery(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, specBytes, _, _, _, err := wire.DecodeQueryArgs(args)
	if err != nil {
		return nil, err
	}
	var pbSpec pb.QuerySpec
	if uerr := proto.Unmarshal(specBytes, &pbSpec); uerr != nil {
		return nil, fmt.Errorf("ops: decode query spec: %w", uerr)
	}
	spec, cerr := wire.QuerySpecFromProto(&pbSpec, 0)
	if cerr != nil {
		return nil, cerr
	}
	// NESTED MULTI-lane FUSION: ship the per-partition UNFUSED tree-lanes (the flat
	// pre-order spec-tree lane list) instead of pre-fusing nested FUSION nodes, so the
	// coordinator folds EVERY FUSION node over the cross-partition GLOBAL union ⇒
	// P>1==P1 EXACT. Mirrors handleVectorQuery; the codec choice is the PURE spec shape
	// (SpecHasNestedFusion), evaluated identically on the coordinator. A spec WITHOUT a
	// nested multi-lane FUSION node falls through to the flat NamedQuery path
	// BYTE-IDENTICALLY.
	if vector.SpecHasNestedFusion(spec) {
		lanes, lerr := tx.vectors.NamedQueryTreeLanes(name, spec)
		if lerr != nil {
			return nil, lerr
		}
		return wire.EncodeQueryTreeLanes(lanes), nil
	}
	qr, qerr := tx.vectors.NamedQuery(name, spec)
	if qerr != nil {
		return nil, qerr
	}
	return wire.EncodeQueryResult(qr), nil
}

// handleMVQuery is the vector_mv_query op handler: the MULTI-VECTOR analogue of
// handleVectorQuery / handleNamedQuery. It decodes the args, proto-unmarshals +
// converts the spec (QuerySpecFromProto now handles the mv_maxsim oneof arm + the
// MV-family SparseLeaf reuse), then runs the query against the MV index via
// CollectionStore.MultiQuery (which acquires the *MultiVectorIndex with a ref held,
// mirroring how handleMVHybridSearch reaches the index) and encodes the SAME
// mode-tagged result wire.EncodeQueryResult produces — so the MV result is decoded by
// the exact dense reader (wire.DecodeQueryResult), and the Task-2 fan-out coordinator
// reuses the merge verbatim. Prefetch leaves are MaxSim and/or the doc sparse
// field; no leaf may carry a space (the engine fails loud otherwise).
func handleMVQuery(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, specBytes, _, _, _, err := wire.DecodeQueryArgs(args)
	if err != nil {
		return nil, err
	}
	var pbSpec pb.QuerySpec
	if uerr := proto.Unmarshal(specBytes, &pbSpec); uerr != nil {
		return nil, fmt.Errorf("ops: decode query spec: %w", uerr)
	}
	spec, cerr := wire.QuerySpecFromProto(&pbSpec, 0)
	if cerr != nil {
		return nil, cerr
	}
	// NESTED MULTI-lane FUSION: ship the per-partition UNFUSED tree-lanes so the
	// coordinator folds EVERY FUSION node over the cross-partition GLOBAL union ⇒
	// P>1==P1 EXACT (all MV lanes score-desc; the orientation-aware coordinator fold
	// handles them). Mirrors handleVectorQuery / handleNamedQuery; a spec WITHOUT a
	// nested multi-lane FUSION node falls through to the flat MultiQuery path
	// BYTE-IDENTICALLY.
	if vector.SpecHasNestedFusion(spec) {
		lanes, lerr := tx.vectors.MultiQueryTreeLanes(name, spec)
		if lerr != nil {
			return nil, lerr
		}
		return wire.EncodeQueryTreeLanes(lanes), nil
	}
	qr, qerr := tx.vectors.MultiQuery(name, spec)
	if qerr != nil {
		return nil, qerr
	}
	return wire.EncodeQueryResult(qr), nil
}
