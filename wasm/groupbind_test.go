// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// txForGroup builds a TxContext that reports idx as its dispatcher's shard
// group, which is what the resolver keys on. The cache is nil because nothing
// below invokes anything — only the version resolution is under test.
func txForGroup(idx int) *ops.TxContext {
	tx := ops.NewTxContext(nil)
	tx.SetShardIndex(idx)
	return tx
}

func idOf(s string) ModuleID { return ModuleIDFor([]byte(s), "apply", 0) }

// TestResolveModuleForInvokeUsesTheGroupBinding is the apply-side half of
// per-group version binding: the version that executes a committed entry comes
// from the entry's OWN shard group, not from whatever this node last installed
// under the name.
//
// The scenario is the one the whole design exists for. An update reached group 0
// and has not reached group 1. The node-wide binding — what the ops.Registry
// entry captured — is v2, because this node applied the update from group 0. An
// entry committed in GROUP 1 must still execute with v1, because that is what
// group 1's log has committed, and because a peer that does not host group 0 at
// all has no way to know about v2. If both nodes did not answer v1 here, they
// would apply the same committed entry with different bytes: two successful
// applies, different results, no error, permanent divergence.
func TestResolveModuleForInvokeUsesTheGroupBinding(t *testing.T) {
	v1, v2 := idOf("v1"), idOf("v2")
	rt := &Runtime{}
	rt.PublishGroupBindings(GroupBindings{
		"udf": {0: v2, 1: v1},
	})

	got, err := resolveBinding(rt, "udf", ops.OpReadWrite, v2, txForGroup(1))
	if err != nil {
		t.Fatalf("resolve for group 1: %v", err)
	}
	if got != v1 {
		t.Errorf("group 1 resolved to %s, want the version ITS log committed (%s). The node-wide binding was %s — resolving from it is exactly the silent divergence per-group binding removes",
			got, v1, v2)
	}
	if got, err := resolveBinding(rt, "udf", ops.OpReadWrite, v2, txForGroup(0)); err != nil || got != v2 {
		t.Errorf("group 0 resolved to %s (err %v), want %s", got, err, v2)
	}
}

// TestResolveModuleForInvokeMissIsAnError pins the failure class, which is the
// half of the design that must NOT be forgiving.
//
// The propose-time route gate guarantees a registration for an op sits BELOW
// every invocation of it in the same group's log, and both catch-up routes
// reconstruct the binding from that. So a replica that reaches a committed
// invocation with no binding for its group holds state that disagrees with what
// that group's log says: its peers have the binding and WILL execute the entry.
// Falling back to the node-wide version would execute it with a version the
// group's log never named — the divergence again, by a different door. It must
// fail, and the sentinel must be the one shard/apply_class.go classifies fatal.
func TestResolveModuleForInvokeMissIsAnError(t *testing.T) {
	v1 := idOf("v1")
	rt := &Runtime{}
	rt.PublishGroupBindings(GroupBindings{"udf": {0: v1}})

	got, err := resolveBinding(rt, "udf", ops.OpReadWrite, v1, txForGroup(3))
	if err == nil {
		t.Fatalf("a replicated op with no binding for the applying group resolved to %s instead of failing; the FSM would execute an entry with a version that group's log never named", got)
	}
	if !errors.Is(err, ops.ErrWASMNoGroupBinding) {
		t.Errorf("miss returned %v; it must carry ops.ErrWASMNoGroupBinding or shard.classifyApplyErr advances the applied index over an entry peers executed", err)
	}
	if got != (ModuleID{}) {
		t.Errorf("a failed resolution returned a usable ModuleID %s; a caller ignoring the error would run it", got)
	}
}

// resolveBinding exercises the BINDING half of the resolution — cases (1)..(4)
// — without the residency and kind guards layered on top of it.
//
// The split is what these tests are about. resolveModuleForInvoke answers two
// independent questions: WHICH version does this group execute (a pure function
// of the published table, which is what every test below asserts) and IS that
// version instantiated here (a fact about local byte residency, which thin
// markers made a real, routine, self-healing condition). A bare &Runtime{} holds
// no modules at all, so folding them together would make every binding assertion
// fail for a reason that has nothing to do with binding. The residency and kind
// layer has its own gate: TestResolveRefusesANonResidentOrWritingModule.
func resolveBinding(rt *Runtime, opName string, kind ops.OpKind, bound ModuleID, tx *ops.TxContext) (ModuleID, error) {
	id, _, err := rt.bindModuleForInvoke(opName, kind, bound, tx)
	return id, err
}

// TestResolveModuleForInvokeFallsBackExplicitly pins the four cases that
// deliberately answer with the NODE-WIDE version. Each is a live, correct path
// today, and turning any of them into the miss error above would break it.
func TestResolveModuleForInvokeFallsBackExplicitly(t *testing.T) {
	bound := idOf("node-wide")
	other := idOf("group-bound")

	t.Run("no table published", func(t *testing.T) {
		// The single-node Direct backend (rostam.directStore) shares this type but
		// has no Raft groups and never publishes a table. Its TxContext still
		// reports a real shard index, so this case cannot be left to fall out of the
		// map lookups.
		rt := &Runtime{}
		got, err := resolveBinding(rt, "udf", ops.OpReadWrite, bound, txForGroup(2))
		if err != nil || got != bound {
			t.Errorf("got (%s, %v), want the node-wide binding %s: Direct mode would fail every WASM call", got, err, bound)
		}
	})

	t.Run("op not in the table", func(t *testing.T) {
		// An operator-configured module (cfg.WASMModules). It was never proposed
		// into anyone's log, so there is no group prefix to bind it to.
		rt := &Runtime{}
		rt.PublishGroupBindings(GroupBindings{"replicated": {0: other}})
		got, err := resolveBinding(rt, "config_udf", ops.OpReadWrite, bound, txForGroup(0))
		if err != nil || got != bound {
			t.Errorf("got (%s, %v), want the node-wide binding %s: every operator-configured WASM op would fail", got, err, bound)
		}
	})

	t.Run("no group provenance", func(t *testing.T) {
		rt := &Runtime{}
		rt.PublishGroupBindings(GroupBindings{"udf": {0: other}})
		got, err := resolveBinding(rt, "udf", ops.OpReadWrite, bound, txForGroup(ops.NoShardIndex))
		if err != nil || got != bound {
			t.Errorf("got (%s, %v), want the node-wide binding %s", got, err, bound)
		}
		// A nil TxContext reports NoShardIndex too.
		if got, err := resolveBinding(rt, "udf", ops.OpReadWrite, bound, nil); err != nil || got != bound {
			t.Errorf("nil tx: got (%s, %v), want %s", got, err, bound)
		}
	})

	t.Run("read-only op", func(t *testing.T) {
		// shard.Store.Call serves an OpReadOnly op from local state WITHOUT
		// proposing anything, so no group's log carries the call and there is
		// nothing to bind. The route gate does not gate read-only ops for the same
		// reason; resolving them per group would newly fail every read served by a
		// group the registration never reached.
		rt := &Runtime{}
		rt.PublishGroupBindings(GroupBindings{"udf": {0: other}})
		got, err := resolveBinding(rt, "udf", ops.OpReadOnly, bound, txForGroup(5))
		if err != nil || got != bound {
			t.Errorf("got (%s, %v), want the node-wide binding %s: a read-only WASM op would stop being servable from any group the registration had not reached — a path that is live, ungated and safe today", got, err, bound)
		}
	})
}

// TestPublishGroupBindingsReplacesAndClears pins the publication contract the
// copy-on-write reader depends on.
func TestPublishGroupBindingsReplacesAndClears(t *testing.T) {
	a, b := idOf("a"), idOf("b")
	rt := &Runtime{}
	if rt.groupBindings() != nil {
		t.Fatal("a fresh Runtime must have no published table")
	}
	rt.PublishGroupBindings(GroupBindings{"udf": {0: a}})
	if got := rt.groupBindings()["udf"][0]; got != a {
		t.Fatalf("published table holds %s, want %s", got, a)
	}
	rt.PublishGroupBindings(GroupBindings{"udf": {0: b}})
	if got := rt.groupBindings()["udf"][0]; got != b {
		t.Errorf("republishing did not replace the table: %s", got)
	}
	rt.PublishGroupBindings(nil)
	if rt.groupBindings() != nil {
		t.Error("publishing nil must clear the table, returning every resolution to the node-wide binding")
	}
}

// TestResolveRefusesANonResidentOrWritingModule pins the layer thin markers
// added on top of the binding: a resolution that lands on a version this node
// does not HOLD, and one that lands on a module whose imports contradict the
// op's declared Kind.
//
// It applies to ALL FOUR binding cases, not only the per-group one, which is why
// it is asserted through resolveModuleForInvoke rather than through
// resolveBinding. A node-wide fallback names a version just as fetchable — and
// just as possibly-absent — as a group binding does.
//
// The not-resident answer must be ops.ErrWASMModuleNotResident and must carry the
// coordinates the fetcher needs, because shard.classifyApplyErr maps that
// sentinel to classRetry (wait and re-run) while the sentinel it most resembles,
// ops.ErrWASMNoGroupBinding, is classFatal (halt). Getting the two confused in
// either direction is a node that crash-loops or a replica that diverges.
func TestResolveRefusesANonResidentOrWritingModule(t *testing.T) {
	t.Run("not resident", func(t *testing.T) {
		// A bare Runtime holds no modules, so any binding resolves to a version it
		// does not have — exactly a node that applied a marker before its fetch
		// landed.
		id := idOf("never-fetched")
		rt := &Runtime{}
		rt.PublishGroupBindings(GroupBindings{"udf": {3: id}})
		got, err := rt.resolveModuleForInvoke("udf", ops.OpReadWrite, id, txForGroup(3))
		if err == nil {
			t.Fatalf("a binding to a module this node does not hold resolved to %s; Invoke would then fail on an unknown module instead of the apply waiting for the bytes", got)
		}
		if !errors.Is(err, ops.ErrWASMModuleNotResident) {
			t.Fatalf("got %v; it must carry ops.ErrWASMModuleNotResident or shard.classifyApplyErr will not retry — it would either advance past an entry peers execute, or halt on a self-healing condition", err)
		}
		var nre *ops.WASMNotResidentError
		if !errors.As(err, &nre) {
			t.Fatal("the error must be the typed carrier: the retry hook recovers the fingerprint to fetch from its (op, group), and a string-parsed one breaks silently on the first rewording")
		}
		if nre.Op != "udf" || nre.Group != 3 {
			t.Fatalf("carrier holds (%q, %d), want (udf, 3): those are what the fetcher resolves to a blob address", nre.Op, nre.Group)
		}
		if got != (ModuleID{}) {
			t.Errorf("a failed resolution returned a usable ModuleID %s; a caller ignoring the error would run it", got)
		}
	})

	t.Run("node-wide fallback is covered too", func(t *testing.T) {
		// Case (1): no table published at all — Direct mode's path. The version is
		// still named by something, and can still be absent.
		id := idOf("never-fetched")
		rt := &Runtime{}
		if _, err := rt.resolveModuleForInvoke("udf", ops.OpReadWrite, id, txForGroup(2)); !errors.Is(err, ops.ErrWASMModuleNotResident) {
			t.Fatalf("a node-wide fallback to an absent module gave %v; the residency guard must cover every binding case, not only the per-group one", err)
		}
	})
}
