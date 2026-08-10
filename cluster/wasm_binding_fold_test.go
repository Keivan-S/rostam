// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// testBindings builds a binding table that binds r in each of gs. It is the
// shorthand for "these groups' logs carry exactly this registration", which is
// what the pre-per-group-binding tests expressed as a bare map[int]struct{}.
func testBindings(r ops.WASMRegistration, gs ...int) map[int]wasmGroupBinding {
	if len(gs) == 0 {
		return nil
	}
	out := make(map[int]wasmGroupBinding, len(gs))
	for _, g := range gs {
		out[g] = newWASMGroupBinding(r)
	}
	return out
}

// wasmModuleIDOf is the ModuleID a registration resolves to — what a group's
// binding holds and what actually executes.
func wasmModuleIDOf(r ops.WASMRegistration) wasm.ModuleID {
	return wasm.ModuleIDForBlob(r.Blob, r.ExportName, r.MaxFuel)
}

// wasmMarker is one (shard group, registration) pair: a __register_wasm__ entry
// as it reaches the fold, tagged with the group whose log carried it.
type wasmMarker struct {
	group int
	reg   ops.WASMRegistration
}

// foldMarkers runs the REAL fold (installedWASM.bindWith) over a marker sequence
// and renders the resulting op → group → version table in a comparable form.
//
// It deliberately drives the production method rather than reimplementing it: a
// model that agreed with itself would prove nothing.
func foldMarkers(markers []wasmMarker) map[string]map[int]string {
	st := make(map[string]installedWASM, 4)
	for _, m := range markers {
		in := st[m.reg.Name]
		groups, _ := in.bindWith(m.group, m.reg)
		in.groups = groups
		st[m.reg.Name] = in
	}
	out := make(map[string]map[int]string, len(st))
	for name, in := range st {
		if len(in.groups) == 0 {
			continue
		}
		byGroup := make(map[int]string, len(in.groups))
		for g, b := range in.groups {
			byGroup[g] = b.id.String()
		}
		out[name] = byGroup
	}
	return out
}

// renderBindingTable flattens a fold result into a single deterministic string,
// so a permutation mismatch reports WHAT differs rather than "not deep-equal".
func renderBindingTable(tbl map[string]map[int]string) string {
	names := make([]string, 0, len(tbl))
	for name := range tbl {
		names = append(names, name)
	}
	sort.Strings(names)
	s := ""
	for _, name := range names {
		groups := make([]int, 0, len(tbl[name]))
		for g := range tbl[name] {
			groups = append(groups, g)
		}
		sort.Ints(groups)
		for _, g := range groups {
			s += fmt.Sprintf("%s@%d=%s;", name, g, tbl[name][g][:12])
		}
	}
	return s
}

// permuteMarkers calls fn with every permutation of in (Heap's algorithm).
func permuteMarkers(in []wasmMarker, fn func([]wasmMarker) bool) bool {
	work := append([]wasmMarker(nil), in...)
	var rec func(k int) bool
	rec = func(k int) bool {
		if k == 1 {
			return fn(work)
		}
		for i := 0; i < k; i++ {
			if !rec(k - 1) {
				return false
			}
			if k%2 == 0 {
				work[i], work[k-1] = work[k-1], work[i]
			} else {
				work[0], work[k-1] = work[k-1], work[0]
			}
		}
		return true
	}
	if len(work) == 0 {
		return fn(work)
	}
	return rec(len(work))
}

// markerReg is a registration whose only distinguishing content is `body`, so a
// test can name versions readably.
//
// EVERY marker it builds shares ONE CONTRACT — OpReadWrite with the one legal key
// extractor — which is exactly why a permutation test built on it can never enter
// checkWASMGroupContract's refusal branch. That is deliberate here and it is the
// scope limit of TestWASMBindingFoldIsPermutationInvariantWithinOneContract. Tests
// that need the contract-differing case build their registrations explicitly; see
// TestWASMBindingIsPrefixDeterministic.
func markerReg(name, body string, epoch uint64) ops.WASMRegistration {
	return ops.WASMRegistration{
		Name:       name,
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint([]byte(body)),
		ExportName: "apply",
		Epoch:      epoch,
	}
}

// TestWASMBindingFoldIsPermutationInvariantWithinOneContract pins what
// installedWASM.bindWith ALONE gives, and its scope is exactly as narrow as its
// name says. READ THIS BEFORE TREATING IT AS THE GATE FOR PER-GROUP BINDING — it
// is not, and an earlier version of this file (which called it
// TestWASMBindingFoldIsPermutationInvariant and described it as "THE gate")
// claimed it was.
//
// WHAT IT PROVES. bindWith is the (Epoch, fingerprint) MAXIMUM over the
// registrations of a name in a group's prefix, and a maximum over a set is
// commutative and associative — so every permutation of a marker set folds to the
// same table. markerReg hardcodes one Kind, so every
// registration here shares one CONTRACT, which is the case where that holds.
//
// WHAT IT DOES NOT PROVE. The fold that runs in production is bindWith COMPOSED
// WITH checkWASMGroupContract (applyWASMRegistration), and the composition is
// order-DEPENDENT: the contract check has nothing to compare a group's FIRST
// registration against, so that first registration pins the group's contract.
// Feed this fold two registrations with different extractors and the two orders
// give two different bindings. Nothing in this test enters that branch — every
// marker it builds shares one contract — so it cannot see it.
//
// The property the design actually rests on is PREFIX-DETERMINISM: the same
// ORDERED log prefix yields the same binding on every replica of the group. That
// is what TestWASMBindingIsPrefixDeterministic gates, over the real
// applyWASMRegistration path and over a marker set that DOES change the contract.
// This test remains worth keeping as the narrower statement it is: within one
// contract the maximum really is permutation-invariant, which is what makes an
// in-place BYTES update converge no matter how a group's applies interleave with
// a snapshot re-delivery.
//
// WHAT A FAILURE HERE MEANS. Two replicas of one group that received the same
// same-contract registrations in different orders would execute the same
// committed entry with different bytes. Both applies succeed; there is no error
// to classify and no halt; the two replicas simply write different values.
func TestWASMBindingFoldIsPermutationInvariantWithinOneContract(t *testing.T) {
	v1 := markerReg("udf", "v1", 1)
	v2 := markerReg("udf", "v2", 2)
	v3 := markerReg("udf", "v3", 3)
	// tieA and tieB share an Epoch, so the fold must fall through to the content
	// fingerprint. That tiebreak is the part an "install whatever arrived last"
	// rule would get wrong without ever looking wrong.
	tieA := markerReg("udf", "tie-a", 5)
	tieB := markerReg("udf", "tie-b", 5)
	other := markerReg("other", "o1", 1)

	cases := []struct {
		name    string
		markers []wasmMarker
	}{
		{
			// The plain update: v1 broadcast to two groups, then v2 to both. Whatever
			// order the four applies interleave in, both groups end on v2.
			name: "update reaches both groups",
			markers: []wasmMarker{
				{0, v1}, {1, v1}, {0, v2}, {1, v2},
			},
		},
		{
			// THE CASE THE DESIGN EXISTS FOR: the update reached group 0 and never
			// reached group 1. Group 0 must end on v2 and group 1 on v1, in every
			// order — a node-wide answer cannot express this at all.
			name: "update reaches one group only",
			markers: []wasmMarker{
				{0, v1}, {1, v1}, {0, v2},
			},
		},
		{
			// Three versions, unevenly distributed, plus a second op sharing the
			// table. 6! = 720 orders.
			name: "three versions across three groups plus a second op",
			markers: []wasmMarker{
				{0, v1}, {1, v2}, {2, v3}, {0, v3}, {1, v1}, {2, other},
			},
		},
		{
			// An Epoch tie in one group: the fingerprint tiebreak decides, and it must
			// decide the same way from either arrival order.
			name: "epoch tie resolved by fingerprint",
			markers: []wasmMarker{
				{0, tieA}, {0, tieB}, {1, tieA},
			},
		},
		{
			// A stale re-delivery (the snapshot-then-log and log-then-snapshot pair):
			// an older registration arriving after a newer one must not move the
			// binding backwards.
			name: "stale re-delivery after a newer one",
			markers: []wasmMarker{
				{0, v3}, {0, v1}, {0, v2}, {1, v2}, {1, v3},
			},
		},
		{
			// ops.NoShardIndex markers bind nothing and must not perturb the fold —
			// they are the snapshot install section and the no-dispatcher test path.
			name: "unattributed markers bind nothing",
			markers: []wasmMarker{
				{ops.NoShardIndex, v3}, {0, v1}, {ops.NoShardIndex, v2}, {0, v2},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := renderBindingTable(foldMarkers(tc.markers))
			perms := 0
			ok := permuteMarkers(tc.markers, func(p []wasmMarker) bool {
				perms++
				got := renderBindingTable(foldMarkers(p))
				if got != want {
					order := make([]string, 0, len(p))
					for _, m := range p {
						order = append(order, fmt.Sprintf("g%d:%s", m.group, ops.WASMBlobHex(m.reg.Blob)[:12]))
					}
					t.Errorf("bindWith is ORDER-DEPENDENT for SAME-CONTRACT registrations, which would mean it is not a maximum at all.\n  order   %v\n  gives   %s\n  want    %s\ntwo replicas of one group that received these registrations in different orders would execute the same committed entry with different bytes, silently. (Order dependence ACROSS contracts is expected and is gated by TestWASMBindingIsPrefixDeterministic.)",
						order, got, want)
					return false
				}
				return true
			})
			if !ok {
				return
			}
			if perms == 0 {
				t.Fatal("no permutations were checked")
			}
			t.Logf("%d permutations agree on %s", perms, want)
		})
	}
}

// TestWASMBindingFoldPicksTheMaximum pins WHICH version the same-contract fold
// settles on, since permutation-invariance alone would also be satisfied by a
// fold that always kept the FIRST registration.
//
// It must be the (Epoch, fingerprint) maximum — the same total order the
// node-wide install uses (ops.WASMRegistrationNewer). Keeping the first would
// make an update unable to take effect in a group at all; keeping the last would
// not be permutation-invariant.
func TestWASMBindingFoldPicksTheMaximum(t *testing.T) {
	v1 := markerReg("udf", "v1", 1)
	v2 := markerReg("udf", "v2", 2)
	tieA := markerReg("udf", "tie-a", 5)
	tieB := markerReg("udf", "tie-b", 5)

	var in installedWASM
	groups, changed := in.bindWith(0, v1)
	if !changed {
		t.Fatal("a first registration must establish a binding")
	}
	in.groups = groups
	if got, want := in.groups[0].id, wasm.ModuleIDForBlob(v1.Blob, v1.ExportName, v1.MaxFuel); got != want {
		t.Fatalf("binding = %s, want v1 %s", got, want)
	}

	groups, changed = in.bindWith(0, v2)
	if !changed {
		t.Fatal("a strictly newer registration must move the binding forward, or an update can never take effect in a group")
	}
	in.groups = groups

	if _, changed := in.bindWith(0, v1); changed {
		t.Error("an older registration moved the binding BACKWARDS: a stale re-delivery (snapshot after log replay) would undo an update on one replica and not another")
	}
	if _, changed := in.bindWith(0, v2); changed {
		t.Error("re-applying the SAME registration reported a change: on a k-group node the hook runs k times per registration, and every repeat would rewrite the sidecar")
	}

	// The fingerprint tiebreak must be total and consistent with the node-wide rule.
	hi, lo := tieA, tieB
	if ops.WASMRegistrationNewer(tieB, tieA) {
		hi, lo = tieB, tieA
	}
	var tie installedWASM
	tie.groups, _ = tie.bindWith(0, lo)
	if _, changed := tie.bindWith(0, hi); !changed {
		t.Error("the fingerprint tiebreak did not move the binding: two same-epoch registrations would settle differently depending on arrival order")
	}
	tie.groups, _ = tie.bindWith(0, hi)
	if _, changed := tie.bindWith(0, lo); changed {
		t.Error("the fingerprint tiebreak is not a total order: the fold would oscillate")
	}
}

// applyMarkersTo drives the PRODUCTION fold — applyWASMRegistration, contract
// check, compile, disk writes and all — over a marker sequence.
//
// A per-group contract refusal (ErrWASMUpdateUnsupported) is an EXPECTED outcome
// of a marker set that changes the contract, not a test failure: it is the
// deterministic verdict every replica of that group reaches. Any other error is
// fatal, so a compile failure or a bad name cannot masquerade as a refusal and
// quietly make the assertions below vacuous.
func applyMarkersTo(t *testing.T, r *wasmReplica, markers []wasmMarker) {
	t.Helper()
	for i, m := range markers {
		err := applyWASMRegistration(r.dir, r.rt, r.reg, r.st, m.reg, m.group, nil)
		if err != nil && !errors.Is(err, ErrWASMUpdateUnsupported) {
			t.Fatalf("marker %d (%q epoch %d into group %d): %v", i, m.reg.Name, m.reg.Epoch, m.group, err)
		}
	}
}

// bindingIn renders the version a replica has bound for name in group g, or
// "<unbound>". A string keeps a mismatch readable in a failure message.
func bindingIn(r *wasmReplica, name string, g int) string {
	b, ok := r.st.installed[name].groups[g]
	if !ok {
		return "<unbound>"
	}
	return b.id.String()[:12]
}

// TestWASMBindingIsPrefixDeterministic is THE gate for per-group version
// binding, and it drives the fold that actually ships.
//
// WHY IT EXISTS SEPARATELY FROM THE PERMUTATION TEST. The permutation test
// exercises installedWASM.bindWith alone, over registrations that all share one
// contract. The fold in production is bindWith COMPOSED WITH
// checkWASMGroupContract inside applyWASMRegistration, and that composition is
// NOT commutative: the contract check fires only against an EXISTING binding, so
// a group's FIRST registration is accepted unchecked and pins that group's Kind
// for the rest of its log. Permutation-invariance is
// therefore the WRONG property to assert of it — it is false, and asserting it
// over a contract-identical marker set (which is all the permutation test can
// build) reports a green that means nothing about the shipped path.
//
// THE PROPERTY THAT HOLDS, and the one asserted here:
//
//	PREFIX-DETERMINISM. The binding group g holds for a name is a pure function
//	of the SEQUENCE of registrations of that name in g's log prefix — and of
//	nothing else. Not of which OTHER groups the applying node hosts, not of how
//	far along them it is, not of how those groups' applies interleave with g's.
//
// That is exactly what safety needs, because a Raft log IS a sequence: every
// replica of g has applied the same one. It is strictly weaker than permutation
// invariance and strictly stronger than nothing, and the difference is the whole
// argument.
//
// HOW IT IS TESTED. For each ordering of group 0's log, three replicas with
// deliberately DIFFERENT placements and interleavings apply that same ordering
// into group 0 and must end on the same group-0 binding:
//
//	solo      — hosts group 0 only.
//	before    — hosts groups 0 and 3, and applies all of group 3's traffic FIRST.
//	after     — hosts groups 0 and 3, and applies all of group 3's traffic LAST.
//
// Group 3's traffic carries the highest Epoch in the test, so `before` and
// `after` hold a NODE-WIDE install that `solo` does not — which is precisely the
// contamination per-group binding exists to exclude.
//
// AND IT PINS THE ORDER DEPENDENCE ITSELF, in the second subtest. If every
// ordering happened to agree, this test would have degenerated back into the
// permutation test without anyone noticing: the marker set would no longer be
// entering the refusal branch. So it requires at least two orderings to DISAGREE,
// and pins the exact A,B / B,A counterexample written down in
// installedWASM.groups.
func TestWASMBindingIsPrefixDeterministic(t *testing.T) {
	const name = "udf"
	// Three registrations of ONE name, all destined for group 0's log.
	//
	// THE CONTRACT-DIFFERING MARKER DIFFERS IN Kind, and it has to be Kind rather
	// than the key extractor. checkWASMGroupContract compares BOTH halves of the
	// contract, so either one reaches its refusal branch — but only Kind is a
	// field a registration is still free to vary. The key extractor is pinned to
	// one legal value cluster-wide (see validateWASMKeyExtractor), so a marker set
	// that differed there would be refused before the fold ever saw it and this
	// test would prove nothing.
	//
	//   b's OpReadOnly declaration over a module that DOES write state is legal
	//   here, and deliberately so: the OpReadOnly/writes-state guard moved to
	//   wasm.Runtime.resolveModuleForInvoke, which runs per INVOCATION, and this
	//   test only ever applies registrations. Nothing on the apply path judges the
	//   pairing (see wasm.RegisterModule's "THE GUARD IS NOT RUN HERE" note).
	//
	//   a — the OpReadWrite contract, oldest.
	//   b — a CONTRACT CHANGE (OpReadOnly), middling Epoch. This is the marker the
	//       permutation test cannot build and the whole reason this test exists.
	//   c — the OpReadWrite contract again, newest of the three.
	a := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readTestWASM(t, "../wasm/testdata/put.wasm")),
		ExportName: "apply", Epoch: 1,
	}
	b := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadOnly, Blob: ops.WASMBlobFingerprint(readIncrWASM(t)),
		ExportName: "apply", Epoch: 5,
	}
	c := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readDelWASM(t)),
		ExportName: "apply", Epoch: 9,
	}
	// Group 3's traffic: the highest Epoch in the test, so a node that hosts group
	// 3 ends with a node-wide install NO group-0-only node has. If the group-0
	// binding were contaminated by node-wide state, this is what would show it.
	d := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readIncrWASM(t)),
		ExportName: "apply", Epoch: 100,
	}
	groupThree := []wasmMarker{{3, a}, {3, d}}

	label := map[uint64]string{1: "a(RW,1)", 5: "b(RO,5)", 9: "c(RW,9)", 100: "d(RW,100)"}
	orderOf := func(ms []wasmMarker) string {
		s := ""
		for _, m := range ms {
			s += label[m.reg.Epoch] + " "
		}
		return s
	}

	// Every ordering of group 0's log, and the binding each one produces.
	bindings := make(map[string]string)

	permuteMarkers([]wasmMarker{{0, a}, {0, b}, {0, c}}, func(p []wasmMarker) bool {
		zero := append([]wasmMarker(nil), p...)
		order := orderOf(zero)

		placements := map[string][]wasmMarker{
			"solo":   zero,
			"before": append(append([]wasmMarker(nil), groupThree...), zero...),
			"after":  append(append([]wasmMarker(nil), zero...), groupThree...),
		}
		got := make(map[string]string, len(placements))
		for who, seq := range placements {
			r := newWASMReplica(t)
			applyMarkersTo(t, r, seq)
			got[who] = bindingIn(r, name, 0)
		}
		if got["solo"] != got["before"] || got["solo"] != got["after"] {
			t.Errorf("GROUP 0's BINDING DEPENDS ON SOMETHING OTHER THAN GROUP 0's LOG.\n  group 0's order  %s\n  solo             %s\n  +group 3 first   %s\n  +group 3 last    %s\nthese three replicas applied the SAME group-0 sequence and disagree, so two replicas of group 0 with different placements would execute one committed entry with different bytes — silently, with no error to classify",
				order, got["solo"], got["before"], got["after"])
			return false
		}
		if b, dup := bindings[order]; dup && b != got["solo"] {
			t.Errorf("the same group-0 order %s bound %s and then %s: the fold is not a function of the prefix at all", order, b, got["solo"])
			return false
		}
		bindings[order] = got["solo"]
		return true
	})
	if len(bindings) != 6 {
		t.Fatalf("checked %d orderings of a 3-marker log, want 6", len(bindings))
	}

	t.Run("the fold is order-DEPENDENT across contracts, and that is the documented property", func(t *testing.T) {
		distinct := make(map[string]struct{}, 2)
		for _, v := range bindings {
			distinct[v] = struct{}{}
		}
		if len(distinct) < 2 {
			t.Fatalf("every ordering bound the same version (%v): the marker set is no longer reaching checkWASMGroupContract's refusal branch, so the test above has silently degenerated into a permutation test and proves nothing about the shipped fold", bindings)
		}

		// The exact counterexample written down in installedWASM.groups: with a
		// contract-differing pair, a,b binds a (b is refused) and b,a binds b (b is
		// unchecked, then a loses the maximum).
		ab, ba := newWASMReplica(t), newWASMReplica(t)
		applyMarkersTo(t, ab, []wasmMarker{{0, a}, {0, b}})
		applyMarkersTo(t, ba, []wasmMarker{{0, b}, {0, a}})
		gotAB, gotBA := bindingIn(ab, name, 0), bindingIn(ba, name, 0)
		wantAB := wasmModuleIDOf(a).String()[:12]
		wantBA := wasmModuleIDOf(b).String()[:12]
		if gotAB != wantAB {
			t.Errorf("order a,b bound %s, want a (%s): the FIRST registration in a group's log must pin that group's contract, so the contract-changing b has to be refused", gotAB, wantAB)
		}
		if gotBA != wantBA {
			t.Errorf("order b,a bound %s, want b (%s): b is the group's first registration and is accepted unchecked, and a is not newer, so nothing may move the binding", gotBA, wantBA)
		}
	})
}

// TestWASMBindingFoldIgnoresUnattributedMarkers pins the ops.NoShardIndex rule.
//
// A registration with no group provenance — a snapshot's install section, or a
// handler invoked directly in a test — must bind NOTHING. Recording it against
// group 0 would be a false claim about a log that may never have carried it,
// which is exactly the over-broad route-gate evidence that lets a replica propose
// invocations into a group with no registration.
func TestWASMBindingFoldIgnoresUnattributedMarkers(t *testing.T) {
	var in installedWASM
	groups, changed := in.bindWith(ops.NoShardIndex, markerReg("udf", "v1", 1))
	if changed || len(groups) != 0 {
		t.Fatalf("an unattributed registration bound %v (changed=%v); it must bind nothing", sortedGroups(groups), changed)
	}
}

// TestWASMBindingFoldDoesNotMutateAPublishedTable pins the copy-on-write
// contract.
//
// The binding table is published to the WASM runtime and read WITHOUT A LOCK by
// every shard group's apply goroutine (see wasm.GroupBindings). A fold that
// mutated the map in place would be a data race against every concurrent
// invocation, and would also let a group's version change under an apply that had
// already resolved it.
func TestWASMBindingFoldDoesNotMutateAPublishedTable(t *testing.T) {
	v1 := markerReg("udf", "v1", 1)
	v2 := markerReg("udf", "v2", 2)

	in := installedWASM{groups: testBindings(v1, 0, 1)}
	published := in.groups
	before := published[0].id

	next, changed := in.bindWith(0, v2)
	if !changed {
		t.Fatal("precondition: v2 must supersede v1")
	}
	if &next == &published {
		t.Fatal("bindWith returned the same map")
	}
	if published[0].id != before {
		t.Error("the fold mutated the PUBLISHED table: every concurrently-running apply that had already read it would see the version change underneath it, and the read is unlocked, so it is also a data race")
	}
	if _, gained := published[2]; gained {
		t.Error("the fold added a group to the published table in place")
	}
	if next[0].id == before {
		t.Error("the fresh table did not take the new version")
	}
}
