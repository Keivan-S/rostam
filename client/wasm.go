// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"

	"github.com/rostamlabs/rostam/ops"
)

// RegisterWASM ships a compiled WASM module via __register_wasm__. The Client
// routes the call to a node that broadcasts the registration into EVERY shard
// group's Raft log and returns once each group has committed it.
//
// UPDATING A LIVE MODULE IS SUPPORTED; CHANGING ITS CONTRACT IS NOT. A second
// registration under a live name may change the module — the bytes, the export
// symbol, the fuel cap — and should carry a higher Epoch. It may NOT change the
// op's Kind; that is refused at propose time with
// cluster.ErrWASMUpdateUnsupported (HTTP 400), and a registration needing either
// belongs under a NEW op name.
//
// The asymmetry is structural, not a policy choice. The module a committed entry
// executes is resolved from the SHARD GROUP whose log carried it, so every
// replica of that group derives the same version for the same entry however far
// along its other groups it is — which is what makes an in-place module update
// safe. Kind cannot be resolved that way: it is read before any group is known
// (it decides whether the invocation is replicated at all), so it is frozen at
// first registration. The key extractor is the other field that cannot be per
// group — it is what COMPUTES the group index — and it is not frozen so much as
// CONSTANT: every WASM op uses the same one, so there is no field to change. See
// ops.WASMKeyExtractorHandle.
//
// The propose-time refusal is BEST-EFFORT — it is judged against the receiving
// node's own install state, so a node that has not applied the original
// registration cannot recognise a contract change and will accept it. It is
// backed by an apply-time refusal that is not best-effort: each shard group
// refuses a contract change against its own binding, identically on every replica
// of that group.
//
// AN UPDATE ROLLS OUT PER GROUP, NOT ATOMICALLY. Each shard group switches to the
// new module when its own log commits the registration, so for a short window
// different groups run different versions. Every replica of a given group agrees
// throughout; what an update does not give you is a single instant at which the
// whole cluster changes over.
//
// SUCCESS DOES NOT MEAN THE OP IS IMMEDIATELY INVOCABLE EVERYWHERE. A node will
// not propose an invocation into a shard group it hosts until it has itself
// applied the registration from that group's log (cluster's route gate — the
// invariant that keeps a replica from ever meeting an invocation for an op it
// cannot look up). A return from here means every group COMMITTED the entry; the
// remaining nodes apply it moments later, each opening the gate for its own
// groups as it does. Until then an invocation can come back with
// cluster.ErrWASMOpNotInThisGroup ("op not registered in this shard group yet",
// HTTP 503) or, on a node that has applied nothing yet, cluster.ErrUnknownOp
// (HTTP 404).
//
// Both are transient and both are safe to retry: registration is idempotent, so
// re-running RegisterWASM with the SAME r is also safe and is explicitly allowed
// by the no-update rule above. A caller that needs the op live before it proceeds
// should retry the first invocation briefly rather than assume it.
//
// An error return does NOT mean nothing happened: the broadcast attempts every
// group even after one fails, and the groups that accepted keep the entry. The
// op is then usable for keys routing to those groups and errors for the rest,
// until a retry lands it everywhere.
//
// pushReport IS PART OF THE RESULT, NOT DIAGNOSTICS. Before the registration
// enters any log, the receiving node pushes the module's BYTES to every member it
// can reach and requires a compile verdict from each one that answers; a member
// that refuses fails this call. pushReport is empty when every member acked. When
// it is not, it names the members that rendered no verdict — unreachable, or on a
// build that does not know the push op — and those are exactly the members that
// do not hold the bytes. THE MARKER NO LONGER CARRIES THE MODULE, so a member named
// here has to FETCH it before it can execute the op, and an invocation that
// arrives first BLOCKS that shard group until it does — which also stops the
// group snapshotting and compacting its Raft log. A caller that ignores this is
// choosing not to know which nodes are one step from a stalled shard group.
//
// The push is also subject to a DURABILITY FLOOR: a majority of cluster members
// must hold the module before the marker is proposed at all, so a call that
// SUCCEEDS guarantees a reachable source exists for every later fetch. A cluster
// too degraded to reach that majority fails this call rather than accepting a
// registration that could become permanently unrunnable. See cluster.pushWASMBlob.
// THE MODULE IS A SEPARATE ARGUMENT NOW, and that is the visible face of thin
// registration markers. r describes the OP — its name, contract, epoch — and
// module is the WASM binary. Only the module's 32-byte content address travels in
// the replicated marker; the binary itself goes to the receiving node once, is
// pushed from there to a majority of cluster members before anything is proposed,
// and is fetched on demand by any member the push missed. r.Blob is derived from
// module by the encoder, so the two can never disagree.
func (c *Client) RegisterWASM(ctx context.Context, r ops.WASMRegistration, module []byte) (pushReport string, err error) {
	reply, err := c.Call(ctx, "__register_wasm__", ops.EncodeWASMRegistrationRequest(r, module))
	if err != nil {
		return "", err
	}
	return string(reply), nil
}
