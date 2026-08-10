// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// RegisterWASM broadcasts the registration into every shard group's Raft log.
//
// UPDATING A LIVE MODULE IS SUPPORTED: a second registration under a live name
// may change the module (send it with a higher Epoch), and each shard group
// switches to it when its own log commits the registration, so every replica of a
// group agrees on what executes there throughout. CHANGING THE OP'S KIND is not,
// because it is read before any shard group is known; that needs a NEW op name.
// (There is no key extractor to change: every WASM op uses the same one — see
// ops.WASMKeyExtractorHandle.) See Store.RegisterWASM and
// cluster.checkWASMUpdateGate.
// The returned pushReport names the members that did not take the module's bytes
// during the pre-registration push; it is empty when every member acked. See
// Store.RegisterWASM.
func (e *embedded) RegisterWASM(_ context.Context, r WASMRegistration, module []byte) (string, error) {
	reply, err := e.node.Call("__register_wasm__", ops.EncodeWASMRegistrationRequest(r, module))
	if err != nil {
		return "", err
	}
	return string(reply), nil
}

// RegisterWASM forwards the registration to a node, which broadcasts it. The
// frozen-contract rule on embedded.RegisterWASM applies identically — it is
// enforced by the node that receives the call, not by this client.
//
// It calls Client.Call rather than Client.RegisterWASM, so the push report has to
// be surfaced HERE too: routing this through the client method would not do it
// for free, and discarding the reply on this one path would leave the facade most
// callers actually use blind to a partial push. See Store.RegisterWASM.
func (n *networkedStore) RegisterWASM(ctx context.Context, r WASMRegistration, module []byte) (string, error) {
	reply, err := n.c.Call(ctx, "__register_wasm__", ops.EncodeWASMRegistrationRequest(r, module))
	if err != nil {
		return "", err
	}
	return string(reply), nil
}

// RegisterWASM registers the WASM module locally only. Direct mode has no
// Raft so modules are not replicated to other nodes.
//
// Updating a live module is not supported here either, but for an unrelated
// reason and with a different failure: this path uses wasm.RegisterModule (not
// the replacing variant), so a second registration under a live name is rejected
// by the ops registry with ops.ErrDuplicateOp. The refusal is now clean — the
// runtime is content addressed, so AddModule adds the new module alongside the
// old one instead of rebinding the name to it, and the registry entry (which the
// rejected call never touched) still runs the original bytes. Register under a
// NEW name.
// The pushReport is always empty here: direct mode has no peers, so there is
// nothing to push and nothing that could be missing the bytes.
func (d *directStore) RegisterWASM(_ context.Context, r WASMRegistration, module []byte) (string, error) {
	// Take the all-shards barrier so module compile/registration is mutually
	// exclusive with every read-write op (including running WASM ops) and with
	// concurrent RegisterWASM calls — the same exclusion the old global lock gave.
	d.lockAllShards()
	defer d.unlockAllShards()
	if d.wasmRT == nil {
		rt, err := wasm.NewRuntime()
		if err != nil {
			return "", err
		}
		d.wasmRT = rt
	}
	m, err := wasm.Compile(module)
	if err != nil {
		return "", err
	}
	// The OpReadOnly/writes-state guard, run EXPLICITLY on the compiled module.
	// wasm.RegisterModule no longer runs it: on the replicated path that refusal
	// would depend on whether the node had fetched the bytes yet, which is not a
	// function of the log, so it moved to the resolver. Direct mode has no
	// replicas and always holds the bytes, so refusing HERE is both safe and the
	// better error — the caller learns at registration rather than at the first
	// invocation.
	if err := wasm.ValidateModuleKind(r.Name, r.Kind, m); err != nil {
		_ = m.Close()
		return "", err
	}
	id, err := d.wasmRT.AddModule(m, r.ExportName, r.MaxFuel)
	if err != nil {
		_ = m.Close()
		return "", err
	}
	return "", wasm.RegisterModule(d.registry, d.wasmRT, r.Name, id, r.Kind, ops.WASMKeyExtractor())
}
