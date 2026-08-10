// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// __wasm_blob_put__ and __wasm_blob_get__ are the WASM blob transport
// (cluster/wasm_blob_transport.go): the channel by which a node that lacks a
// module's bytes obtains them. The put carries attacker-supplied module bytes
// and writes them into a node's code store; the get returns module bytes out of
// it. Both require admin.
//
// THESE TESTS EXIST BECAUSE THE CLASSIFICATION IS OTHERWISE UNPINNED, which is
// the same defect the __register_wasm_shard__ tests in register_wasm_authz_test.go
// were written for. Neither op is in the ops registry, so actionFor would fall
// through to the deny-by-default "admin" return and both would BE admin today
// with no adminOps entry at all — by coincidence, not by decision. Their sibling
// __register_wasm__ IS registered OpReadWrite, so registering these alongside it
// (an entirely natural refactor, and the obvious way to make them dispatchable
// through the registry rather than through n.adminOps) would silently demote them
// to "write" and hand code loading, and module exfiltration, to any write:* key.
//
// Every test below therefore registers the op OpReadWrite FIRST, so what is
// covered is the DANGEROUS configuration rather than the fallthrough.

func newOpsRegWithWASMBlobOps(t *testing.T, names ...string) *ops.Registry {
	t.Helper()
	r := ops.NewRegistry()
	if err := ops.RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	if err := ops.RegisterWASMRegisterOp(r, nil); err != nil {
		t.Fatalf("RegisterWASMRegisterOp: %v", err)
	}
	for _, name := range names {
		if err := r.Register(name, ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
			return nil, nil
		}); err != nil {
			t.Fatalf("register %s: %v", name, err)
		}
		if _, kind, _, ok := r.Lookup(name); !ok || kind != ops.OpReadWrite {
			t.Fatalf("precondition: %s must be registered OpReadWrite (ok=%v kind=%v)", name, ok, kind)
		}
	}
	return r
}

func TestActionForWASMBlobOpsIsAdmin(t *testing.T) {
	for _, op := range []string{"__wasm_blob_put__", "__wasm_blob_get__"} {
		// Fail-closed with no registry at all (the admin set is consulted first).
		if got := actionFor(op, nil); got != ActionAdmin {
			t.Errorf("actionFor(%s,nil)=%q want %q", op, got, ActionAdmin)
		}
		// And, the case that matters: registered OpReadWrite the way its sibling
		// __register_wasm__ is. Without the adminOps entry this is where the op
		// silently becomes "write".
		reg := newOpsRegWithWASMBlobOps(t, op)
		if got := actionFor(op, reg); got != ActionAdmin {
			t.Errorf("actionFor(%s) after OpReadWrite registration = %q want %q", op, got, ActionAdmin)
		}
	}
}

func TestRBACWASMBlobOpsRequireAdmin(t *testing.T) {
	// Both registered OpReadWrite so the end-to-end grant is exercised against
	// the dangerous configuration, not against the deny-by-default fallthrough.
	opsReg := newOpsRegWithWASMBlobOps(t, "__wasm_blob_put__", "__wasm_blob_get__")
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:*"}},
		vector.APIKey{Token: "writer", Tenant: "acme", Scopes: []string{"write:*"}},
		vector.APIKey{Token: "admin", Tenant: "acme", Scopes: []string{"admin:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL-SVC-TOKEN")

	cases := []struct {
		op    string
		token string
		want  bool
		desc  string
	}{
		{"__wasm_blob_put__", "reader", false, "read:* may not plant module bytes on a node"},
		{"__wasm_blob_put__", "writer", false, "cluster-wide write:* is NOT enough to plant module bytes on a node"},
		{"__wasm_blob_put__", "admin", true, "admin:* may"},
		{"__wasm_blob_put__", "super", true, "*:* may"},
		// The get is the deliberate, arguable half: it returns bytes every cluster
		// member already holds, and it is STILL admin. A read:* key is not a member.
		{"__wasm_blob_get__", "reader", false, "cluster-wide read:* may NOT read module bytes out of the blob store"},
		{"__wasm_blob_get__", "writer", false, "cluster-wide write:* may NOT read module bytes out of the blob store"},
		{"__wasm_blob_get__", "admin", true, "admin:* may"},
		{"__wasm_blob_get__", "super", true, "*:* may"},
	}
	for _, c := range cases {
		if got := auth(AuthRequest{Token: c.token, Op: c.op, Args: nil}); got != c.want {
			t.Errorf("authorize(%s, %s)=%v want %v — %s", c.token, c.op, got, c.want, c.desc)
		}
	}

	// The only legitimate caller is a peer, which carries the internal service
	// token and is granted before the adminOps map is ever consulted. Admin
	// therefore costs the push path nothing.
	for _, op := range []string{"__wasm_blob_put__", "__wasm_blob_get__"} {
		if !auth(AuthRequest{Token: "INTERNAL-SVC-TOKEN", Op: op}) {
			t.Errorf("the internal service principal must be able to call %s", op)
		}
	}
}
