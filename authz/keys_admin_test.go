// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// keysOps is the set of online key-admin coordinator virtual-op names. They MUST
// classify as admin so only an admin-scoped key passes the authorize gate.
var keysOps = []string{ops.OpKeysAdd, ops.OpKeysRevoke, ops.OpKeysList}

// TestKeysOpsClassifiedAdmin pins the three keys ops to ActionAdmin via actionFor
// — the single source of the admin gate. If any drops to read/write a low-priv
// key could mutate the key registry.
func TestKeysOpsClassifiedAdmin(t *testing.T) {
	opsReg := newOpsReg(t)
	for _, op := range keysOps {
		if got := actionFor(op, opsReg); got != ActionAdmin {
			t.Errorf("actionFor(%q)=%q, want %q (keys ops must be admin-gated)", op, got, ActionAdmin)
		}
	}
}

// TestKeysOpsAdminGate is the teeth test for the authorize gate: a NON-admin key
// (read-only and write-only) is DENIED all three keys ops, while an admin-scoped
// key (and a superuser) is ALLOWED. The keys ops target the empty/cluster
// resource, so only "admin:*"/"*:*" covers them — a "read:*"/"write:*" scope
// never does.
func TestKeysOpsAdminGate(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:*"}},
		vector.APIKey{Token: "writer", Tenant: "acme", Scopes: []string{"read:*", "write:*"}},
		vector.APIKey{Token: "admin", Tenant: "acme", Scopes: []string{"admin:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL")

	for _, op := range keysOps {
		// Non-admin keys denied.
		for _, tok := range []string{"reader", "writer"} {
			if auth(AuthRequest{Token: tok, Op: op}) {
				t.Errorf("non-admin key %q must be DENIED keys op %q", tok, op)
			}
		}
		// Admin/superuser allowed.
		for _, tok := range []string{"admin", "super"} {
			if !auth(AuthRequest{Token: tok, Op: op}) {
				t.Errorf("admin key %q must be ALLOWED keys op %q", tok, op)
			}
		}
		// Unknown / no token denied.
		if auth(AuthRequest{Token: "nope", Op: op}) {
			t.Errorf("unknown token must be DENIED keys op %q", op)
		}
		if auth(AuthRequest{Op: op}) {
			t.Errorf("no-principal request must be DENIED keys op %q", op)
		}
		// Internal service token is superuser.
		if !auth(AuthRequest{Token: "INTERNAL", Op: op}) {
			t.Errorf("internal token must be ALLOWED keys op %q", op)
		}
	}
}
