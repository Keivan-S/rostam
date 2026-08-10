// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// __register_wasm__ loads an attacker-supplied WASM module that then runs on
// every replica's FSM Apply path with host functions that can address ANY key.
// It is registered OpReadWrite in the ops registry, but its true privilege is
// admin (code loading, strictly more dangerous than a collection write). These
// tests pin that classification and the end-to-end grant so a broad-but-non-admin
// write:* scope can never register a module.

func newOpsRegWithWASM(t *testing.T) *ops.Registry {
	t.Helper()
	r := ops.NewRegistry()
	if err := ops.RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	// Register __register_wasm__ exactly as production does: OpReadWrite. Without
	// the adminOps entry, actionFor would fall through to the OpReadWrite branch
	// and classify it "write" — the defect this test guards against.
	if err := ops.RegisterWASMRegisterOp(r, nil); err != nil {
		t.Fatalf("RegisterWASMRegisterOp: %v", err)
	}
	return r
}

func TestActionForRegisterWASMIsAdmin(t *testing.T) {
	reg := newOpsRegWithWASM(t)
	// It IS present in the registry as OpReadWrite...
	if _, kind, _, ok := reg.Lookup("__register_wasm__"); !ok || kind != ops.OpReadWrite {
		t.Fatalf("precondition: __register_wasm__ must be registered OpReadWrite (ok=%v kind=%v)", ok, kind)
	}
	// ...yet actionFor must classify it admin, overriding the OpReadWrite fallthrough.
	if got := actionFor("__register_wasm__", reg); got != ActionAdmin {
		t.Fatalf("actionFor(__register_wasm__)=%q want %q", got, ActionAdmin)
	}
	// Fail-closed even with a nil registry (admin set is consulted first).
	if got := actionFor("__register_wasm__", nil); got != ActionAdmin {
		t.Fatalf("actionFor(__register_wasm__,nil)=%q want %q", got, ActionAdmin)
	}
}

func TestRBACRegisterWASMRequiresAdmin(t *testing.T) {
	opsReg := newOpsRegWithWASM(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "writer", Tenant: "acme", Scopes: []string{"write:*"}},
		vector.APIKey{Token: "admin", Tenant: "acme", Scopes: []string{"admin:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL-SVC-TOKEN")

	cases := []struct {
		token string
		want  bool
		desc  string
	}{
		{"writer", false, "cluster-wide write:* is NOT enough to register a WASM module"},
		{"admin", true, "admin:* may register a WASM module"},
		{"super", true, "*:* may register a WASM module"},
	}
	for _, c := range cases {
		got := auth(AuthRequest{Token: c.token, Op: "__register_wasm__", Args: nil})
		if got != c.want {
			t.Errorf("authorize(%s, __register_wasm__)=%v want %v — %s", c.token, got, c.want, c.desc)
		}
	}
}

// __register_wasm_shard__ is the INTERNAL wrapper that carries one group's leg
// of the registration broadcast to a peer (cluster/wasm_broadcast.go). Its
// payload is an attacker-supplied WASM module, exactly like __register_wasm__'s,
// so it must require the same admin privilege.
//
// These tests exist because it is admin today only by ACCIDENT: it is absent
// from the ops registry, so actionFor falls through to the deny-by-default
// "admin" return. Its sibling __register_wasm__ IS registered OpReadWrite, so
// registering the wrapper too — an entirely natural refactor — would flip it to
// "write" and let any write:* key load code onto every replica. The explicit
// adminOps entry is what these pin.

func TestActionForRegisterWASMShardIsAdmin(t *testing.T) {
	reg := newOpsRegWithWASM(t)
	if got := actionFor("__register_wasm_shard__", reg); got != ActionAdmin {
		t.Fatalf("actionFor(__register_wasm_shard__)=%q want %q", got, ActionAdmin)
	}
	if got := actionFor("__register_wasm_shard__", nil); got != ActionAdmin {
		t.Fatalf("actionFor(__register_wasm_shard__,nil)=%q want %q", got, ActionAdmin)
	}
	// The classification must NOT depend on the op being missing from the
	// registry. Register it OpReadWrite — the way its sibling is registered — and
	// it must still be admin. Without the adminOps entry this is the case that
	// silently downgrades it to "write".
	if err := reg.Register("__register_wasm_shard__", ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("register __register_wasm_shard__: %v", err)
	}
	if _, kind, _, ok := reg.Lookup("__register_wasm_shard__"); !ok || kind != ops.OpReadWrite {
		t.Fatalf("precondition: expected OpReadWrite registration (ok=%v kind=%v)", ok, kind)
	}
	if got := actionFor("__register_wasm_shard__", reg); got != ActionAdmin {
		t.Fatalf("actionFor(__register_wasm_shard__) after OpReadWrite registration = %q want %q", got, ActionAdmin)
	}
}

func TestRBACRegisterWASMShardRequiresAdmin(t *testing.T) {
	opsReg := newOpsRegWithWASM(t)
	// Registered OpReadWrite so the test covers the dangerous configuration, not
	// just the deny-by-default fallthrough.
	if err := opsReg.Register("__register_wasm_shard__", ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("register __register_wasm_shard__: %v", err)
	}
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "writer", Tenant: "acme", Scopes: []string{"write:*"}},
		vector.APIKey{Token: "admin", Tenant: "acme", Scopes: []string{"admin:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL-SVC-TOKEN")

	cases := []struct {
		token string
		want  bool
		desc  string
	}{
		{"writer", false, "cluster-wide write:* is NOT enough to drive a WASM registration into a shard group"},
		{"admin", true, "admin:* may"},
		{"super", true, "*:* may"},
	}
	for _, c := range cases {
		got := auth(AuthRequest{Token: c.token, Op: "__register_wasm_shard__", Args: nil})
		if got != c.want {
			t.Errorf("authorize(%s, __register_wasm_shard__)=%v want %v — %s", c.token, got, c.want, c.desc)
		}
	}
}
