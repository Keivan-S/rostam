// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
)

// TestGetArgsOptsByteIdenticalWhenZero proves the rc opts trailer on the shared
// get codec (dense / named / MV all use EncodeVectorGetArgs) is BYTE-IDENTICAL to
// the legacy encoder when rc==0 && opa==0 — the AnyReplica default path is
// wire-unchanged (zero added cost). A non-zero rc appends the self-delimiting
// [marker][rc][opa] trailer.
func TestGetArgsOptsByteIdenticalWhenZero(t *testing.T) {
	const col, id, flags = "docs", uint64(42), uint8(0x3)
	legacy := EncodeVectorGetArgs(col, id, flags)

	// rc==0 && opa==0 ⇒ byte-identical.
	zero := EncodeVectorGetArgsOpts(col, id, flags, ConsistencyAnyReplica, 0, 0)
	if !bytes.Equal(legacy, zero) {
		t.Fatalf("rc==0 get args not byte-identical to legacy:\n legacy=%v\n opts  =%v", legacy, zero)
	}

	// A non-zero rc MUST extend the wire (the trailer rides).
	lin := EncodeVectorGetArgsOpts(col, id, flags, ConsistencyLinearizable, 1, 0)
	if len(lin) != len(legacy)+3 {
		t.Fatalf("linearizable get args length = %d, want legacy+3 (%d)", len(lin), len(legacy)+3)
	}
	if !bytes.HasPrefix(lin, legacy) {
		t.Fatalf("linearizable get args do not extend the legacy base block")
	}
}

// TestGetArgsOptsRoundTrip proves DecodeVectorGetArgsOpts recovers rc/opa and
// that the legacy DecodeVectorGetArgs tolerates the trailer (ignores it).
func TestGetArgsOptsRoundTrip(t *testing.T) {
	const col, id, flags = "c", uint64(7), uint8(0x1)
	args := EncodeVectorGetArgsOpts(col, id, flags, ConsistencyLinearizable, 1, 0)

	gotCol, gotID, gotFlags, rc, opa, _, err := DecodeVectorGetArgsOpts(args)
	if err != nil {
		t.Fatalf("decode opts: %v", err)
	}
	if gotCol != col || gotID != id || gotFlags != flags || rc != ConsistencyLinearizable || opa != 1 {
		t.Fatalf("round-trip mismatch: col=%q id=%d flags=%d rc=%d opa=%d", gotCol, gotID, gotFlags, rc, opa)
	}

	// Legacy decoder ignores the trailer (backward compatible).
	bcCol, bcID, bcFlags, err := DecodeVectorGetArgs(args)
	if err != nil {
		t.Fatalf("legacy decode of rc-carrying args: %v", err)
	}
	if bcCol != col || bcID != id || bcFlags != flags {
		t.Fatalf("legacy decode mismatch: col=%q id=%d flags=%d", bcCol, bcID, bcFlags)
	}

	// Legacy args (no trailer) decode with rc=0,opa=0.
	legacy := EncodeVectorGetArgs(col, id, flags)
	_, _, _, lrc, lopa, _, err := DecodeVectorGetArgsOpts(legacy)
	if err != nil {
		t.Fatalf("opts decode of legacy args: %v", err)
	}
	if lrc != 0 || lopa != 0 {
		t.Fatalf("legacy args decoded rc=%d opa=%d, want 0/0", lrc, lopa)
	}
}

// TestGetConfigArgsOptsByteIdenticalWhenZero proves the get_config codecs (dense /
// named-name / MV) carry the rc trailer byte-identically when zero and recover it
// when set.
func TestGetConfigArgsOptsByteIdenticalWhenZero(t *testing.T) {
	const name = "coll"
	cases := []struct {
		family       string
		legacy       func() []byte
		opts         func(rc, opa uint8) []byte
		decodeOpts   func([]byte) (string, uint8, uint8, uint64, error)
		legacyDecode func([]byte) (string, error)
	}{
		{
			"dense_get_config",
			func() []byte { return EncodeGetConfigArgs(name) },
			func(rc, opa uint8) []byte { return EncodeGetConfigArgsOpts(name, rc, opa, 0) },
			DecodeGetConfigArgsOpts,
			DecodeGetConfigArgs,
		},
		{
			"named_name",
			func() []byte { return EncodeNamedNameArgs(name) },
			func(rc, opa uint8) []byte { return EncodeNamedNameArgsOpts(name, rc, opa, 0) },
			DecodeNamedNameArgsOpts,
			DecodeNamedNameArgs,
		},
		{
			"mv_get_config",
			func() []byte { return EncodeMVGetConfigArgs(name) },
			func(rc, opa uint8) []byte { return EncodeMVGetConfigArgsOpts(name, rc, opa, 0) },
			DecodeMVGetConfigArgsOpts,
			DecodeMVGetConfigArgs,
		},
	}
	for _, tc := range cases {
		t.Run(tc.family, func(t *testing.T) {
			legacy := tc.legacy()
			if zero := tc.opts(0, 0); !bytes.Equal(legacy, zero) {
				t.Fatalf("%s rc==0 not byte-identical:\n legacy=%v\n opts  =%v", tc.family, legacy, zero)
			}
			lin := tc.opts(ConsistencyLinearizable, 1)
			gotName, rc, opa, _, err := tc.decodeOpts(lin)
			if err != nil {
				t.Fatalf("%s decode opts: %v", tc.family, err)
			}
			if gotName != name || rc != ConsistencyLinearizable || opa != 1 {
				t.Fatalf("%s round-trip mismatch: name=%q rc=%d opa=%d", tc.family, gotName, rc, opa)
			}
			// legacy decoder tolerates the trailer
			if bcName, err := tc.legacyDecode(lin); err != nil || bcName != name {
				t.Fatalf("%s legacy decode of rc args: name=%q err=%v", tc.family, bcName, err)
			}
		})
	}
}

// TestReadConsistencyOfGetFamily is the ANTI-SILENT-DROP guard: ReadConsistencyOf
// MUST report the Linearizable rc for every get / get_config op, because that is
// what arms the shard readIndex barrier in shard.Store.Call. If any of these ops
// were removed from ReadConsistencyOf (or the rc dropped from the encoder), a
// Linearizable get would silently serve stale — this test goes RED then.
func TestReadConsistencyOfGetFamily(t *testing.T) {
	getArgs := func(rc uint8) []byte { return EncodeVectorGetArgsOpts("c", 1, 0, rc, 0, 0) }
	cfgArgs := EncodeGetConfigArgsOpts
	nameArgs := EncodeNamedNameArgsOpts
	mvCfgArgs := EncodeMVGetConfigArgsOpts

	cases := []struct {
		op   string
		args []byte
	}{
		{"vector_get", getArgs(ConsistencyLinearizable)},
		{"vector_named_get", getArgs(ConsistencyLinearizable)},
		{"vector_mv_get", getArgs(ConsistencyLinearizable)},
		{"vector_get_config", cfgArgs("c", ConsistencyLinearizable, 0, 0)},
		{"vector_named_get_config", nameArgs("c", ConsistencyLinearizable, 0, 0)},
		{"vector_mv_get_config", mvCfgArgs("c", ConsistencyLinearizable, 0, 0)},
	}
	for _, tc := range cases {
		t.Run(tc.op, func(t *testing.T) {
			rc, ok := ReadConsistencyOf(tc.op, tc.args)
			if !ok {
				t.Fatalf("ReadConsistencyOf(%q) ok=false — op missing from the consistency table (silent-stale hole)", tc.op)
			}
			if rc != ConsistencyLinearizable {
				t.Fatalf("ReadConsistencyOf(%q) rc=%d, want Linearizable (%d)", tc.op, rc, ConsistencyLinearizable)
			}
		})
	}
}

// TestReadConsistencyOfGetAnyReplica proves an AnyReplica (rc==0) get still
// returns ok=true with rc=0 — the op IS in the table, it just does not arm the
// barrier (no stale-serve risk, zero added cost).
func TestReadConsistencyOfGetAnyReplica(t *testing.T) {
	legacy := EncodeVectorGetArgs("c", 1, 0) // legacy, no trailer
	rc, ok := ReadConsistencyOf("vector_get", legacy)
	if !ok || rc != 0 {
		t.Fatalf("ReadConsistencyOf(vector_get, legacy) = (%d,%v), want (0,true)", rc, ok)
	}
}
