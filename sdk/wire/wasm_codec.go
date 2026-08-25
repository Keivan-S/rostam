// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
)

// WASMRegistration is the THIN MARKER that carries a WASM op registration
// through every shard group's Raft log. It is serialised via
// EncodeWASMRegistration / DecodeWASMRegistration and travels as the args of the
// "__register_wasm__" built-in op.
//
// ################ IT DOES NOT CARRY THE MODULE BYTES ANY MORE ################
//
// It names them (Blob) and nothing more. That is the whole marker. A
// registration is replicated into EVERY shard group's log (see
// cluster/wasm_broadcast.go), so an inline module cost NumShards × its size in
// Raft log on every node, and the same again in each group's snapshot once it
// compacted — 64 × 4 MiB of replicated log for one client-visible registration at
// the default shard count and the module cap. Replacing the module with its
// 32-byte content address makes a marker a few dozen bytes whatever the module
// weighs.
//
// THE BYTES TRAVEL OUT OF BAND, and the two halves of that are not symmetric:
//
//   - PUSH, before the marker is proposed: the receiving node stores the module
//     locally and delivers it to the cluster with __wasm_blob_put__, requiring a
//     compile verdict from each member that answers and a DURABILITY FLOOR — a
//     majority of cluster members must hold it — before anything enters a log.
//     See cluster.pushWASMBlob;
//   - FETCH, on demand: a node that applies a marker (or restores one from a
//     snapshot) and does not hold the bytes gets them with __wasm_blob_get__ from
//     a group peer or any cluster member. See cluster.wasmBlobFetcher.
//
// WHAT THAT COSTS, stated plainly because it is the price of the whole stage:
// applying a marker NO LONGER IMPLIES HOLDING THE MODULE. Before this change,
// "the peers of my group applied this same entry" was a proof that they had the
// bytes, and it is not any more. A committed invocation that resolves to a
// version this node has not fetched yet is therefore a real, reachable state, and
// it is neither an error nor a divergence — it is ErrWASMModuleNotResident, which
// shard.classifyApplyErr maps to classRetry: the apply mutates nothing, advances
// nothing, halts nothing, and re-runs when the blob lands. The durability floor
// is what guarantees a source exists for it to land FROM.
type WASMRegistration struct {
	// Name is the op name under which the module is registered (e.g. "my_udf").
	Name string
	// Kind controls whether the op bypasses Raft (OpReadOnly) or goes through
	// it (OpReadWrite).
	Kind OpKind
	// Blob is the CONTENT ADDRESS of the module bytes: WASMBlobFingerprint over
	// the module and nothing else, which is also the basename of the
	// content-addressed blob file (<dataDir>/wasm/blobs/<hex>.wasm).
	//
	// It replaced the inline Bytes field. It is a HASH OF THE BYTES ALONE rather
	// than of the registration, so it remains the self-verifying name of a file
	// whose contents are the bytes alone: a blob is valid iff sha256(contents)
	// equals its own name, and no sidecar has to be trusted to check it. See
	// WASMBlobFingerprint for why the registration fingerprint could not serve.
	//
	// A zero Blob is not a valid registration and is refused at both propose and
	// apply time — it names no module and nothing could ever resolve it.
	Blob [sha256.Size]byte
	// ExportName is the WASM export symbol that implements the handler.
	ExportName string
	// THERE IS NO KeyExtractorHandle FIELD, and its absence is the fix for a
	// silent cross-replica divergence rather than a simplification. See
	// WASMKeyExtractorHandle.
	//
	// MaxFuel caps the WASM instruction budget (0 = use the default cap).
	MaxFuel uint64
	// Epoch is the registration's version counter. It exists to make two racing
	// FIRST-TIME registrations of the same op name CONVERGE across replicas.
	//
	// IT DOES NOT MAKE UPDATING A LIVE MODULE SAFE, and the cluster refuses to try
	// (cluster.ErrWASMUpdateUnsupported): a second registration of a name that is
	// already registered is rejected at propose time unless it is byte-identical
	// to the installed one, Epoch included. Register the new module under a NEW
	// op name instead.
	//
	// WHY. What Epoch orders is the INSTALL, which is per-node; what would have to
	// be ordered is the EXECUTION of an entry, which is per-group. The effective
	// module version is node-wide (one ops.Registry entry per op NAME, whose
	// handler is bound to one module version — the wasm.Runtime itself is content
	// addressed and holds them all), while registrations arrive through per-group Raft logs that
	// commit at independent times — so a node that has not applied the update yet
	// holds complete, correct local state, proposes an invocation, and a peer that
	// HAS applied it executes that entry with different bytes. Both applies
	// succeed and the replicas silently diverge. No epoch discipline reaches that;
	// binding the version to the group's log prefix would (see the deferred design
	// note in cluster/wasm_gate.go).
	//
	// The field is kept because the convergence property below is real and worth
	// having for the case it does cover.
	//
	// A registration is replicated into EVERY shard Raft group (see
	// cluster/wasm_broadcast.go), and those groups are INDEPENDENTLY ordered. Two
	// registrations of the same Name carrying different bytes can therefore be
	// applied in one order by group 3's log and the opposite order by group 7's,
	// so a "last writer wins by arrival" rule leaves different replicas executing
	// DIFFERENT WASM for the same op name — silent divergence.
	//
	// The install rule is instead ORDER-INDEPENDENT: a registration replaces the
	// installed one only if its (Epoch, fingerprint) pair is strictly greater
	// lexicographically, with the content fingerprint (see
	// WASMRegistrationFingerprint) as the deterministic tiebreak for two
	// registrations that share an Epoch. Every replica therefore ends on the same
	// maximum regardless of arrival order — which is what stops a race between two
	// first-time registrations from leaving replicas on different modules.
	Epoch uint64
}

// WASMRegistrationFingerprint is the deterministic content hash used as the
// Epoch tiebreak and as the identity of an installed registration.
//
// It covers EVERY field that affects observable behaviour — name, kind, the
// module's content address, export symbol and fuel cap — but deliberately NOT
// Epoch, which is the ordering key rather than content. Covering Kind matters: it
// decides whether the op bypasses Raft, so two registrations differing only there
// are genuinely different registrations and must not compare equal. (The other
// routing field is gone: every WASM op uses one key extractor, so there is
// nothing left for a fingerprint to distinguish. See WASMKeyExtractorHandle.)
// (wasm.Runtime's own module fingerprint covers only bytes+export+fuel, so it
// cannot serve this purpose.)
func WASMRegistrationFingerprint(r WASMRegistration) [sha256.Size]byte {
	r.Epoch = 0
	return sha256.Sum256(EncodeWASMRegistration(r))
}

// WASMBlobFingerprint is the content address of a module's BYTES: plain
// sha256 over Bytes and nothing else. It is the filename (hex) of the
// content-addressed blob the cluster package writes under
// <dataDir>/wasm/blobs/, and it is deliberately NOT
// WASMRegistrationFingerprint.
//
// WHY THE BYTES ALONE. A blob file contains the module bytes and nothing else,
// and the invariant that makes it worth having is that the file is
// SELF-VERIFYING: a blob is valid iff sha256(its contents) equals its own name.
// Only a hash of the bytes can satisfy that. Addressing the file by
// WASMRegistrationFingerprint would name it after data the file does not contain
// (Name, Kind, ExportName, MaxFuel), so a reader could not
// check the name against the contents at all — it could only re-derive the name
// from a sidecar it has already chosen to trust, which is exactly the trust the
// self-verification is there to remove.
//
// The second reason is that those extra fields are properties of the OP, not of
// the bytes. Two ops that legitimately share a module — the same UDF registered
// under two names, say — differ in Name, so a registration-keyed layout would
// store the identical bytes twice under two names and the deduplication content
// addressing exists for would not happen.
//
// The two hashes therefore answer different questions and both are needed:
// WASMRegistrationFingerprint identifies a REGISTRATION (it is the convergence
// tiebreak and the update gate's identity, and must cover the routing fields);
// this identifies STORED BYTES. A third value, wasm.ModuleID, identifies an
// instantiated runtime slot and covers bytes+export+fuel because those three are
// compiled into it.
func WASMBlobFingerprint(b []byte) [sha256.Size]byte {
	return sha256.Sum256(b)
}

// WASMBlobHex renders a blob fingerprint in the ONE form every consumer of it
// uses: lower-case hex. That form is the blob file's basename, the sidecar's
// blob reference, and the wire payload of __wasm_blob_put__ / __wasm_blob_get__,
// and cluster.wasmBlobPath refuses anything that is not exactly it.
//
// It is a function rather than a method on WASMRegistration because the same
// rendering is needed for fingerprints that have no registration behind them —
// a blob transported on its own is exactly that case.
func WASMBlobHex(fp [sha256.Size]byte) string { return hex.EncodeToString(fp[:]) }

// ZeroWASMBlob is the fingerprint no module can have: a registration carrying it
// names nothing, so it is refused at propose time and at apply time rather than
// installed as an op whose bytes can never arrive.
//
// It is not merely "unlikely" — it is the value a zero-valued WASMRegistration
// has, which is what a caller who forgot to attach a module produces, and what a
// truncated-then-zero-padded frame would produce. Naming it makes both refusals
// read as the same check.
var ZeroWASMBlob [sha256.Size]byte

// EncodeWASMRegistrationRequest builds the CLIENT-EDGE payload of
// "__register_wasm__": the thin marker plus the module bytes it names.
//
//	[reg_len u32][EncodeWASMRegistration][module bytes]
//
// ############ THIS IS NOT WHAT ENTERS A RAFT LOG. THAT IS THE POINT. #########
//
// Two payloads are deliberately distinguished, and conflating them is the easiest
// way to undo this entire stage:
//
//   - THIS one is what a client sends to ONE node, once. It carries the module,
//     because that node is the one that has to store it and push it to the
//     cluster (cluster.pushWASMBlob) before anything is proposed;
//   - the MARKER (EncodeWASMRegistration alone) is what that node then broadcasts
//     into every shard group's log, what a peer's shard-scoped leg forwards, what
//     a snapshot carries, and what every replica applies. It carries no module.
//
// r.Blob IS SET FROM module HERE rather than trusted from the caller, so the two
// halves of the request cannot disagree by construction. A caller that computed a
// fingerprint itself gets the same value; a caller that left it zero gets a
// correct one instead of a marker that names nothing.
func EncodeWASMRegistrationRequest(r WASMRegistration, module []byte) []byte {
	r.Blob = WASMBlobFingerprint(module)
	enc := EncodeWASMRegistration(r)
	out := make([]byte, 4+len(enc)+len(module))
	binary.BigEndian.PutUint32(out[:4], uint32(len(enc))) //nolint:gosec // bounded by the marker size (a few dozen bytes plus three bounded strings)
	copy(out[4:], enc)
	copy(out[4+len(enc):], module)
	return out
}

// DecodeWASMRegistrationRequest parses what EncodeWASMRegistrationRequest built
// and PROVES THE TWO HALVES AGREE: the module must hash to the fingerprint the
// marker claims.
//
// That check is not decoration. Everything downstream — the push, every peer's
// blob store, every group's marker, every later fetch — addresses the module by
// r.Blob, so a frame whose marker names one module and whose body is another
// would push bytes nobody will ever ask for while broadcasting a marker nothing
// can resolve: a registration that is accepted, replicated, and permanently
// unrunnable on every node. Checking it here, on the one leg where both halves
// exist together, is the only place it CAN be checked.
//
// The returned module aliases b. Callers that retain it past the request must
// copy — cluster.pushWASMBlob writes it to disk and pushes it synchronously, so
// it does not.
func DecodeWASMRegistrationRequest(b []byte) (WASMRegistration, []byte, error) {
	var zero WASMRegistration
	if len(b) < 4 {
		return zero, nil, errors.New("ops: wasm registration request too short for reg_len")
	}
	regLen := int(binary.BigEndian.Uint32(b[:4]))
	// The bound is written as a SUBTRACTION on the length, not `4+regLen > len(b)`.
	// That addition overflows: regLen just below MaxInt32 makes 4+regLen wrap
	// negative on a 32-bit int, the comparison is then false, and the slice below
	// panics -- which the regLen < 0 check alone does NOT catch, because regLen
	// itself is positive here.
	if regLen < 0 || regLen > len(b)-4 {
		return zero, nil, fmt.Errorf("ops: wasm registration request declares a %d-byte marker but carries %d bytes", regLen, len(b)-4)
	}
	r, err := DecodeWASMRegistration(b[4 : 4+regLen])
	if err != nil {
		return zero, nil, err
	}
	// The marker must be exactly the canonical encoding of what it decoded to.
	// DecodeWASMRegistration does not assert it CONSUMED its input, so without
	// this a declared length that overstates the encoding would let padding ride
	// inside the marker — the same hole checkWASMRegistrationArgs closes on the
	// outer frame, arrived at through the inner one.
	if enc := EncodeWASMRegistration(r); len(enc) != regLen || !bytes.Equal(enc, b[4:4+regLen]) {
		return zero, nil, errors.New("ops: wasm registration request marker is not canonical")
	}
	module := b[4+regLen:]
	if got := WASMBlobFingerprint(module); got != r.Blob {
		return zero, nil, fmt.Errorf("ops: wasm registration request: the %d-byte module hashes to %s, not to the %s the marker names",
			len(module), WASMBlobHex(got), WASMBlobHex(r.Blob))
	}
	return r, module, nil
}

// WASMRegistrationNewer reports whether candidate strictly supersedes installed
// under the (Epoch, fingerprint) total order described on WASMRegistration.Epoch.
// It is a pure function of the two registrations, so every replica agrees.
func WASMRegistrationNewer(candidate, installed WASMRegistration) bool {
	if candidate.Epoch != installed.Epoch {
		return candidate.Epoch > installed.Epoch
	}
	cf := WASMRegistrationFingerprint(candidate)
	inf := WASMRegistrationFingerprint(installed)
	return bytes.Compare(cf[:], inf[:]) > 0
}

// EncodeWASMRegistration serialises r into a length-prefixed wire format:
//
//	[name_len u16][name][kind u8][blob 32]
//	[export_len u16][export][maxfuel u64][epoch u64]
//
// All multi-byte integers are big-endian. The encoding is CANONICAL — the same
// WASMRegistration always produces the same bytes — because
// WASMRegistrationFingerprint hashes it, and every replica must derive the same
// fingerprint for the convergence rule to agree.
//
// THE FORMAT BREAK IS UNMIGRATED, deliberately (the app is unreleased): the
// [bytes_len u32][bytes] pair became a fixed 32-byte content address, so a marker
// written by an older build does not decode to the same registration here — it
// decodes to garbage or, far more often, fails the length checks below. Nothing
// bridges the two, and nothing should: a v1 marker's meaning ("the module is
// these inline bytes") has no expression in this format, which says "the module
// is whatever hashes to this". Every carrier of the old shape breaks with it —
// the Raft log entries of __register_wasm__, the wasm snapshot section
// (cluster.wasmSnapshotBlobVersion, bumped alongside), and the shard-scoped
// wrapper op's payload.
func EncodeWASMRegistration(r WASMRegistration) []byte {
	nameB := []byte(r.Name)
	exportB := []byte(r.ExportName)

	size := 2 + len(nameB) + // name_len + name
		1 + // kind
		sha256.Size + // blob content address
		2 + len(exportB) + // export_len + export
		8 + // maxfuel
		8 // epoch

	buf := make([]byte, size)
	off := 0

	binary.BigEndian.PutUint16(buf[off:off+2], uint16(len(nameB))) //nolint:gosec // bounded by string length, well within uint16
	off += 2
	copy(buf[off:], nameB)
	off += len(nameB)

	buf[off] = uint8(r.Kind)
	off++

	copy(buf[off:], r.Blob[:])
	off += sha256.Size

	binary.BigEndian.PutUint16(buf[off:off+2], uint16(len(exportB))) //nolint:gosec // bounded by string length, well within uint16
	off += 2
	copy(buf[off:], exportB)
	off += len(exportB)

	binary.BigEndian.PutUint64(buf[off:off+8], r.MaxFuel)
	off += 8

	binary.BigEndian.PutUint64(buf[off:off+8], r.Epoch)

	return buf
}

// DecodeWASMRegistration parses a buffer produced by EncodeWASMRegistration.
// It never panics and returns an error on any truncation or malformed input.
func DecodeWASMRegistration(b []byte) (WASMRegistration, error) {
	var r WASMRegistration
	off := 0

	// name_len (u16)
	if len(b) < 2 {
		return r, errors.New("ops: wasm registration too short for name_len")
	}
	nameLen := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2

	// name + kind(1) + blob(32) minimum check
	if off+nameLen+1+sha256.Size > len(b) {
		return r, errors.New("ops: wasm registration truncated at name/kind/blob")
	}
	r.Name = string(b[off : off+nameLen])
	off += nameLen

	// kind (u8)
	r.Kind = OpKind(b[off])
	off++

	// blob (32 bytes, fixed width — no length prefix, so there is nowhere for a
	// padding byte to hide inside this field)
	copy(r.Blob[:], b[off:off+sha256.Size])
	off += sha256.Size

	// export_len (u16)
	if off+2 > len(b) {
		return r, errors.New("ops: wasm registration truncated at export_len")
	}
	exportLen := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2

	// export + maxfuel(8) + epoch(8)
	if off+exportLen+8+8 > len(b) {
		return r, errors.New("ops: wasm registration truncated at export/maxfuel/epoch")
	}
	r.ExportName = string(b[off : off+exportLen])
	off += exportLen

	// maxfuel (u64)
	r.MaxFuel = binary.BigEndian.Uint64(b[off : off+8])
	off += 8

	// epoch (u64)
	r.Epoch = binary.BigEndian.Uint64(b[off : off+8])

	return r, nil
}

// WASMKeyExtractorHandle names the ONE key extractor every WASM op is registered
// with, and the reason WASMRegistration has no field to put it in is a silent
// cross-replica divergence.
//
// IT IS DOCUMENTATION AND A TEST ANCHOR, NOT A SETTING. Nothing on the
// registration path reads it — WASMKeyExtractor is what the code calls. It is
// kept because clients have to know the args shape they must produce, and because
// a named constant is what a test can assert the registration path actually uses.
//
// WHAT THE CHOICE COST. The extractor COMPUTES the shard group a proposal routes
// to (cluster.Node.Call: Lookup → ke → shardOf). The node-wide contract for an op
// is a fold over the set of registrations a node RECEIVED, and
// cluster.Node.checkWASMUpdateGate's forwarded-leg gate makes that set
// node-dependent ON PURPOSE. So two FIRST-TIME registrations of one name
// declaring different handles leave two nodes routing INV(X) to DIFFERENT shard
// groups: different replica sets apply it, every apply SUCCEEDS, and nothing
// errors. A differing Kind fails closed (errPBApplyReadOnly is classFatal); a
// differing extractor had no backstop at all, because there is no error to
// classify — the entry simply lands somewhere else.
//
// WHY NO FIELD RATHER THAN A VALIDATED FIELD. A validated field makes the bad
// state refusable; NO field makes it UNREPRESENTABLE. There is no wire byte to
// carry a second extractor, no struct member to set one on, and therefore no
// validator anywhere that can be forgotten, bypassed by a new entry point, or
// skipped on a path nobody thought of. That is the same move the blob store made
// when it stopped scanning *.wasm and scanned sidecars instead: express the
// invariant structurally and there is no check left to forget.
//
// WHY "std" IS THE SURVIVOR. It is [keyLen u16][key][payload] — the only shape
// that lets a routable op carry a payload distinct from its routing key. The
// retired "raw" handle (the whole args blob is the key) is a strict subset of it:
// any "raw" call is a "std" call whose payload is empty.
//
// A MODULE MUST SKIP THE PREFIX. The extractor selects the routing key; it does
// not rewrite the args the module receives, which are the full
// [keyLen u16][key][payload] frame. A module that uses its raw args as a cache
// key therefore writes under a key that does NOT route to the group it is
// executing in — legal (WASM ops are registered cross-shard) but almost never
// intended.
const WASMKeyExtractorHandle = "std"

// WASMKeyExtractor is the key extractor EVERY WASM op is registered with. It
// takes no argument, and that is the entire point: there is nothing to select, so
// there is nothing two nodes can select differently.
//
// It is a function rather than an exported variable so no caller can reassign the
// one extractor the whole cluster's routing agrees on.
func WASMKeyExtractor() KeyExtractor { return StdKeyExtractor }

// keyExtractorByHandle resolves a named handle to its KeyExtractor: "std" to
// StdKeyExtractor, anything else to nil (a SHARDLESS registration).
//
// NO WASM PATH CALLS THIS ANY MORE. WASM registrations use WASMKeyExtractor
// unconditionally; this survives for a caller registering a plain Go op, which
// may legitimately be shardless (ops.Registry.Register) — see
// docs/kv/custom-ops.md. Such a caller chooses the handle in its own source, so
// the value cannot differ between two nodes running the same binary, which is
// what made the WASM case unsafe and this one not.
//
// The "raw" arm (the whole args blob is the key) is GONE — see
// WASMKeyExtractorHandle.
func keyExtractorByHandle(handle string) KeyExtractor {
	if handle == WASMKeyExtractorHandle {
		return StdKeyExtractor
	}
	return nil
}

// KeyExtractorByHandle is the exported wrapper around keyExtractorByHandle: how a
// caller registering a plain Go op asks for the standard [keyLen u16][key]
// extractor. See docs/kv/custom-ops.md.
func KeyExtractorByHandle(handle string) KeyExtractor {
	return keyExtractorByHandle(handle)
}
