// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
)

// defaultMaxFuel is the per-Invoke instruction budget applied when a
// registration declares MaxFuel == 0 (the documented "unspecified" value). Fuel
// bounds pure-compute loops: when it is exhausted the guest traps. ~500M
// instructions is generous for legitimate update functions while still
// terminating a runaway (loop br 0).
//
// It lives in this build-tag-free file rather than beside the other runtime
// limits because ModuleIDFor NORMALIZES a zero budget to it, and that
// normalization has to produce the same ModuleID under CGO_ENABLED=0 as under
// CGO_ENABLED=1 — the ID is written into the on-disk sidecar's blob reference and
// compared across processes.
const defaultMaxFuel uint64 = 500_000_000

// ModuleID is the content address of an instantiated module slot: a hash of the
// (source bytes, export name, fuel budget) triple AddModule bakes into a
// runtime module. It is the Runtime's map key.
//
// IT COVERS THE EXPORT NAME AND THE FUEL BUDGET, not just the bytes, and that is
// not incidental. Both are compiled INTO the slot — the export is resolved to a
// function handle at instantiation and the budget is charged to the store on
// every Invoke — so two registrations that share bytes but declare a different
// export or a different cap are genuinely different executable objects and must
// not share a slot. Addressing them by sha256(bytes) alone would silently give
// the second one the first one's export and fuel.
//
// It is deliberately NOT the same value as the on-disk blob address (see
// ops.WASMBlobFingerprint), which hashes the BYTES alone — a blob file contains
// the bytes alone, and its whole contract is that it can be verified against its
// own filename.
type ModuleID [sha256.Size]byte

// String renders the ID as hex, for error messages and logs.
func (id ModuleID) String() string { return hex.EncodeToString(id[:]) }

// ModuleIDFor computes the ModuleID a call to AddModule with these arguments
// would produce. It applies the SAME normalization AddModule does (an empty
// export name means "apply", a zero fuel budget means defaultMaxFuel), so a
// caller cannot derive an ID that AddModule would never store.
func ModuleIDFor(src []byte, exportName string, maxFuel uint64) ModuleID {
	return ModuleIDForBlob(sha256.Sum256(src), exportName, maxFuel)
}

// ModuleIDForBlob is ModuleIDFor addressed by the module's CONTENT ADDRESS
// instead of by the module. blob is sha256 over the bytes alone — exactly
// ops.WASMBlobFingerprint, and exactly the value a thin registration marker
// carries.
//
// ############### THIS IS WHAT MAKES A MARKER SELF-SUFFICIENT ###############
//
// A registration names its module rather than carrying it, so
// everything downstream of the marker has to be derivable from the marker ALONE:
// the per-group binding table published to every apply goroutine, the route
// gate's evidence, the sidecar, the snapshot section. All of them need the
// ModuleID, and a node that has not fetched the bytes cannot compute one from
// them. Hashing the fingerprint instead of the bytes closes that: the ID is a
// pure function of (blob, export, fuel), all three of which are marker fields.
//
// IT AGREES WITH ModuleIDFor BY CONSTRUCTION, not by convention — ModuleIDFor is
// literally this function composed with sha256. That is load-bearing, because
// AddModule derives its key from the BYTES it just compiled while a binding
// derives the same key from a marker, and the two must name the same slot or a
// freshly fetched module would be invisible to the binding that asked for it.
// (This changed the ID's VALUE, since a fingerprint is hashed where the bytes
// used to be. Nothing persists a ModuleID — the sidecar stores the blob address
// and the module parameters, never the derived ID — so the change is confined to
// process memory.)
//
// Lengths are hashed alongside the values so no two distinct inputs can
// concatenate to the same byte stream. The fingerprint needs no length: it is
// fixed-width.
func ModuleIDForBlob(blob [sha256.Size]byte, exportName string, maxFuel uint64) ModuleID {
	if exportName == "" {
		exportName = "apply"
	}
	if maxFuel == 0 {
		maxFuel = defaultMaxFuel
	}
	h := sha256.New()
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], maxFuel)
	_, _ = h.Write(n[:])
	binary.BigEndian.PutUint64(n[:], uint64(len(exportName)))
	_, _ = h.Write(n[:])
	_, _ = h.Write([]byte(exportName))
	_, _ = h.Write(blob[:])
	var out ModuleID
	h.Sum(out[:0])
	return out
}
