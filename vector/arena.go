// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"os"
	"sync/atomic"
	"unsafe"
)

// arena holds fixed-stride float32 vectors and an id<->slot mapping. It is
// the canonical storage for an HNSW index: vectors are laid out contiguously
// for SIMD-friendly access, and slot indices (not user ids) flow through the
// hot graph-traversal paths.
//
// arena is not safe for concurrent use; the embedding VectorIndex
// implementation owns the mutex.
type arena struct {
	dim      int
	vecs     []float32  // vecs[slot*dim : (slot+1)*dim]
	expires  []uint64   // parallel to slots; 0 = no expiry, otherwise unix-millis deadline
	metadata []Metadata // parallel to slots; nil entries mean "no metadata"

	// versions is parallel to slots: a per-point MONOTONIC version counter for
	// optimistic concurrency (CAS). A live point has version >= 1 (its first
	// insert sets 1); every in-place mutation (upsert / payload op) bumps it by 1.
	// A reclaimed/free slot's version is cleared to 0, so an absent point reads 0.
	// It is a pure counter (no wall-clock/RNG), so the check+bump under the engine
	// lock is deterministic across Raft replicas. Persisted verbatim (snapshot /
	// WAL / sidecar) so a version survives restart, and carried on reshard copy.
	versions []uint64

	// keyExpires is parallel to metadata: per-slot, an optional map of payload
	// key -> absolute unix-millis deadline (0 / absent = no per-key TTL). A nil
	// per-slot map (the common case) means no payload key on that point expires.
	// A key is logically expired iff its deadline != 0 && <= now; expired keys
	// are dropped lazily on every read (never swept). Stored as ABSOLUTE
	// deadlines so snapshot/WAL replay is time-stable.
	keyExpires []map[string]uint64
	sparse     []*SparseVector   // parallel to slots; nil entries mean "no sparse vector"
	ids        []uint64          // parallel to slots; reverse of idMap (slot -> id). Stale for freed slots.
	free       []uint32          // LIFO free list of tombstoned slots
	idMap      map[uint64]uint32 // user id -> slot

	// deadlinePoints/deadlineKeys count slots currently carrying a point deadline
	// (expires[slot] != 0) / a non-empty per-key deadline map, respectively. The
	// TTL sweep's fast-path gate (ttl.go sweepOnce) reads their sum via
	// DeadlineSlots WITHOUT taking the owning index's write lock, so they must be
	// atomic even though every WRITE to them already happens under that lock
	// (arena is single-writer — see the type doc). Kept as two INDEPENDENT
	// counters (point vs. key) rather than one "either" counter so
	// SetExpires/SetKeyExpires each only need to compare their OWN before/after
	// value — no cross-field lookup. A slot carrying BOTH kinds is counted in
	// both; DeadlineSlots sums them, so the gate (sum == 0) is unaffected by the
	// slight overcount — the fast path only needs "definitely nothing to sweep",
	// never an exact distinct-slot count.
	deadlinePoints atomic.Int64
	deadlineKeys   atomic.Int64

	// Quantization (nil quant = disabled). When enabled, codes holds one
	// fixed-length code per slot (codes[slot*codeLen : (slot+1)*codeLen]),
	// encoded on Insert from the same vector stored in vecs. The float32 vecs
	// are retained for the exact rescore stage.
	quant   quantizer
	codeLen int
	codes   []byte

	// vecsDropped marks the IVF-PQ PQ-only state where the resident float32 vecs
	// have been released (only codes stay in RAM). When set, vecs is nil and slot
	// allocation / Capacity track nslots instead of len(vecs)/dim, since the
	// vectors are no longer the source of truth for the slot count.
	vecsDropped bool
	nslots      int

	// Mmap backing for vecs (nil mmapF = heap-backed). When set, vecs aliases
	// the memory-mapped region instead of the Go heap, so the float32 vectors
	// are not resident in heap memory; only the codes above stay in RAM.
	mmapF      *os.File
	mmapRegion []byte

	// poisoned marks a TERMINAL failure of an mmap slab growth. growVecMmap unmaps
	// the old view BEFORE it truncates and remaps; a grow that fails there now FIRST
	// tries to restore a mapping of the old (intact, append-only) data. What that
	// buys depends on WHICH slab was growing, because the two grows sit on opposite
	// sides of the point commit:
	//   - arena.growVecs (the vecs slab) runs INSIDE arena.Insert, BEFORE the point
	//     is committed. A restored failure aborts that Insert as a clean no-op, so
	//     the index stays fully usable and this flag is NOT set — only the single op
	//     errors. Poison is reached only in the rare DOUBLE failure (restore also
	//     fails), leaving the region freed and vecs nil'd.
	//   - hnsw.growLevel0 (the graph slab) runs from setNode, AFTER the point is
	//     committed to idMap/indexes, with no clean rollback. A grow failure there is
	//     ALWAYS terminal (poison) even when the mapping was restored, because
	//     "restored but usable" would leave a torn half-committed insert — strictly
	//     worse than poison. See growLevel0's error branch for the full rationale.
	// Either way, once set the guard is LAYERED so no op dereferences a freed region:
	//   - write / persist / snapshot chokepoints (insertBody, SavePersist,
	//     Snapshot) check the flag at the top and return ErrIndexPoisoned;
	//   - read chokepoints check UNDER h.mu, which also closes the TOCTOU against a
	//     concurrent grow: the error-returning read ops (SearchMMR, group, discover,
	//     the hybrid lane builders) reject with ErrIndexPoisoned, and the two shared
	//     under-lock helpers — searchDenseLockedWith (all dense search) and vecFor
	//     (the Get / materialization path) — return an empty/nil result so the
	//     no-error ops (SearchText, Get*) stay panic-safe;
	//   - the nil-out of the freed headers at the failure site is a BACKSTOP, so a
	//     path that ever slips past the flag faults cleanly instead of reading
	//     unmapped memory.
	// Set once (never cleared) under the owning index's write lock; read lock-free
	// via atomic so read-path ops can gate cheaply. A fresh arena installed by
	// snapshot restore starts un-poisoned, which is the intended recovery. A
	// poisoned arena OR a poisoned graph poisons the index; the graph failure sets
	// this same flag through h.arena (the arena is always present), keeping the
	// terminal state in one place.
	poisoned atomic.Bool

	// vecsRes, when non-nil, is the address-space reservation vecs is carved
	// out of — the mechanism that makes growth O(1) with a STABLE base address
	// instead of an O(n) copy/remap under the write lock (see reserve.go). It
	// backs BOTH modes: file-backed when mmapF is set, anonymous otherwise, so
	// a heap-mode arena gets the same stall-free growth as an mmap one. nil
	// means the slab is still on the legacy path (too small to be worth a
	// reservation, or the platform has none) — correct, just slower to grow.
	vecsRes *slabReservation

	// codesRes is the same thing for the quantization codes, which are the THIRD
	// flat per-slot slab and not a small one: SQ8 is dim bytes per slot, a
	// quarter of the float32 vectors, so at 1M x 768d its doubling copy is a
	// ~750 MB memcpy under the write lock — measured at 15.7 ms for a mere 25 MB
	// of codes, i.e. the dominant remaining stall on a quantized index once vecs
	// and level0 stopped copying. Always anonymous (codes are never
	// file-backed).
	codesRes *slabReservation

	// maxVectorsHint is the configured MaxVectors cap, or 0 when uncapped. It
	// sizes each slab's address-space reservation (per-slab stride applied at
	// use); it is never a limit — see slabReserveSize.
	maxVectorsHint int64
}

// mmapInitVectors is the initial vector capacity reserved when an arena is
// first mapped. Growth is geometric from there, so this only bounds the number
// of early remaps.
const mmapInitVectors = 1024

// growVecMmapFailpoint, when non-nil, is invoked by the real growVecMmap impls
// (mmap_linux.go and mmap_windows.go — the mmap_other.go stub has no mapping to
// grow and never consults it) right after the old view is unmapped; its returned
// error (if any) forces the grow to fail there — the Truncate/mmap is skipped and
// control goes straight to the restore path with that error. It exists ONLY so a
// test can deterministically simulate a Truncate/mmap failure (disk-full,
// address-space exhaustion) without one actually occurring; it is nil in
// production. Declared here (untagged) so the single definition is shared across
// the platform files and is visible to the platform-agnostic test.
var growVecMmapFailpoint func() error

// floatsOver reinterprets a byte region as a []float32 spanning its full
// length (len == cap == len(region)/4). The region must outlive the slice;
// arena rebuilds this header whenever the mapping moves.
func floatsOver(region []byte) []float32 {
	if len(region) == 0 {
		return nil
	}
	n := len(region) / 4
	// Audited: float32 contains no pointers (GC need not scan it), n is bounded
	// by the region length, and the mmap region outlives every slice header the
	// arena derives from it (rebuilt on remap, dropped on Close).
	//nolint:gosec // G103: reviewed unsafe reinterpretation of mmap bytes as float32.
	return unsafe.Slice((*float32)(unsafe.Pointer(&region[0])), n)
}

// newArena creates an arena with the given dimensionality. initCap is a hint
// for how many vectors to reserve initially (zero is allowed). Panics if
// dim <= 0 — Config.Validate already rejects that upstream, so reaching
// here with a non-positive dim is a programmer error (and would otherwise
// cause a divide-by-zero in Insert).
func newArena(dim, initCap int) *arena {
	if dim <= 0 {
		panic("vector: arena dim must be > 0")
	}
	return &arena{
		dim:        dim,
		vecs:       make([]float32, 0, initCap*dim),
		expires:    make([]uint64, 0, initCap),
		versions:   make([]uint64, 0, initCap),
		metadata:   make([]Metadata, 0, initCap),
		keyExpires: make([]map[string]uint64, 0, initCap),
		sparse:     make([]*SparseVector, 0, initCap),
		ids:        make([]uint64, 0, initCap),
		idMap:      make(map[uint64]uint32, initCap),
	}
}

// Reserve pre-grows all parallel slices to exactly n entries (len == cap == n)
// so a bulk concurrent build can write disjoint slots without append/realloc —
// keeping every slice header stable, which is what makes concurrent Vec reads
// race-free. Must be called on an empty arena before any Insert.
//
// For an mmap-backed arena the float32 region is grown to hold n vectors once,
// up front; the concurrent build then only writes disjoint slots (in its
// single-threaded setup phase) and reads the stable region (during the parallel
// link phase), so no remap happens mid-build.
func (a *arena) Reserve(n int) error {
	if len(a.idMap) != 0 {
		return ErrReserveNonEmpty
	}
	if a.mmapF == nil && len(a.vecs) != 0 {
		return ErrReserveNonEmpty
	}
	if err := a.growVecs(n * a.dim); err != nil {
		return err
	}
	a.vecs = a.vecs[:n*a.dim]
	a.expires = make([]uint64, n)
	a.versions = make([]uint64, n)
	a.metadata = make([]Metadata, n)
	a.keyExpires = make([]map[string]uint64, n)
	a.sparse = make([]*SparseVector, n)
	a.ids = make([]uint64, n)
	a.idMap = make(map[uint64]uint32, n)
	if a.quant != nil {
		a.growCodes(n * a.codeLen)
		a.codes = a.codes[:n*a.codeLen]
	}
	return nil
}

// PutAt writes vec (and id) directly into the pre-Reserved slot, without append.
// Used by the concurrent builder, which assigns slots by index. Concurrent calls
// for distinct slots are safe because the slices are pre-sized (no realloc) and
// write disjoint regions; the idMap write is serialized by the caller.
func (a *arena) PutAt(slot uint32, id uint64, vec []float32) {
	copy(a.vecs[int(slot)*a.dim:int(slot)*a.dim+a.dim], vec)
	a.ids[slot] = id
	a.versions[slot] = 1 // a bulk-built point is a fresh insert → version 1
	if a.quant != nil {
		a.quant.Encode(a.codes[int(slot)*a.codeLen:int(slot)*a.codeLen+a.codeLen], vec)
	}
}

// setQuant enables quantization on the arena. It must be called before any
// Insert (immediately after newArena). Codes are encoded on each Insert from
// the stored vector; the float32 vecs are kept for exact rescoring.
func (a *arena) setQuant(q quantizer) {
	a.quant = q
	a.codeLen = q.CodeLen()
	a.releaseCodesBacking() // see growCodes: a direct assignment must not strand a reservation
	a.codes = make([]byte, 0, cap(a.expires)*a.codeLen)
}

// enableCodes turns on a per-slot codes side-array of codeLen bytes WITHOUT a
// quantizer, so Insert/PutAt do NOT auto-encode. This is the IVF-PQ path: the
// codes are RESIDUAL PQ codes (vec − coarse-centroid), which only the IVF can
// compute, so the IVF encodes them and writes them via SetCode. Must be called
// before any Insert. Distinct from setQuant (which auto-encodes the raw vector).
func (a *arena) enableCodes(codeLen int) {
	a.codeLen = codeLen
	a.releaseCodesBacking() // see growCodes: a direct assignment must not strand a reservation
	a.codes = make([]byte, 0, cap(a.expires)*codeLen)
}

// SetCode writes code into slot's code slot (len(code) must be codeLen). Grows
// the codes array to cover slot when needed (an incremental Insert appends a
// fresh slot before its residual code is known). Used by the IVF-PQ path, which
// computes the residual code itself; valid only when codeLen > 0.
func (a *arena) SetCode(slot uint32, code []byte) {
	need := (int(slot) + 1) * a.codeLen
	if need > len(a.codes) {
		a.growCodes(need)
		a.codes = a.codes[:need]
	}
	copy(a.codes[int(slot)*a.codeLen:int(slot)*a.codeLen+a.codeLen], code)
}

// dropVecs releases the resident float32 vectors (PQ-only compression: only the
// codes stay in RAM). Heap-only; an mmap-backed arena is never used with IVF-PQ
// (IVF rejects Persistent). It snapshots the current slot count into nslots so
// Capacity/Insert keep working after vecs is nil, and flips vecsDropped so the
// slot-allocation path no longer derives the count from len(vecs). After this,
// Vec must not be called — the IVF reconstructs an approximate vector from the
// code + centroid when one is needed.
func (a *arena) dropVecs() {
	a.nslots = len(a.vecs) / a.dim
	a.vecs = nil
	a.vecsDropped = true
	a.releaseVecsBacking()
}

// releaseVecsBacking hands the vector slab's address-space reservation back to
// the OS. THE RULE: a reservation is off-heap, so nil-ing the vecs header does
// NOT free it and there is no finalizer to catch the omission — every path that
// discards or replaces a.vecs must come through here (or through Close). This
// exists as its own method precisely because that rule has more than one caller:
// dropVecs, and the IVF's PQ-drop restore, which reconstructs the dropped state
// field by field rather than calling dropVecs.
func (a *arena) releaseVecsBacking() {
	if a.vecsRes == nil {
		return
	}
	_ = a.vecsRes.release()
	a.vecsRes = nil
	a.mmapRegion = nil
}

// releaseCodesBacking is releaseVecsBacking for the codes slab. Every site that
// assigns a.codes a FRESH backing array directly — setQuant, enableCodes, and
// the PQ-drop restore, all of which legitimately bypass growCodes because they
// are (re)initializing rather than growing — must call it first. Skipping it
// would both leak the reservation and, worse, silently strand any codes already
// written into it.
func (a *arena) releaseCodesBacking() {
	if a.codesRes == nil {
		return
	}
	_ = a.codesRes.release()
	a.codesRes = nil
}

// useMmap backs the arena's float32 vectors with a memory-mapped file at path
// instead of the Go heap. It must be called before any Insert (immediately
// after setQuant). The initial mapping reserves mmapInitVectors slots; the
// arena grows it geometrically on demand.
func (a *arena) useMmap(path string) error {
	size := int64(mmapInitVectors * a.dim * 4)
	f, region, err := openVecMmap(path, size)
	if err != nil {
		return err
	}
	a.mmapF = f
	a.mmapRegion = region
	a.vecs = floatsOver(region)[:0]
	return nil
}

// loadMmapVecs attaches an EXISTING vector mmap file (written by a prior run),
// mapping it at its current on-disk size without truncating, and slices vecs to
// the first n vectors. Used by instant-restart (persist.go); the file content
// is reused as-is, so the float32 vectors come back zero-copy.
func (a *arena) loadMmapVecs(path string, n int) error {
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	if fi.Size() < int64(n*a.dim*4) {
		return fmt.Errorf("vector: vector file %s too small (%d bytes) for %d x %d", path, fi.Size(), n, a.dim)
	}
	f, region, err := openVecMmap(path, fi.Size())
	if err != nil {
		return err
	}
	a.mmapF = f
	a.mmapRegion = region
	a.vecs = floatsOver(region)[:n*a.dim]
	return nil
}

// restoreDense rebuilds the per-slot bookkeeping for an arena from a persisted
// id list: ids/idMap, empty side-arrays, and re-encoded codes (from the
// already-mapped vecs). Used by instant-restart. Slots in holes (reclaimed free
// holes) are excluded from idMap so their stale ids never resolve; the caller
// repopulates a.free. Tombstoned slots stay live in idMap (filtered at query).
func (a *arena) restoreDense(ids []uint64, holes map[uint32]bool) {
	n := len(ids)
	a.ids = ids
	a.idMap = make(map[uint64]uint32, n)
	for slot, id := range ids {
		if holes[uint32(slot)] { //nolint:gosec // slot < n < 2^32
			continue
		}
		a.idMap[id] = uint32(slot) //nolint:gosec // slot < n < 2^32
	}
	a.expires = make([]uint64, n)
	a.versions = make([]uint64, n)
	a.metadata = make([]Metadata, n)
	a.keyExpires = make([]map[string]uint64, n)
	a.sparse = make([]*SparseVector, n)
	a.free = nil
	if a.quant != nil {
		// Through growCodes, so an instant-restart lands on the same reservation an
		// incrementally grown index would have: the loop below writes every slot, so
		// nothing depends on the region starting zeroed.
		a.growCodes(n * a.codeLen)
		a.codes = a.codes[:n*a.codeLen]
		for slot := 0; slot < n; slot++ {
			a.quant.Encode(a.codes[slot*a.codeLen:(slot+1)*a.codeLen], a.vecs[slot*a.dim:(slot+1)*a.dim])
		}
	}
}

// restoreDenseDropped is the float-drop (PQDropVecs) variant of restoreDense used
// by instant-restart (persist.go v9). The resident floats are gone, so it does NOT
// map a vecs file or re-encode codes; instead it loads the codes VERBATIM (the only
// post-drop source of truth) and flips the arena into the dropped state (vecs nil,
// vecsDropped, nslots tracks the slot count for Capacity/Insert). Everything else
// mirrors restoreDense: ids/idMap (excluding holes) + empty per-slot side-arrays.
func (a *arena) restoreDenseDropped(ids []uint64, holes map[uint32]bool, codeLen int, codes []byte) {
	n := len(ids)
	a.ids = ids
	a.idMap = make(map[uint64]uint32, n)
	for slot, id := range ids {
		if holes[uint32(slot)] { //nolint:gosec // slot < n < 2^32
			continue
		}
		a.idMap[id] = uint32(slot) //nolint:gosec // slot < n < 2^32
	}
	a.expires = make([]uint64, n)
	a.versions = make([]uint64, n)
	a.metadata = make([]Metadata, n)
	a.keyExpires = make([]map[string]uint64, n)
	a.sparse = make([]*SparseVector, n)
	a.free = nil
	a.codeLen = codeLen
	// Both slabs are being REPLACED wholesale (codes verbatim from the sidecar,
	// vecs gone for good), so any reservation either of them was carved out of has
	// to go back to the OS first — nil-ing an off-heap header frees nothing and
	// there is no finalizer. See growCodes / releaseVecsBacking.
	a.releaseCodesBacking()
	a.codes = codes
	a.releaseVecsBacking()
	a.vecs = nil
	a.vecsDropped = true
	a.nslots = n
}

// growVecs ensures the vecs slice has capacity for at least needFloats
// elements, growing geometrically. It is the SINGLE growth chokepoint for the
// vector slab in every backing — mmap, heap, and either of those on a
// reservation — so the ordering of the three strategies lives in exactly one
// place. The logical length is preserved; the caller reslices up to its target.
//
// The strategies, cheapest first:
//
//  1. Commit forward inside the existing reservation. One syscall against the
//     delta, base address unchanged, nothing copied. This is the steady state
//     for any slab past slabReserveThreshold and the whole point of the design.
//  2. Take (or, on exhaustion, retake at a larger size) a reservation. Costs one
//     copy for a heap slab and zero for an mmap one (the bytes are in the file,
//     which the new reservation maps from offset 0). Rare by construction.
//  3. The legacy path: remap the whole file, or reallocate and copy the whole
//     slab. Correct, O(n), and what every platform without reservations does.
//
// Every path preserves the existing contents, and every path runs under the
// index write lock, so the slab-growth invariant (grow_race_test.go) is
// unchanged — only the cost of the critical section moves.
func (a *arena) growVecs(needFloats int) error {
	if cap(a.vecs) >= needFloats {
		return nil
	}
	newCap := cap(a.vecs) * 2
	if newCap < needFloats {
		newCap = needFloats
	}
	oldLen := len(a.vecs)
	newBytes := int64(newCap) * 4

	if a.vecsRes != nil {
		err := a.vecsRes.commitTo(newBytes)
		if err == nil {
			a.adoptVecs(a.vecsRes.region(), oldLen)
			return nil
		}
		if !errors.Is(err, errSlabReserveExhausted) {
			return err
		}
	}
	if newBytes >= slabReserveThreshold {
		ok, err := a.reserveVecs(newBytes, oldLen)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
	}
	if a.mmapF != nil {
		region, err := growVecMmap(a.mmapF, a.mmapRegion, newBytes)
		if err != nil {
			if region != nil {
				// Grow failed, but growVecMmap restored a valid mapping of the OLD data
				// (grow only appends, so bytes [0,oldLen) are intact). Rebind to it at the
				// old logical length: the index stays fully usable and only THIS operation
				// fails. len(region) == oldLen*4 bytes, so the reslice is in-bounds. No poison.
				a.mmapRegion = region
				a.vecs = floatsOver(region)[:oldLen]
				return err
			}
			// Grow AND restore both failed: the old backing is gone with nothing valid to
			// fall back to. Poison the index so every public op rejects with
			// ErrIndexPoisoned, and nil the headers (belt-and-suspenders) so a bug that
			// bypasses the guard faults cleanly instead of reading unmapped memory.
			// (a.codes has no file backing — leave it.) This is now the rare backstop.
			a.poisoned.Store(true)
			a.mmapRegion = nil
			a.vecs = nil
			return err
		}
		a.mmapRegion = region
		a.vecs = floatsOver(region)[:oldLen]
		return nil
	}
	grown := make([]float32, oldLen, newCap)
	copy(grown, a.vecs)
	a.vecs = grown
	return nil
}

// adoptVecs rebuilds the vecs header over region, keeping logicalLen elements.
// For a file-backed slab the region is also what gets msync'd, so mmapRegion
// tracks it.
func (a *arena) adoptVecs(region []byte, logicalLen int) {
	if a.mmapF != nil {
		a.mmapRegion = region
	}
	a.vecs = floatsOver(region)[:logicalLen]
}

// reserveVecs moves the slab onto a fresh reservation sized for newBytes,
// carrying the first oldLen floats across. Reports false (with no error) when no
// reservation could be made, which is not a failure — the caller falls back to
// the legacy growth path.
func (a *arena) reserveVecs(newBytes int64, oldLen int) (bool, error) {
	hint := slabHintBytes(a.maxVectorsHint, a.dim*4)
	res, err := newSlabReservation(a.mmapF, slabReserveSize(newBytes, hint), newBytes)
	if err != nil {
		return false, nil //nolint:nilerr // a reservation is best-effort; the caller has a correct slower path
	}
	old := a.vecsRes
	if a.mmapF != nil {
		// File-backed: the contents live in the FILE and the new reservation maps
		// that same file from offset 0, so every already-written byte is already
		// visible at the new base — nothing is copied. Only the OLD mapping of the
		// file has to go.
		if old != nil {
			_ = old.release()
		} else if uerr := unmapVecMmap(a.mmapRegion); uerr != nil {
			_ = res.release()
			return false, uerr
		}
		a.vecsRes = res
		a.adoptVecs(res.region(), oldLen)
		return true, nil
	}
	// Anonymous: the one copy this scheme ever pays, and only when a heap slab
	// first crosses the threshold or outgrows its reservation outright.
	dst := floatsOver(res.region())
	copy(dst[:oldLen], a.vecs[:oldLen])
	a.vecsRes = res
	a.vecs = dst[:oldLen]
	if old != nil {
		_ = old.release()
	}
	return true, nil
}

// growCodes ensures the codes slab has capacity for at least needBytes,
// growing geometrically. Mirrors growVecs, minus the file backing (codes are
// always anonymous) — and minus the error return, because its two callers
// (Insert and the IVF's SetCode) have no useful failure to report: a
// reservation that cannot be made or extended simply falls back to a heap
// reallocation, which is what the code did before reservations existed.
//
// PRECONDITION, and the reason this comment is long. growCodes is the only
// place allowed to GROW a.codes. Anything that assigns a.codes a different
// backing array must go through releaseCodesBacking first, because a.codesRes
// and a.codes are two halves of one fact: if a.codesRes is live while a.codes
// points somewhere else, the reservation is both leaked (no finalizer, by
// design — see reserve.go) and, far worse, still the destination the NEXT
// growCodes will commit into and copy the now-wrong prefix from. That is silent
// data loss, not a crash, so it cannot be left to be noticed at runtime.
//
// The legitimate direct assigners are the (re)initializers — setQuant,
// enableCodes, restoreDenseDropped — and each calls releaseCodesBacking. The
// restore paths in snapshot.go / ivf.go assign a.codes too, but only on arenas
// they just constructed with newArena, where codesRes is nil by construction.
func (a *arena) growCodes(needBytes int) {
	if cap(a.codes) >= needBytes {
		return
	}
	newCap := cap(a.codes) * 2
	if newCap < needBytes {
		newCap = needBytes
	}
	oldLen := len(a.codes)

	if a.codesRes != nil {
		if err := a.codesRes.commitTo(int64(newCap)); err == nil {
			a.codes = a.codesRes.region()[:oldLen]
			return
		}
		// Exhausted (or, in principle, a kernel refusal): relocate below.
	}
	if int64(newCap) >= slabReserveThreshold {
		hint := slabHintBytes(a.maxVectorsHint, a.codeLen)
		if res, err := newSlabReservation(nil, slabReserveSize(int64(newCap), hint), int64(newCap)); err == nil {
			dst := res.region()
			copy(dst[:oldLen], a.codes[:oldLen])
			old := a.codesRes
			a.codesRes = res
			a.codes = dst[:oldLen]
			if old != nil {
				_ = old.release()
			}
			return
		}
	}
	grown := make([]byte, oldLen, newCap)
	copy(grown, a.codes)
	a.codes = grown
}

// Close releases the vector backing, if any: the address-space reservation
// and/or the mmap. Idempotent, and a no-op for a heap-backed arena that never
// grew past the reservation threshold.
func (a *arena) Close() error {
	if a.codesRes != nil {
		a.releaseCodesBacking() // same chokepoint as every other discard site
		a.codes = nil
	}
	if a.vecsRes == nil && a.mmapF == nil {
		return nil
	}
	var err error
	if a.vecsRes != nil {
		if serr := a.vecsRes.sync(); serr != nil {
			err = serr
		}
		if rerr := a.vecsRes.release(); rerr != nil && err == nil {
			err = rerr
		}
		a.vecsRes = nil
		if a.mmapF != nil {
			if cerr := a.mmapF.Close(); cerr != nil && err == nil {
				err = fmt.Errorf("vector: close: %w", cerr)
			}
		}
	} else {
		err = closeVecMmap(a.mmapF, a.mmapRegion)
	}
	a.mmapF = nil
	a.mmapRegion = nil
	a.vecs = nil
	return err
}

// codePtr returns a pointer to the start of slot's code, for issuing a prefetch.
// Valid only when quantization is enabled and slot < Capacity.
func (a *arena) codePtr(slot uint32) unsafe.Pointer {
	return unsafe.Pointer(&a.codes[int(slot)*a.codeLen])
}

// Code returns the code slice for slot. Valid only when quantization is
// enabled. The returned slice aliases arena storage; callers must not retain
// it across Insert / Delete.
func (a *arena) Code(slot uint32) []byte {
	start := int(slot) * a.codeLen
	return a.codes[start : start+a.codeLen]
}

// Insert stores vec under id and returns the assigned slot. Returns
// ErrDimMismatch if len(vec) != dim, or ErrDuplicateID if id is already
// present (caller must Delete first).
func (a *arena) Insert(id uint64, vec []float32) (uint32, error) {
	if len(vec) != a.dim {
		return 0, ErrDimMismatch
	}
	if _, ok := a.idMap[id]; ok {
		return 0, ErrDuplicateID
	}
	var slot uint32
	if n := len(a.free); n > 0 {
		slot = a.free[n-1]
		a.free = a.free[:n-1]
		if !a.vecsDropped {
			copy(a.vecs[int(slot)*a.dim:int(slot)*a.dim+a.dim], vec)
		}
		a.expires[slot] = 0
		a.versions[slot] = 0 // cleared on reuse; the caller sets the new point's version
		a.metadata[slot] = nil
		a.keyExpires[slot] = nil
		a.sparse[slot] = nil
		a.ids[slot] = id
		if a.quant != nil {
			a.quant.Encode(a.codes[int(slot)*a.codeLen:int(slot)*a.codeLen+a.codeLen], vec)
		}
	} else {
		if a.vecsDropped {
			// PQ-only: floats are gone; allocate the next slot from the tracked
			// count and skip the vecs write entirely (the IVF encodes the residual
			// code via SetCode after this returns).
			slot = uint32(a.nslots) //nolint:gosec // bounded by 2^32 vectors per arena
			a.nslots++
		} else {
			slot = uint32(len(a.vecs) / a.dim) //nolint:gosec // bounded by 2^32 vectors per arena
			// One path for every backing: ask growVecs for the room, then write in
			// place. It replaces the heap-mode `append`, which would silently pull a
			// reservation-backed slab back onto the Go heap (and copy it) the moment
			// it overflowed.
			need := len(a.vecs) + a.dim
			if need > cap(a.vecs) {
				if err := a.growVecs(need); err != nil {
					return 0, err
				}
			}
			a.vecs = a.vecs[:need]
			copy(a.vecs[need-a.dim:need], vec)
		}
		a.expires = append(a.expires, 0)
		a.versions = append(a.versions, 0) // grown alongside; the caller sets the new point's version
		a.metadata = append(a.metadata, nil)
		a.keyExpires = append(a.keyExpires, nil)
		a.sparse = append(a.sparse, nil)
		a.ids = append(a.ids, id)
		if a.quant != nil {
			base := len(a.codes)
			a.growCodes(base + a.codeLen)
			a.codes = a.codes[:base+a.codeLen]
			a.quant.Encode(a.codes[base:base+a.codeLen], vec)
		}
	}
	a.idMap[id] = slot
	return slot, nil
}

// ID returns the user id stored at slot. Valid only for live slots (those
// present in idMap); freed slots retain a stale id until reused. The reverse
// of idMap, kept as a slice so slot->id is O(1) on the search result path.
func (a *arena) ID(slot uint32) uint64 {
	return a.ids[slot]
}

// Allocated reports whether slot currently holds a live point, via the idMap
// round-trip (slot -> id -> idMap slot must come back to slot).
//
// This is the EXPLICIT form of a check that used to be spelled `ID(slot) == 0`.
// That spelling was wrong twice over, because a.ids is a plain zero-valued
// []uint64 and the arena hands out slots from two sources:
//
//   - A reserved-but-unwritten slot (Reserve pre-sizes a.ids to n) reads back id
//     0 — but so does a point genuinely stored under user id 0, so the old check
//     made id 0 permanently unreachable on every path that used it.
//   - A slot on the free list retains a STALE, usually NON-zero id until reused,
//     which the old check waved straight through.
//
// The round-trip answers both correctly and is what liveSlotLockedAt and
// arenaSlotActiveLocked already use. Callers must hold the index lock (read is
// enough): idMap is only mutated under the write lock.
func (a *arena) Allocated(slot uint32) bool {
	if int(slot) >= len(a.ids) {
		return false
	}
	cur, ok := a.idMap[a.ids[slot]]
	return ok && cur == slot
}

// ExpiresAt returns the unix-millisecond deadline for slot, or 0 if no expiry.
func (a *arena) ExpiresAt(slot uint32) uint64 {
	return a.expires[slot]
}

// SetExpires sets the unix-millisecond deadline for slot. 0 clears expiry.
// Maintains deadlinePoints: a 0->nonzero transition increments it, a
// nonzero->0 transition decrements it (a nonzero->nonzero overwrite, e.g. a
// TTL refresh, is a no-op for the counter).
func (a *arena) SetExpires(slot uint32, ms uint64) {
	old := a.expires[slot]
	if old == 0 && ms != 0 {
		a.deadlinePoints.Add(1)
	} else if old != 0 && ms == 0 {
		a.deadlinePoints.Add(-1)
	}
	a.expires[slot] = ms
}

// Version returns the per-point version counter at slot. A live point has
// version >= 1; a reclaimed/free slot reads 0.
func (a *arena) Version(slot uint32) uint64 {
	return a.versions[slot]
}

// SetVersion sets the version counter at slot verbatim — the version-restoring
// primitive used by snapshot/WAL/sidecar replay and reshard copy (which must
// preserve the exact version, NOT re-bump it).
func (a *arena) SetVersion(slot uint32, v uint64) {
	a.versions[slot] = v
}

// BumpVersion increments the version counter at slot by 1 and returns the new
// value — the post-mutation step every successful in-place write performs.
func (a *arena) BumpVersion(slot uint32) uint64 {
	a.versions[slot]++
	return a.versions[slot]
}

// Sparse returns the sparse vector for slot, or nil if none. The returned
// pointer aliases arena storage; callers MUST NOT mutate it.
func (a *arena) Sparse(slot uint32) *SparseVector {
	return a.sparse[slot]
}

// SetSparse stores sv for slot. A nil sv clears any existing sparse vector.
// The pointer is stored by reference (the caller hands off ownership).
func (a *arena) SetSparse(slot uint32, sv *SparseVector) {
	a.sparse[slot] = sv
}

// Metadata returns the metadata map for slot, or nil if absent. The returned
// map aliases the arena's storage; callers MUST NOT mutate it.
func (a *arena) Metadata(slot uint32) Metadata {
	return a.metadata[slot]
}

// SetMetadata stores meta for slot. A nil meta clears any existing entry.
// The map is stored by reference; the caller must not mutate it after the
// hand-off.
func (a *arena) SetMetadata(slot uint32, meta Metadata) {
	a.metadata[slot] = meta
}

// KeyExpires returns the per-key deadline map for slot, or nil if the slot has
// no per-key TTL (the common case). The returned map aliases arena storage;
// callers MUST NOT mutate it (the set path clones-on-write).
func (a *arena) KeyExpires(slot uint32) map[string]uint64 {
	return a.keyExpires[slot]
}

// SetKeyExpires stores m as the per-key deadline map for slot. A nil or empty
// map clears any existing per-key TTL. The map is stored by reference; the
// caller must not mutate it after the hand-off. Maintains deadlineKeys: an
// empty->non-empty transition increments it, a non-empty->empty transition
// decrements it.
func (a *arena) SetKeyExpires(slot uint32, m map[string]uint64) {
	before := len(a.keyExpires[slot]) > 0
	after := len(m) > 0
	if !before && after {
		a.deadlineKeys.Add(1)
	} else if before && !after {
		a.deadlineKeys.Add(-1)
	}
	if len(m) == 0 {
		a.keyExpires[slot] = nil
		return
	}
	a.keyExpires[slot] = m
}

// DeadlineSlots returns the current count of slots carrying a TTL deadline of
// EITHER kind (point or per-key) — 0 iff there is nothing for the TTL sweep to
// do. Safe to call without the owning index's write lock (see the field doc
// on deadlinePoints/deadlineKeys).
func (a *arena) DeadlineSlots() int64 {
	return a.deadlinePoints.Load() + a.deadlineKeys.Load()
}

// KeyDeadlineSlots returns the count of slots carrying a PER-KEY deadline map —
// 0 iff liveMeta is guaranteed to return the arena's metadata unchanged for
// every slot. The bitset admission gate reads it to decide whether the payload
// index can be trusted as an EXACT answer: a key past its deadline is hidden
// from the predicate by liveMeta but is still in the posting set until the slot
// is reindexed, so a non-zero count downgrades the gate to a pre-filter. Same
// lock-freedom as DeadlineSlots.
func (a *arena) KeyDeadlineSlots() int64 {
	return a.deadlineKeys.Load()
}

// RecomputeDeadlineCounts recounts deadlinePoints/deadlineKeys from scratch
// against the current expires/keyExpires arrays. Used after a bulk restore
// (openPersist / hnsw.Restore) that populates those arrays directly rather
// than through SetExpires/SetKeyExpires, bypassing their incremental counter
// maintenance.
func (a *arena) RecomputeDeadlineCounts() {
	var points, keys int64
	for _, e := range a.expires {
		if e != 0 {
			points++
		}
	}
	for _, ke := range a.keyExpires {
		if len(ke) > 0 {
			keys++
		}
	}
	a.deadlinePoints.Store(points)
	a.deadlineKeys.Store(keys)
}

// Vec returns the slice for the vector at slot. The returned slice aliases
// the arena's storage; callers must not retain it across Insert / Delete.
func (a *arena) Vec(slot uint32) []float32 {
	start := int(slot) * a.dim
	return a.vecs[start : start+a.dim]
}

// Slot returns the slot for id, or false if absent.
func (a *arena) Slot(id uint64) (uint32, bool) {
	s, ok := a.idMap[id]
	return s, ok
}

// Delete removes id from the arena, freeing its slot for reuse. Returns
// true if id was present.
func (a *arena) Delete(id uint64) bool {
	slot, ok := a.idMap[id]
	if !ok {
		return false
	}
	// THE chokepoint for the TTL-gate counters on slot teardown. A slot on the
	// free list carries no deadline of either kind, and Insert's free-list reuse
	// zeroes expires/keyExpires DIRECTLY (bypassing the setters), so if the
	// decrement did not happen here it would never happen at all — the counters
	// would drift up forever, one per reused slot that still held a deadline.
	// Owning it here means no caller can forget: hnsw's insertLocked resurrects
	// an EXPIRED-but-not-yet-swept slot whose deadlines are still live, and
	// ivf's resurrect path does the same; neither Reclaim clears anything.
	//
	// Both setters are IDEMPOTENT in the clearing direction (0->0 and empty->
	// empty touch no counter), so the paths that already clear before calling
	// Delete — hnsw's tombstone and sweep paths clear at death so the gate
	// closes immediately, long before Reclaim frees the slot — cannot
	// double-decrement. Overcounting only pins the sweep gate open (a perf
	// loss); UNDERCOUNTING would make TTLs silently never expire, so the
	// idempotence is load-bearing, not incidental.
	a.SetExpires(slot, 0)
	a.SetKeyExpires(slot, nil)
	delete(a.idMap, id)
	a.free = append(a.free, slot)
	return true
}

// Size returns the number of live vectors.
func (a *arena) Size() int {
	return len(a.idMap)
}

// Capacity returns the total number of slots in use (live + tombstoned).
// dim is guaranteed positive by newArena's invariant, so no zero-guard needed.
func (a *arena) Capacity() int {
	if a.vecsDropped {
		return a.nslots
	}
	return len(a.vecs) / a.dim
}
