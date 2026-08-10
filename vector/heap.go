// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// slotDist is one (slot, distance) pair flowing through search heaps.
// Slot is the arena slot index; dist is the distance under the active metric
// (smaller = more similar; can be negative for DotProduct).
type slotDist struct {
	slot uint32
	dist float32
}

// The search heaps do NOT store slotDist. They store a PACKED uint64 key whose
// unsigned ordering is exactly the (dist, slot) lexicographic ordering:
//
//	bits 63..32 = orderBits(dist) — a monotone float32 -> uint32 mapping
//	bits 31..0  = slot
//
// Two things follow. First, every heap comparison is a single integer compare,
// which the Go compiler lowers to CMP + CMOV instead of the data-dependent
// branch a float32 compare emits — the sift loops become branchless in their
// hot inner step (the float compare was measured at 24% of searchLayerCore,
// with the mispredicted branch the dominant term). Second, ties in distance are
// broken deterministically by ascending slot instead of by arbitrary heap
// order, so a result set over exact-distance ties is now reproducible.
//
// ±Inf order correctly under this mapping. NaN does not — it has no place in a
// total order at all — so orderBits maps it to a sentinel that sorts LAST; see
// there for why that specific choice, and why "NaN cannot occur" was not a
// defensible assumption to build on.

// nanOrder is the sentinel orderBits assigns to NaN. It is above every real
// distance (the largest of which, +Inf, maps to 0xFF800000), so a NaN-distanced
// candidate sorts last and can never displace a genuine result.
const nanOrder = math.MaxUint32

// orderBits maps a float32 onto a uint32 whose UNSIGNED ordering matches the
// float's ordering. The standard total-order transform: for the raw bit pattern
// x, flip the sign bit when the value is non-negative (moving positives above
// the negative half), and flip ALL bits when it is negative (reversing the
// sign-magnitude ordering of the negatives into ascending order).
//
// A naive bit-cast is WRONG here: pickDist returns -dotProduct for the
// DotProduct metric (see distance.go), so negative distances are routine.
//
// NaN is special-cased, and NOT defensively: under the plain transform a
// NEGATIVE NaN maps to 0x003FFFFF, which sorts below -Inf — i.e. first. That is
// reachable. Under DotProduct pickDist returns -dot, and negating a +NaN yields
// a -NaN, so a single NaN component in a stored vector would make that point
// rank #1 for EVERY query. Nor is it hypothetical that a NaN gets stored:
// arena.Insert validates only dimension, and while the HTTP/JSON path is
// incidentally safe (JSON has no NaN literal and encoding/json rejects float
// overflow), the gRPC/protobuf ingress carries an arbitrary 4-byte float
// unchecked, the raft apply path re-decodes the same raw bits without
// revalidating, and a Go API caller can pass one directly.
//
// Mapping NaN last also restores the pre-packed-key behavior: the old float
// compare `d < worstKept` is false for NaN, so a NaN candidate could never
// displace a real result. Here it cannot either — and now it does so
// deterministically rather than by whatever order the heap happened to be in.
//
// The test is integer (exponent all ones, mantissa non-zero) on a value already
// in a register, so it costs an AND and a CMP against a branch that is never
// taken in practice — nothing measurable next to the distance kernel that
// produced the input.
func orderBits(dist float32) uint32 {
	b := math.Float32bits(dist)
	if b&0x7FFFFFFF > 0x7F800000 { // NaN: no total order — sort it last
		return nanOrder
	}
	if b&0x80000000 != 0 {
		return ^b
	}
	return b ^ 0x80000000
}

// fromOrderBits inverts orderBits. The NaN sentinel maps back to a NaN (a
// different payload than went in, but still NaN): a caller reading the distance
// of such a result should see that it is not a number, not a plausible-looking
// finite value.
func fromOrderBits(o uint32) float32 {
	if o&0x80000000 != 0 {
		return math.Float32frombits(o ^ 0x80000000)
	}
	return math.Float32frombits(^o)
}

// packKey builds the sortable heap key for (slot, dist).
func packKey(slot uint32, dist float32) uint64 {
	return uint64(orderBits(dist))<<32 | uint64(slot)
}

// keyDist returns the ordering bits of a packed key's distance. Callers that
// must ignore the slot tiebreak — the ef-termination test and the "closer than
// the worst kept" admission test — compare these instead of whole keys, which
// preserves their original semantics on equal distances.
//
// orderBits is strictly monotone on DISTINCT float32 values, so for any two
// distances that compare unequal, comparing keyDist gives the same answer as
// comparing the floats. The one input pair where it differs is -0.0 vs +0.0:
// they are float-equal, but orderBits maps them to 0x7FFFFFFF and 0x80000000,
// so keyDist reports -0.0 as marginally closer. That is a tiebreak between two
// candidates at distance zero, not a mis-ordering — both are equally near, and
// which one wins was already arbitrary.
func keyDist(key uint64) uint32 { return uint32(key >> 32) }

// keySlot returns the slot packed into a heap key.
func keySlot(key uint64) uint32 { return uint32(key) }

// keyUnpack expands a packed key back into a slotDist.
func keyUnpack(key uint64) slotDist {
	return slotDist{slot: keySlot(key), dist: fromOrderBits(keyDist(key))}
}

// minHeap is a binary min-heap of packed keys ordered ascending. Used as the
// search frontier: pop returns the closest unexplored candidate.
//
// The sift operations are open-coded on the concrete []uint64 instead of going
// through container/heap, which would box every element into an interface{} on
// Push/Pop — one allocation per graph edge explored. Both sift loops use the
// "hole" formulation (shift the losing element into the vacancy, write the
// moving key once at the end) rather than swapping: half the stores, and the
// key stays in a register across the whole descent.
type minHeap []uint64

func (h minHeap) len() int { return len(h) }

func (h *minHeap) push(key uint64) {
	a := append(*h, key)
	i := len(a) - 1
	for i > 0 {
		parent := (i - 1) / 2
		pv := a[parent]
		if pv <= key {
			break
		}
		a[i] = pv
		i = parent
	}
	a[i] = key
	*h = a
}

// pop removes and returns the minimum key. Caller must ensure len > 0.
func (h *minHeap) pop() uint64 {
	a := *h
	top := a[0]
	n := len(a) - 1
	if n > 0 {
		siftDownMin(a[:n], a[n])
	}
	*h = a[:n]
	return top
}

// siftDownMin places key at the root of the (otherwise valid) min-heap a,
// sifting it down to its ordered position. len(a) must be > 0.
//
// The child-index tests are written as unsigned comparisons: uint(l) < uint(n)
// tells the prover both l >= 0 (ruling out the 2*i+1 overflow it otherwise has
// to assume) and l < len(a) in one go, which retires the bounds checks on the
// two child loads. The vacancy index i is carried in a bounded local so its
// store is checked once per level rather than per access.
func siftDownMin(a []uint64, key uint64) {
	n := len(a)
	i := 0
	for {
		l := 2*i + 1
		if uint(l) >= uint(n) {
			break
		}
		bi, bv := l, a[l]
		if r := l + 1; uint(r) < uint(n) {
			if rv := a[r]; rv < bv {
				bi, bv = r, rv
			}
		}
		if key <= bv {
			break
		}
		a[i] = bv
		i = bi
	}
	a[i] = key
}

// maxHeap is a binary max-heap of packed keys ordered descending. Used as the
// bounded-k "nearest so far" set: peek returns the furthest current member,
// pop evicts it when a closer candidate arrives.
type maxHeap []uint64

func (h maxHeap) len() int { return len(h) }

func (h *maxHeap) push(key uint64) {
	a := append(*h, key)
	i := len(a) - 1
	for i > 0 {
		parent := (i - 1) / 2
		pv := a[parent]
		if pv >= key {
			break
		}
		a[i] = pv
		i = parent
	}
	a[i] = key
	*h = a
}

// pop removes and returns the maximum key. Caller must ensure len > 0.
func (h *maxHeap) pop() uint64 {
	a := *h
	top := a[0]
	n := len(a) - 1
	if n > 0 {
		siftDownMax(a[:n], a[n])
	}
	*h = a[:n]
	return top
}

// replaceTop overwrites the root with key and sifts it down — the fused form of
// push-then-pop for the bounded-k set, which the search's inner loop reaches
// only when key is strictly closer than the current root. It does ONE descent
// instead of push's ascent plus pop's descent, and never touches the slice
// header. Caller must ensure len > 0 and key < the current root.
func (h maxHeap) replaceTop(key uint64) { siftDownMax(h, key) }

// siftDownMax places key at the root of the (otherwise valid) max-heap a,
// sifting it down to its ordered position. len(a) must be > 0. See siftDownMin
// for why the child tests are unsigned.
func siftDownMax(a []uint64, key uint64) {
	n := len(a)
	i := 0
	for {
		l := 2*i + 1
		if uint(l) >= uint(n) {
			break
		}
		bi, bv := l, a[l]
		if r := l + 1; uint(r) < uint(n) {
			if rv := a[r]; rv > bv {
				bi, bv = r, rv
			}
		}
		if key >= bv {
			break
		}
		a[i] = bv
		i = bi
	}
	a[i] = key
}

// peek returns the top of the heap without popping. Caller must check len() > 0.
func (h maxHeap) peek() uint64 { return h[0] }
