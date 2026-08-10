// SPDX-License-Identifier: Apache-2.0

package analysis

// FNV-1a (32-bit) constants. See https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
const (
	fnvOffset32 uint32 = 2166136261
	fnvPrime32  uint32 = 16777619
)

// TermID maps a stem to a 32-bit term id via FNV-1a over the stem's UTF-8
// bytes. It is a pure, deterministic function of the input: the same stem
// always yields the same id, on every process and every node. Partitioned
// nodes rely on this stability — they must agree on dim ids without sharing a
// dictionary.
func TermID(stem string) uint32 {
	h := fnvOffset32
	for i := 0; i < len(stem); i++ {
		h ^= uint32(stem[i])
		h *= fnvPrime32
	}
	return h
}

// TermIDMod is TermID folded into [0, width). A width of 0 means "no modulo":
// the full uint32 space is used and the result equals TermID(stem). Like
// TermID, it is a pure deterministic function so partitioned nodes agree.
func TermIDMod(stem string, width uint32) uint32 {
	id := TermID(stem)
	if width == 0 {
		return id
	}
	return id % width
}
