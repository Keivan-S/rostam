// SPDX-License-Identifier: Apache-2.0

package cache

// slabRef points at an entry in a shard's pages, packed into a single uint64
// so map[uint64]slabRef stores 16-byte (key, value) pairs — tight enough that
// every swisstable group fits more entries per cache line during matchH2
// probing.
//
// Layout (LSB at bit 0):
//
//	bits [0:32]  offset within the page (page size capped at 1 GiB)
//	bits [32:48] pageIdx (MaxPagesPerShard ≤ 65535)
//	bits [48:64] generation of the page object the ref was minted against
//
// The entry size is no longer stored: page.Read recovers it from the entry
// header on the fly. The header is read on every Get anyway, so dropping
// the field is free at the call site and shrinks the map value 12B → 8B.
//
// The generation gates the lock-free heap-mode ringbuf read path: an evicted
// heap page is retired by swapping in a fresh page object with a new
// generation, so a reader that resolves this ref, loads the current page
// object, and finds gen(page) != gen(ref) knows the entry was evicted and
// returns a miss WITHOUT reading the (possibly under-append) fresh page's
// bytes. See indexTable.get and shard.retirePageLocked.
type slabRef uint64

// maxPagesPerShardCap is the largest page count a shard may have: pageIdx
// occupies 16 bits, so page indices must fit a uint16. Config.Validate enforces
// this so makeSlabRef's uint16(pageIdx) cast below can never wrap.
const maxPagesPerShardCap = 1<<16 - 1 // 65535

func makeSlabRef(pageIdx, gen uint16, offset uint32) slabRef {
	return slabRef(uint64(gen))<<48 | slabRef(uint64(pageIdx))<<32 | slabRef(offset)
}

func (r slabRef) pageIdx() uint16 { return uint16(r >> 32) }
func (r slabRef) offset() uint32  { return uint32(r) }
func (r slabRef) gen() uint16     { return uint16(r >> 48) }
