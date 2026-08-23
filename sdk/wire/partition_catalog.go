// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"fmt"
)

// CatalogStore is the minimal persistence the partition catalog needs. embedded
// backs it with a reserved-shard KV (consensus-durable); tests use an in-memory
// map. Keys are catalog keys (see CatalogKey); values are encoded records.
type CatalogStore interface {
	GetCatalog(key []byte) ([]byte, bool)
	PutCatalog(key, val []byte) error
}

// CatalogRecord is the per-collection catalog entry. Version increments on every
// SetPartitions (i.e. create and each resplit) so cached readers detect change.
// Generation tracks the resplit generation; old 8-byte records decode as gen 0.
//
// Status/TargetP/TargetGen carry online-reshard state (added like Generation was,
// with backward-compatible short-record decode). Status 0 = Stable (the steady
// state); Status 1 = Resharding (a live repartition is in progress, dual-writing
// to the target gen). When Stable (Status==0 and both targets 0) the record is
// written as the legacy 12 bytes, byte-identical to pre-reshard on-disk data.
// When Resharding the full 32 bytes are written. Partitions/Generation remain the
// LIVE read gen; TargetP/TargetGen are the new gen being copied into.
//
// SourceP/SourceGen pin the OLD generation at reshard-begin so the dual-write
// keeps writing the old gen even AFTER the cutover flips the live Partitions/
// Generation to the new gen (deriving Old from the live record post-cutover would
// read the new gen and silently drop the old gen — the linearizable-catalog bug).
// They are written as the trailing 8 bytes (24..32) only while Resharding; a
// Stable record omits them (legacy 12-byte form) and a Resharding record written
// by pre-upgrade code (24 bytes, no Source) decodes them as 0 — see
// DecodeCatalogRecord.
type CatalogRecord struct {
	Partitions uint32
	Version    uint32
	Generation uint32
	Status     uint32
	TargetP    uint32
	TargetGen  uint32
	SourceP    uint32
	SourceGen  uint32
}

func CatalogKey(collection string) []byte {
	return append([]byte("__vcat__/"), vectorRouteKey(collection)...)
}

// EncodeCatalogRecord writes a little-endian record. A Stable record (Status==0
// and both targets 0) is written as the legacy 12 bytes
// [Partitions:4][Version:4][Generation:4], byte-identical to pre-reshard data, so
// existing readers and on-disk records are unaffected. A Resharding record is
// written as 32 bytes, appending
// [Status:4][TargetP:4][TargetGen:4][SourceP:4][SourceGen:4]. (SourceP/SourceGen
// pin the old gen so the dual-write survives the cutover flip.)
func EncodeCatalogRecord(r CatalogRecord) []byte {
	stable := r.Status == 0 && r.TargetP == 0 && r.TargetGen == 0
	size := 12
	if !stable {
		size = 32
	}
	b := make([]byte, size)
	binary.LittleEndian.PutUint32(b[0:4], r.Partitions)
	binary.LittleEndian.PutUint32(b[4:8], r.Version)
	binary.LittleEndian.PutUint32(b[8:12], r.Generation)
	if !stable {
		binary.LittleEndian.PutUint32(b[12:16], r.Status)
		binary.LittleEndian.PutUint32(b[16:20], r.TargetP)
		binary.LittleEndian.PutUint32(b[20:24], r.TargetGen)
		binary.LittleEndian.PutUint32(b[24:28], r.SourceP)
		binary.LittleEndian.PutUint32(b[28:32], r.SourceGen)
	}
	return b
}

// DecodeCatalogRecord decodes an 8-, 12-, 24-, or 32-byte record (and tolerates
// any truncated length in between). Records shorter than 8 bytes are invalid.
// 8-byte records (written by older code) decode with Generation=0; records of 12
// bytes (the legacy Stable form) decode with Status=0 and zero targets; 24-byte
// records (a Resharding record written by pre-source-field code) decode with
// SourceP=SourceGen=0 (the caller degrades to today's collapse-at-cutover
// behavior); 32-byte records carry the full Source pin. The reshard fields are
// read only when present, so short records never panic.
func DecodeCatalogRecord(b []byte) (CatalogRecord, bool) {
	if len(b) < 8 {
		return CatalogRecord{}, false
	}
	rec := CatalogRecord{
		Partitions: binary.LittleEndian.Uint32(b[0:4]),
		Version:    binary.LittleEndian.Uint32(b[4:8]),
	}
	if len(b) >= 12 {
		rec.Generation = binary.LittleEndian.Uint32(b[8:12])
	}
	if len(b) >= 16 {
		rec.Status = binary.LittleEndian.Uint32(b[12:16])
	}
	if len(b) >= 20 {
		rec.TargetP = binary.LittleEndian.Uint32(b[16:20])
	}
	if len(b) >= 24 {
		rec.TargetGen = binary.LittleEndian.Uint32(b[20:24])
	}
	if len(b) >= 28 {
		rec.SourceP = binary.LittleEndian.Uint32(b[24:28])
	}
	if len(b) >= 32 {
		rec.SourceGen = binary.LittleEndian.Uint32(b[28:32])
	}
	return rec, true
}

// Catalog is a pure read-through reader/writer for per-collection partition
// counts. Every call delegates directly to the underlying CatalogStore (which
// handles its own concurrency); there is no in-process cache or mutex here.
// The Version field in CatalogRecord exists to enable a future cache-on-version
// optimisation without changing the on-disk format.
type Catalog struct {
	store CatalogStore
}

func NewCatalog(store CatalogStore) *Catalog {
	return &Catalog{store: store}
}

// PartitionsGen returns the partition count, generation, and whether a catalog
// entry exists for the collection. Unknown -> (1, 0, false).
func (c *Catalog) PartitionsGen(collection string) (int, uint32, bool) {
	raw, ok := c.store.GetCatalog(CatalogKey(collection))
	if !ok {
		return 1, 0, false
	}
	rec, ok := DecodeCatalogRecord(raw)
	if !ok || rec.Partitions == 0 {
		return 1, 0, false
	}
	return int(rec.Partitions), rec.Generation, true
}

// SetPartitionsGen writes a new partition count and generation, bumping the
// version past whatever is currently stored. Used by create + resplit.
func (c *Catalog) SetPartitionsGen(collection string, p int, gen uint32) error {
	if p <= 0 {
		return fmt.Errorf("partition count must be positive, got %d", p)
	}
	prev := uint32(0)
	var status, targetP, targetGen, sourceP, sourceGen uint32
	if raw, ok := c.store.GetCatalog(CatalogKey(collection)); ok {
		if rec, ok := DecodeCatalogRecord(raw); ok {
			prev = rec.Version
			// Preserve any in-progress reshard state across a live-gen flip
			// (cutover sets the new read gen while status stays Resharding). The
			// Source pin in particular MUST survive the flip — that is what keeps
			// the dual-write hitting the old gen after the live gen becomes the new.
			status, targetP, targetGen = rec.Status, rec.TargetP, rec.TargetGen
			sourceP, sourceGen = rec.SourceP, rec.SourceGen
		}
	}
	rec := CatalogRecord{
		Partitions: uint32(p),
		Version:    prev + 1,
		Generation: gen,
		Status:     status,
		TargetP:    targetP,
		TargetGen:  targetGen,
		SourceP:    sourceP,
		SourceGen:  sourceGen,
	}
	return c.store.PutCatalog(CatalogKey(collection), EncodeCatalogRecord(rec))
}

// ReshardGen returns the collection's online-reshard state: status (0=Stable,
// 1=Resharding), the target partition count/generation, the SOURCE (old)
// partition count/generation pinned at reshard-begin, and ok=true iff a reshard
// is in progress (Status!=0). A Stable / missing collection reports zeros and
// false. A Resharding record written before the Source fields existed reports
// sourceP=sourceGen=0 (the caller degrades to collapse-at-cutover).
func (c *Catalog) ReshardGen(collection string) (status uint32, targetP uint32, targetGen uint32, sourceP uint32, sourceGen uint32, ok bool) {
	raw, found := c.store.GetCatalog(CatalogKey(collection))
	if !found {
		return 0, 0, 0, 0, 0, false
	}
	rec, decoded := DecodeCatalogRecord(raw)
	if !decoded || rec.Status == 0 {
		return 0, 0, 0, 0, 0, false
	}
	return rec.Status, rec.TargetP, rec.TargetGen, rec.SourceP, rec.SourceGen, true
}

// SetReshardGen records the collection's online-reshard state without changing
// the live partition count/generation. status==0 clears the reshard fields,
// returning the record to the legacy 12-byte Stable form (byte-identical to
// pre-reshard data). sourceP/sourceGen pin the old gen so the dual-write keeps
// hitting it after the cutover flips the live gen. The collection must already
// have a catalog entry (be partitioned); resharding an unpartitioned collection
// is meaningless. The Version is bumped so cached readers observe the change.
func (c *Catalog) SetReshardGen(collection string, status uint32, targetP uint32, targetGen uint32, sourceP uint32, sourceGen uint32) error {
	raw, ok := c.store.GetCatalog(CatalogKey(collection))
	if !ok {
		return fmt.Errorf("catalog: SetReshardGen: collection %q has no catalog entry", collection)
	}
	rec, ok := DecodeCatalogRecord(raw)
	if !ok || rec.Partitions == 0 {
		return fmt.Errorf("catalog: SetReshardGen: collection %q has no valid catalog entry", collection)
	}
	rec.Version++
	if status == 0 {
		rec.Status, rec.TargetP, rec.TargetGen = 0, 0, 0
		rec.SourceP, rec.SourceGen = 0, 0
	} else {
		rec.Status, rec.TargetP, rec.TargetGen = status, targetP, targetGen
		rec.SourceP, rec.SourceGen = sourceP, sourceGen
	}
	return c.store.PutCatalog(CatalogKey(collection), EncodeCatalogRecord(rec))
}

// Partitions returns the partition count for a collection and whether a catalog
// entry exists. Unknown (no entry) -> (1, false): single-partition default.
// Generation is ignored; use PartitionsGen for gen-aware reads.
func (c *Catalog) Partitions(collection string) (int, bool) {
	p, _, ok := c.PartitionsGen(collection)
	return p, ok
}

// SetPartitions writes a new partition count with generation 0, bumping the
// version. Used by create + legacy callers. Use SetPartitionsGen for resplit.
func (c *Catalog) SetPartitions(collection string, p int) error {
	return c.SetPartitionsGen(collection, p, 0)
}
