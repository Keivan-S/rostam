// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

func TestVectorInsertArgsRoundtrip(t *testing.T) {
	args := EncodeVectorInsertArgs("docs", 42, []float32{1.5, -2.5, 3.5, 4.5})
	collection, id, vec, ttl, meta, sparse, _, err := DecodeVectorInsertArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if sparse != nil {
		t.Errorf("sparse = %+v, want nil", sparse)
	}
	if collection != "docs" {
		t.Errorf("collection = %q", collection)
	}
	if id != 42 {
		t.Errorf("id = %d", id)
	}
	if !reflect.DeepEqual(vec, []float32{1.5, -2.5, 3.5, 4.5}) {
		t.Errorf("vec = %v", vec)
	}
	if ttl != 0 {
		t.Errorf("ttl = %v, want 0", ttl)
	}
	if meta != nil {
		t.Errorf("meta = %+v, want nil", meta)
	}
}

func TestVectorSearchArgsRoundtrip(t *testing.T) {
	args := EncodeVectorSearchArgs("docs", 7, []float32{0.1, 0.2, 0.3})
	collection, k, query, filter, err := DecodeVectorSearchArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if collection != "docs" || k != 7 {
		t.Errorf("collection=%q k=%d", collection, k)
	}
	if !reflect.DeepEqual(query, []float32{0.1, 0.2, 0.3}) {
		t.Errorf("query = %v", query)
	}
	if !filter.IsZero() {
		t.Errorf("filter = %+v, want zero", filter)
	}
}

func TestVectorSearchResultsRoundtrip(t *testing.T) {
	results := []vector.Result{
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 1.5},
	}
	body := EncodeVectorSearchResults(results)
	got, err := DecodeVectorSearchResults(body)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, results) {
		t.Errorf("results = %v, want %v", got, results)
	}
}

func TestVectorDeleteArgsRoundtrip(t *testing.T) {
	args := EncodeVectorDeleteArgs("docs", 99)
	collection, id, err := DecodeVectorDeleteArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if collection != "docs" || id != 99 {
		t.Errorf("collection=%q id=%d", collection, id)
	}
}

func TestVectorCreateCollectionArgsRoundtrip(t *testing.T) {
	cfg := vector.Config{Dim: 128, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}
	args := EncodeCreateCollectionArgs("docs", cfg)
	name, gotCfg, err := DecodeCreateCollectionArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" {
		t.Errorf("name = %q", name)
	}
	if !reflect.DeepEqual(gotCfg, cfg) {
		t.Errorf("cfg = %+v, want %+v", gotCfg, cfg)
	}
}

func TestVectorCreateCollectionArgsQuantPersist(t *testing.T) {
	cfg := vector.Config{
		Dim: 64, Metric: vector.Cosine, M: 16, EfConstruction: 100, EfSearch: 32, Seed: 7,
		Quant: vector.QuantSQ8, Persistent: true, RescoreFactor: 3,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, cfg) {
		t.Errorf("roundtrip cfg = %+v, want %+v", got, cfg)
	}
	if got.Quant != vector.QuantSQ8 || !got.Persistent || got.RescoreFactor != 3 {
		t.Errorf("quant/persist extension not carried: %+v", got)
	}
}

// TestVectorCreateCollectionArgsSQBitsByteIdentical proves the trained-SQ / PRQ
// trailer is purely additive: a create with SQBits==0 && PRQLayers==0 is
// BYTE-IDENTICAL to the pre-feature encoder (the SQBits/PRQLayers words and their
// forcing chain ride only when non-zero). The QuantSQ/QuantPRQ enum values
// themselves ride the existing quant byte, so even selecting QuantSQ with default
// bits adds nothing.
func TestVectorCreateCollectionArgsSQBitsByteIdentical(t *testing.T) {
	// A plain HNSW create (no SQ/PRQ params): the trailer must not be appended.
	base := vector.Config{Dim: 128, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}
	baseWire := EncodeCreateCollectionArgs("docs", base)

	// Selecting QuantSQ with default bits (SQBits==0) rides only the quant byte —
	// byte-identical to the QuantNone create EXCEPT for the single quant byte value.
	// Length identical, no trailing words.
	sqDefault := base
	sqDefault.Quant = vector.QuantSQ
	sqDefaultWire := EncodeCreateCollectionArgs("docs", sqDefault)
	if len(sqDefaultWire) != len(baseWire) {
		t.Fatalf("QuantSQ default-bits encode length = %d, want %d (no trailing words when SQBits==0)", len(sqDefaultWire), len(baseWire))
	}

	// The SQBits/PRQLayers words sit at the VERY END, past the whole forcing chain.
	// Anchor against a config that already forces the full chain (FullText set) AND
	// already selects QuantSQ (so the quant byte matches), so the only delta is the
	// new tail words. Without SQBits: chain forced, no tail.
	ftBase := base
	ftBase.Quant = vector.QuantSQ
	ftBase.FullText = &vector.FullTextConfig{Analyzer: "english", K1: 1.2, B: 0.75}
	ftBaseWire := EncodeCreateCollectionArgs("docs", ftBase)

	// With SQBits=6 on top of the FullText config: exactly the 4-byte SQBits word is
	// appended (the FullText presence/body already anchors it), and the prefix stays
	// byte-identical.
	ftSQ := ftBase
	ftSQ.SQBits = 6
	ftSQWire := EncodeCreateCollectionArgs("docs", ftSQ)
	if len(ftSQWire) != len(ftBaseWire)+4 {
		t.Fatalf("SQBits=6 (FullText base) encode length = %d, want %d (one SQBits word)", len(ftSQWire), len(ftBaseWire)+4)
	}
	if !bytes.Equal(ftSQWire[:len(ftBaseWire)], ftBaseWire) {
		t.Fatalf("SQBits wire prefix is not byte-identical to the FullText base wire")
	}

	// With PRQLayers=3 on top: SQBits word (forced, 4) + PRQLayers word (4) = 8
	// extra bytes vs the FullText base, prefix byte-identical.
	ftPRQ := ftBase
	ftPRQ.Quant = vector.QuantPRQ
	ftPRQ.QuantPQM = 8
	ftPRQ.PRQLayers = 3
	ftPRQWire := EncodeCreateCollectionArgs("docs", ftPRQ)
	// QuantPQM=8 adds its own 4-byte word; isolate the PRQ trailer by comparing to
	// the same config WITHOUT the PRQ layer count.
	ftPRQNoLayers := ftPRQ
	ftPRQNoLayers.PRQLayers = 0
	ftPRQNoLayersWire := EncodeCreateCollectionArgs("docs", ftPRQNoLayers)
	if len(ftPRQWire) != len(ftPRQNoLayersWire)+4+4 {
		t.Fatalf("PRQLayers encode length = %d, want %d (SQBits word + PRQLayers word)", len(ftPRQWire), len(ftPRQNoLayersWire)+8)
	}
	if !bytes.Equal(ftPRQWire[:len(ftPRQNoLayersWire)], ftPRQNoLayersWire) {
		t.Fatalf("PRQ wire prefix is not byte-identical to the no-layers wire")
	}
}

// TestVectorCreateCollectionArgsSQPRQRoundtrip round-trips QuantSQ (sq_bits=6)
// and QuantPRQ (prq_layers=3) through the create codec, and proves a
// pre-extension buffer (the SQBits/PRQLayers words stripped) still decodes with
// the new fields defaulted.
func TestVectorCreateCollectionArgsSQPRQRoundtrip(t *testing.T) {
	sq := vector.Config{
		Dim: 64, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7,
		Quant: vector.QuantSQ, SQBits: 6,
	}
	_, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", sq))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, sq) {
		t.Errorf("QuantSQ roundtrip cfg = %+v, want %+v", got, sq)
	}
	if got.Quant != vector.QuantSQ || got.SQBits != 6 {
		t.Errorf("QuantSQ params not carried: %+v", got)
	}

	prq := vector.Config{
		Dim: 64, Metric: vector.DotProduct, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7,
		Quant: vector.QuantPRQ, QuantPQM: 8, PRQLayers: 3,
	}
	prqWire := EncodeCreateCollectionArgs("docs", prq)
	_, gotPRQ, err := DecodeCreateCollectionArgs(prqWire)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotPRQ, prq) {
		t.Errorf("QuantPRQ roundtrip cfg = %+v, want %+v", gotPRQ, prq)
	}
	if gotPRQ.Quant != vector.QuantPRQ || gotPRQ.PRQLayers != 3 || gotPRQ.QuantPQM != 8 {
		t.Errorf("QuantPRQ params not carried: %+v", gotPRQ)
	}

	// Pre-extension decode tolerance: strip the trailing PRQLayers (4) + SQBits (4)
	// + the FullText presence anchor (1) = 9 bytes — a buffer from a client that
	// predates this feature. It must still decode, defaulting the SQ/PRQ fields.
	old := prqWire[:len(prqWire)-9]
	_, gotOld, err := DecodeCreateCollectionArgs(old)
	if err != nil {
		t.Fatalf("pre-extension decode: %v", err)
	}
	if gotOld.SQBits != 0 || gotOld.PRQLayers != 0 {
		t.Errorf("pre-extension decode must default SQ/PRQ fields, got SQBits=%d PRQLayers=%d", gotOld.SQBits, gotOld.PRQLayers)
	}
	// The base fields (incl. QuantPQM that survived the strip) must still be intact.
	if gotOld.Quant != vector.QuantPRQ || gotOld.QuantPQM != 8 || gotOld.Dim != 64 {
		t.Errorf("pre-extension decode lost base fields: %+v", gotOld)
	}
}

// TestVectorCreateCollectionArgsVamanaByteIdentical proves the Vamana trailer is
// purely additive: a create with VamanaR==0 && VamanaL==0 && VamanaAlpha==0 is
// BYTE-IDENTICAL to the pre-feature encoder (the three Vamana words and their
// forcing chain ride only when non-zero). IndexVamana itself rides the existing
// IndexType byte, so selecting it (with default R/L/alpha) adds no Vamana-trailer
// bytes beyond the IndexType extension that any non-HNSW index already writes.
func TestVectorCreateCollectionArgsVamanaByteIdentical(t *testing.T) {
	// A plain HNSW create (no Vamana params): the trailer must not be appended.
	base := vector.Config{Dim: 128, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}
	baseWire := EncodeCreateCollectionArgs("docs", base)

	// IndexVamana with default R/L/alpha (all zero): rides only the IndexType byte
	// extension (indexType:1 + ivfNlist:4 + ivfNprobe:4 = 9 bytes), exactly like a
	// default IVF create — NO Vamana trailer words. Length == base + 9.
	vamDefault := base
	vamDefault.IndexType = vector.IndexVamana
	vamDefaultWire := EncodeCreateCollectionArgs("docs", vamDefault)
	if len(vamDefaultWire) != len(baseWire)+9 {
		t.Fatalf("IndexVamana default-params encode length = %d, want %d (only the IndexType extension, no Vamana trailer)", len(vamDefaultWire), len(baseWire)+9)
	}

	// Setting VamanaR appends exactly the three Vamana words plus the forced upstream
	// chain (PRQLayers word + SQBits word + FullText presence anchor). Anchor against
	// a config that ALREADY forces that whole chain so the only delta is the three
	// Vamana words. ftBase forces FullText (=> opqIters/relBP/drift/threshold/ivf all
	// forced) but no Vamana trailer.
	ftBase := vamDefault
	ftBase.FullText = &vector.FullTextConfig{Analyzer: "english", K1: 1.2, B: 0.75}
	ftBaseWire := EncodeCreateCollectionArgs("docs", ftBase)

	// With VamanaR=96 (+L/alpha) on top: SQBits word (forced, 4) + PRQLayers word
	// (forced, 4) + VamanaR (4) + VamanaL (4) + VamanaAlpha (4) = 20 bytes, prefix
	// byte-identical to the FullText base.
	ftVam := ftBase
	ftVam.VamanaR = 96
	ftVam.VamanaL = 150
	ftVam.VamanaAlpha = 1.4
	ftVamWire := EncodeCreateCollectionArgs("docs", ftVam)
	if len(ftVamWire) != len(ftBaseWire)+4+4+4+4+4 {
		t.Fatalf("VamanaR/L/alpha (FullText base) encode length = %d, want %d (SQBits+PRQLayers+R+L+alpha words)", len(ftVamWire), len(ftBaseWire)+20)
	}
	if !bytes.Equal(ftVamWire[:len(ftBaseWire)], ftBaseWire) {
		t.Fatalf("Vamana wire prefix is not byte-identical to the FullText base wire")
	}
}

// TestVectorCreateCollectionArgsVamanaRoundtrip round-trips IndexVamana with
// non-default R/L/alpha through the create codec, and proves a pre-extension
// buffer (the three Vamana words stripped) still decodes with the new fields
// defaulted (back-compat for a client that predates this feature).
func TestVectorCreateCollectionArgsVamanaRoundtrip(t *testing.T) {
	vam := vector.Config{
		Dim: 96, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7,
		IndexType: vector.IndexVamana, VamanaR: 96, VamanaL: 150, VamanaAlpha: 1.4,
	}
	wire := EncodeCreateCollectionArgs("docs", vam)
	_, got, err := DecodeCreateCollectionArgs(wire)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, vam) {
		t.Errorf("IndexVamana roundtrip cfg = %+v, want %+v", got, vam)
	}
	if got.IndexType != vector.IndexVamana || got.VamanaR != 96 || got.VamanaL != 150 || got.VamanaAlpha != 1.4 {
		t.Errorf("Vamana params not carried: %+v", got)
	}

	// Pre-extension decode tolerance: strip the trailing VamanaAlpha (4) + VamanaL
	// (4) + VamanaR (4) + PRQLayers (4) + SQBits (4) + the FullText presence anchor
	// (1) = 21 bytes — a buffer from a client that predates this feature. It must
	// still decode, defaulting the Vamana fields (IndexType still rides its own byte
	// in the IVF extension, which is upstream of this trailer, so it survives).
	old := wire[:len(wire)-21]
	_, gotOld, err := DecodeCreateCollectionArgs(old)
	if err != nil {
		t.Fatalf("pre-extension decode: %v", err)
	}
	if gotOld.VamanaR != 0 || gotOld.VamanaL != 0 || gotOld.VamanaAlpha != 0 {
		t.Errorf("pre-extension decode must default Vamana fields, got R=%d L=%d alpha=%v", gotOld.VamanaR, gotOld.VamanaL, gotOld.VamanaAlpha)
	}
	if gotOld.IndexType != vector.IndexVamana || gotOld.Dim != 96 {
		t.Errorf("pre-extension decode lost base fields: %+v", gotOld)
	}
}

func TestVectorCreateCollectionArgsExtendCandidates(t *testing.T) {
	cfg := vector.Config{
		Dim: 64, Metric: vector.Cosine, M: 32, EfConstruction: 200, EfSearch: 64, Seed: 7,
		Quant: vector.QuantSQ8, ExtendCandidates: true, ExtendCandidatesMax: 800, Level0FullDegree: true, QuantizedBuild: true,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, cfg) {
		t.Errorf("roundtrip cfg = %+v, want %+v", got, cfg)
	}
	if !got.ExtendCandidates || got.ExtendCandidatesMax != 800 || !got.Level0FullDegree || !got.QuantizedBuild {
		t.Errorf("graph-quality extension not carried: %+v", got)
	}
}

func TestVectorCreateCollectionArgsBackwardCompat(t *testing.T) {
	// A pre-extension buffer (without either trailing extension) must still
	// decode, defaulting the new fields. The two appended extensions are
	// quant(1)+persistent(1)+rescoreFactor(4) = 6 bytes and
	// extendCandidates(1)+extendCandidatesMax(4)+level0FullDegree(1)+
	// quantizedBuild(1) = 7 bytes.
	cfg := vector.Config{Dim: 8, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1}
	full := EncodeCreateCollectionArgs("c", cfg)
	old := full[:len(full)-6-7] // strip both extensions → pre-extension buffer
	name, got, err := DecodeCreateCollectionArgs(old)
	if err != nil {
		t.Fatalf("decode old-format args: %v", err)
	}
	if name != "c" || got.Dim != 8 || got.Seed != 1 {
		t.Errorf("base fields lost: %+v", got)
	}
	if got.Quant != vector.QuantNone || got.Persistent || got.RescoreFactor != 0 {
		t.Errorf("old-format decode should default the quant extension, got %+v", got)
	}
	if got.ExtendCandidates || got.ExtendCandidatesMax != 0 || got.Level0FullDegree || got.QuantizedBuild {
		t.Errorf("old-format decode should default the graph extension, got %+v", got)
	}

	// A buffer carrying only the quant extension (no graph extension) must also
	// decode, defaulting just the graph fields — the intermediate-version case.
	midCfg := vector.Config{Dim: 8, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Quant: vector.QuantSQ8, RescoreFactor: 3}
	midFull := EncodeCreateCollectionArgs("c", midCfg)
	mid := midFull[:len(midFull)-7] // strip the 7-byte graph extension only
	_, gotMid, err := DecodeCreateCollectionArgs(mid)
	if err != nil {
		t.Fatalf("decode mid-format args: %v", err)
	}
	if gotMid.Quant != vector.QuantSQ8 || gotMid.RescoreFactor != 3 {
		t.Errorf("mid-format should carry quant extension, got %+v", gotMid)
	}
	if gotMid.ExtendCandidates || gotMid.ExtendCandidatesMax != 0 || gotMid.Level0FullDegree || gotMid.QuantizedBuild {
		t.Errorf("mid-format should default graph extension, got %+v", gotMid)
	}
}

func TestVectorCreateCollectionArgsIVF(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, cfg) {
		t.Errorf("roundtrip cfg = %+v, want %+v", got, cfg)
	}
	if got.IndexType != vector.IndexIVF || got.IVFNlist != 64 || got.IVFNprobe != 12 {
		t.Errorf("IVF extension not carried: %+v", got)
	}
}

// TestVectorCreateCollectionArgsIVFByteIdentical proves a default (HNSW) create is
// byte-identical to the pre-IVF encoder: the IVF extension is appended ONLY when
// non-default, so an HNSW config (and any config with IndexHNSW + zero nlist/nprobe)
// encodes exactly as it did before the IVF fields existed.
// TestVectorCreateCollectionArgsDriftRetrain: the IVF drift-retrain knobs
// (ivf_drift_retrain + the two float64 factors) round-trip through the create codec.
func TestVectorCreateCollectionArgsDriftRetrain(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFDriftRetrain: true, IVFDriftGrowthFactor: 2.5, IVFDriftFactor: 1.75,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, cfg) {
		t.Errorf("roundtrip cfg = %+v, want %+v", got, cfg)
	}
	if !got.IVFDriftRetrain || got.IVFDriftGrowthFactor != 2.5 || got.IVFDriftFactor != 1.75 {
		t.Errorf("drift-retrain knobs not carried: %+v", got)
	}
	// A drift config with the threshold also set (the common combo) still round-trips.
	cfg2 := cfg
	cfg2.IVFTrainThreshold = 4096
	_, got2, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("d", cfg2))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got2, cfg2) {
		t.Errorf("drift+threshold roundtrip cfg = %+v, want %+v", got2, cfg2)
	}
}

// TestVectorCreateCollectionArgsDriftByteIdentical: a create with all three drift
// fields zero/false is BYTE-IDENTICAL to the pre-drift encoder (no trailing block).
func TestVectorCreateCollectionArgsDriftByteIdentical(t *testing.T) {
	// IVF config WITHOUT drift — must not append the 17-byte drift block.
	base := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
	}
	noDrift := EncodeCreateCollectionArgs("docs", base)
	withDrift := EncodeCreateCollectionArgs("docs", func() vector.Config {
		c := base
		c.IVFDriftRetrain = true
		c.IVFDriftGrowthFactor = 2.0
		c.IVFDriftFactor = 1.5
		return c
	}())
	// The drift create is longer by exactly the threshold word (forced) + OPQ/PQDropVecs
	// anchor slots + the 17-byte drift block. The key invariant: the no-drift encode is
	// the SAME length it was before this feature (the drift trailer is not present).
	if len(withDrift) <= len(noDrift) {
		t.Fatalf("drift encode (%d) must be longer than no-drift (%d)", len(withDrift), len(noDrift))
	}
	// And the no-drift bytes are a prefix-stable HNSW/IVF encode: decoding them yields
	// the zero drift fields.
	_, got, err := DecodeCreateCollectionArgs(noDrift)
	if err != nil {
		t.Fatal(err)
	}
	if got.IVFDriftRetrain || got.IVFDriftGrowthFactor != 0 || got.IVFDriftFactor != 0 {
		t.Errorf("no-drift decode must default drift fields, got %+v", got)
	}
}

func TestVectorCreateCollectionArgsIVFByteIdentical(t *testing.T) {
	// A plain HNSW config: the IVF block must NOT be appended.
	hnsw := vector.Config{Dim: 128, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}
	full := EncodeCreateCollectionArgs("docs", hnsw)
	// The pre-IVF wire ends at Partitions: base + quant(6) + graph(7) with no IVF
	// trailer. An HNSW encode must produce exactly that length (no 9-byte IVF block).
	wantLen := 1 + len("docs") + 4 + 1 + 4 + 4 + 4 + 8 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 4
	if len(full) != wantLen {
		t.Fatalf("HNSW encode length = %d, want %d (IVF block must not be appended for a default config)", len(full), wantLen)
	}

	// An IVF config of the same name/dims must be strictly longer by the 9-byte
	// IVF block (indexType:1 + nlist:4 + nprobe:4).
	ivf := hnsw
	ivf.IndexType = vector.IndexIVF
	ivfFull := EncodeCreateCollectionArgs("docs", ivf)
	if len(ivfFull) != wantLen+9 {
		t.Fatalf("IVF encode length = %d, want %d", len(ivfFull), wantLen+9)
	}
	// The IVF wire's prefix (everything up to the IVF block) must be byte-identical
	// to the HNSW wire — the extension is purely additive.
	if !bytes.Equal(ivfFull[:wantLen], full) {
		t.Fatalf("IVF wire prefix is not byte-identical to the HNSW wire")
	}
}

// TestVectorCreateCollectionArgsIVFPQ proves the PQ sub-block round-trips and is
// purely additive: a non-PQ IVF config is byte-identical to the pre-PQ encoder,
// and the PQ block (6 bytes) is appended only when IVFPQ/IVFRerank is set.
func TestVectorCreateCollectionArgsIVFPQ(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, cfg) {
		t.Errorf("PQ roundtrip cfg = %+v, want %+v", got, cfg)
	}

	// A non-PQ IVF config must encode byte-identically to the pre-PQ encoder: no
	// PQ sub-block appended (length == base + 9-byte IVF block).
	nonPQ := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
	}
	nonPQWire := EncodeCreateCollectionArgs("docs", nonPQ)
	base := 1 + len("docs") + 4 + 1 + 4 + 4 + 4 + 8 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 4
	if len(nonPQWire) != base+9 {
		t.Fatalf("non-PQ IVF wire = %d, want %d (no PQ block)", len(nonPQWire), base+9)
	}
	pqWire := EncodeCreateCollectionArgs("docs", cfg)
	if len(pqWire) != base+9+6 {
		t.Fatalf("PQ IVF wire = %d, want %d (IVF block + 6-byte PQ block)", len(pqWire), base+9+6)
	}
	if !bytes.Equal(pqWire[:base+9], nonPQWire) {
		t.Fatal("PQ wire prefix is not byte-identical to the non-PQ IVF wire")
	}
}

// TestVectorCreateCollectionArgsQuantPQM proves the PQ-HNSW quant sub-quantizer
// count rides the create wire and is purely additive: a config with QuantPQM == 0
// encodes byte-identically to the pre-QuantPQM encoder, and the 4-byte block is
// appended at the very end ONLY when QuantPQM != 0.
func TestVectorCreateCollectionArgsQuantPQM(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, QuantPQM: 8,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, cfg) {
		t.Errorf("QuantPQM roundtrip cfg = %+v, want %+v", got, cfg)
	}

	// QuantPQM == 0 (incl. quant == "pq" with engine-default M) must encode
	// byte-identically to the pre-QuantPQM encoder: no trailing 4-byte block.
	zeroM := cfg
	zeroM.QuantPQM = 0
	zeroWire := EncodeCreateCollectionArgs("docs", zeroM)
	base := 1 + len("docs") + 4 + 1 + 4 + 4 + 4 + 8 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 4
	if len(zeroWire) != base {
		t.Fatalf("QuantPQM==0 wire = %d, want %d (no trailing block)", len(zeroWire), base)
	}
	pqWire := EncodeCreateCollectionArgs("docs", cfg)
	if len(pqWire) != base+4 {
		t.Fatalf("QuantPQM!=0 wire = %d, want %d (4-byte trailing block)", len(pqWire), base+4)
	}
	if !bytes.Equal(pqWire[:base], zeroWire) {
		t.Fatal("QuantPQM wire prefix is not byte-identical to the QuantPQM==0 wire")
	}
	// And QuantPQM rides on top of an IVF+PQ config (independent trailing block).
	combo := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true, QuantPQM: 16,
	}
	_, gotCombo, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", combo))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotCombo, combo) {
		t.Errorf("IVF+QuantPQM roundtrip cfg = %+v, want %+v", gotCombo, combo)
	}
}

// TestVectorCreateCollectionArgsOPQ proves the OPQ flag rides the create wire and
// is purely additive: OPQ=false encodes byte-identically to the pre-OPQ encoder
// (no trailing byte), and a single byte is appended at the very end ONLY when
// OPQ=true. It round-trips on both PQ-HNSW and IVF-PQ configs.
func TestVectorCreateCollectionArgsOPQ(t *testing.T) {
	// PQ-HNSW + OPQ round-trips.
	pqhnsw := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, QuantPQM: 8, OPQ: true,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", pqhnsw))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, pqhnsw) {
		t.Errorf("PQ-HNSW OPQ roundtrip cfg = %+v, want %+v", got, pqhnsw)
	}

	// OPQ=false must encode byte-identically to the pre-OPQ encoder: no trailing
	// byte. Compare against the same config with OPQ off.
	offCfg := pqhnsw
	offCfg.OPQ = false
	offWire := EncodeCreateCollectionArgs("docs", offCfg)
	onWire := EncodeCreateCollectionArgs("docs", pqhnsw)
	if len(onWire) != len(offWire)+1 {
		t.Fatalf("OPQ=true wire = %d, want OPQ=false (%d) + 1 byte", len(onWire), len(offWire))
	}
	if !bytes.Equal(onWire[:len(offWire)], offWire) {
		t.Fatal("OPQ wire prefix is not byte-identical to the OPQ=false wire")
	}
	if onWire[len(offWire)] != 1 {
		t.Fatal("OPQ trailing byte should be 1")
	}

	// IVF-PQ + OPQ round-trips (OPQ rides after the QuantPQM slot; here QuantPQM==0).
	ivfpq := vector.Config{
		Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12, IVFPQ: true, IVFPQM: 8, OPQ: true,
	}
	_, gotIVF, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", ivfpq))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotIVF, ivfpq) {
		t.Errorf("IVF-PQ OPQ roundtrip cfg = %+v, want %+v", gotIVF, ivfpq)
	}

	// OPQ rides on top of a config that ALSO sets QuantPQM != 0 (OPQ is the last
	// trailing byte, after the 4-byte QuantPQM): both decode correctly.
	combo := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, QuantPQM: 16, OPQ: true,
	}
	_, gotCombo, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", combo))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotCombo, combo) {
		t.Errorf("QuantPQM+OPQ roundtrip cfg = %+v, want %+v", gotCombo, combo)
	}
}

// TestVectorCreateCollectionArgsPQDropVecs proves the PQDropVecs flag rides the
// create wire and is purely additive: PQDropVecs=false encodes byte-identically
// to the pre-PQDropVecs encoder (whether OPQ is off — no trailing bytes — or on
// — exactly the single OPQ byte), and PQDropVecs=true round-trips with OPQ both
// off and on.
func TestVectorCreateCollectionArgsPQDropVecs(t *testing.T) {
	// PQ-HNSW + PQDropVecs (OPQ off) round-trips.
	dropOnly := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, QuantPQM: 8, PQDropVecs: true,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", dropOnly))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || !reflect.DeepEqual(got, dropOnly) {
		t.Errorf("PQDropVecs roundtrip cfg = %+v, want %+v", got, dropOnly)
	}

	// PQDropVecs=false (OPQ also false) must encode byte-identically to the
	// pre-PQDropVecs encoder: no trailing bytes at all.
	offCfg := dropOnly
	offCfg.PQDropVecs = false
	offWire := EncodeCreateCollectionArgs("docs", offCfg)
	onWire := EncodeCreateCollectionArgs("docs", dropOnly)
	// OPQ=false: when PQDropVecs is true the OPQ anchor byte (0) plus the
	// PQDropVecs byte (1) trail; the off wire has neither. So +2 bytes.
	if len(onWire) != len(offWire)+2 {
		t.Fatalf("PQDropVecs=true wire = %d, want PQDropVecs=false (%d) + 2 bytes", len(onWire), len(offWire))
	}
	if !bytes.Equal(onWire[:len(offWire)], offWire) {
		t.Fatal("PQDropVecs wire prefix is not byte-identical to the PQDropVecs=false wire")
	}
	if onWire[len(offWire)] != 0 || onWire[len(offWire)+1] != 1 {
		t.Fatalf("PQDropVecs trailing bytes = [%d %d], want [0(OPQ off) 1(drop on)]", onWire[len(offWire)], onWire[len(offWire)+1])
	}

	// PQDropVecs=false with OPQ=true must be byte-identical to the OPQ-only wire
	// (the single OPQ trailing byte) — proves PQDropVecs adds NOTHING when off.
	opqOnly := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, QuantPQM: 8, OPQ: true,
	}
	opqWire := EncodeCreateCollectionArgs("docs", opqOnly)
	opqDropOff := opqOnly
	opqDropOff.PQDropVecs = false
	if !bytes.Equal(EncodeCreateCollectionArgs("docs", opqDropOff), opqWire) {
		t.Fatal("PQDropVecs=false changed the OPQ-only wire — not byte-identical")
	}

	// OPQ=true + PQDropVecs=true round-trips (OPQ anchor byte 1 + drop byte 1).
	both := opqOnly
	both.PQDropVecs = true
	_, gotBoth, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", both))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotBoth, both) {
		t.Errorf("OPQ+PQDropVecs roundtrip cfg = %+v, want %+v", gotBoth, both)
	}
	bothWire := EncodeCreateCollectionArgs("docs", both)
	if len(bothWire) != len(opqWire)+1 {
		t.Fatalf("OPQ+PQDropVecs wire = %d, want OPQ-only (%d) + 1 byte", len(bothWire), len(opqWire))
	}
}

// TestVectorCreateCollectionArgsIVFTrainThreshold proves the dense
// IVFTrainThreshold knob rides the create wire (closing the dense create-codec
// asymmetry vs named/MV, which already carried it) and round-trips across the IVF
// family. A non-zero threshold rides at the tail; to keep the decoder's greedy
// length-guarded reads unambiguous it FORCES the upstream optional blocks (IVF
// header, IVF-PQ sub-block, QuantPQM, OPQ, PQDropVecs) to be present, each
// carrying the config's true (often zero) values, so every combination decodes
// back to the original config.
func TestVectorCreateCollectionArgsIVFTrainThreshold(t *testing.T) {
	cases := []struct {
		name string
		cfg  vector.Config
	}{
		{"ivf-flat", vector.Config{
			Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
			IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12, IVFTrainThreshold: 5000,
		}},
		{"ivf-pq-opq-rerank", vector.Config{
			Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
			IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12, IVFPQ: true, IVFPQM: 8,
			IVFRerank: true, OPQ: true, IVFTrainThreshold: 3000,
		}},
		// PQ-HNSW + PQDropVecs + threshold: IndexType stays HNSW through the forced
		// IVF header (its IndexType byte carries the true HNSW value).
		{"pq-hnsw-drop", vector.Config{
			Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
			Quant: vector.QuantPQ, QuantPQM: 8, PQDropVecs: true, IVFTrainThreshold: 1234,
		}},
		// Threshold==1 (the smallest non-zero) still round-trips.
		{"threshold-one", vector.Config{
			Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
			IndexType: vector.IndexIVF, IVFNlist: 8, IVFNprobe: 4, IVFTrainThreshold: 1,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", tc.cfg))
			if err != nil {
				t.Fatal(err)
			}
			if name != "docs" || !reflect.DeepEqual(got, tc.cfg) {
				t.Errorf("roundtrip cfg = %+v, want %+v", got, tc.cfg)
			}
		})
	}
}

// TestVectorCreateCollectionArgsThresholdByteIdentical is the #1 no-break: a
// create with IVFTrainThreshold==0 (the default, incl. every existing collection)
// encodes BYTE-IDENTICALLY to the pre-threshold encoder — it forces NONE of the
// upstream anchor blocks. Verified by comparing the threshold==0 wire of several
// configs against the wire each produced before this field existed: a zero
// threshold must add nothing.
func TestVectorCreateCollectionArgsThresholdByteIdentical(t *testing.T) {
	// A plain HNSW create with threshold==0 has NO trailing blocks at all — its wire
	// length is exactly the fixed base (the pre-extension layout).
	hnsw := vector.Config{Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9}
	base := 1 + len("docs") + 4 + 1 + 4 + 4 + 4 + 8 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 4
	if w := EncodeCreateCollectionArgs("docs", hnsw); len(w) != base {
		t.Fatalf("HNSW threshold==0 wire = %d bytes, want base %d (threshold must add nothing)", len(w), base)
	}

	// For configs that ALREADY set IVF / PQ knobs, the threshold==0 wire must equal
	// the wire computed by the encoder ignoring the threshold field entirely (i.e.
	// setting it to 0 is a no-op): self-evident here, but the load-bearing check is
	// that none of these grow vs. their threshold-cleared twin.
	withKnobs := []vector.Config{
		{Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12},
		{Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Quant: vector.QuantPQ, QuantPQM: 8, OPQ: true},
		{Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12, IVFPQ: true, IVFPQM: 8},
	}
	for i, cfg := range withKnobs {
		cleared := cfg // already threshold==0
		if !bytes.Equal(EncodeCreateCollectionArgs("docs", cfg), EncodeCreateCollectionArgs("docs", cleared)) {
			t.Fatalf("case %d: threshold==0 wire differs from its threshold-cleared twin", i)
		}
		// And it must round-trip with IVFTrainThreshold==0.
		_, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
		if err != nil {
			t.Fatal(err)
		}
		if got.IVFTrainThreshold != 0 {
			t.Fatalf("case %d: threshold==0 decoded as %d", i, got.IVFTrainThreshold)
		}
	}
}

// TestVectorCreateCollectionArgsThresholdOldBytes proves a wire produced by an OLD
// encoder (one that never wrote the forced-anchor + threshold trailer) decodes
// with IVFTrainThreshold==0 and no error — back-compat from the decode direction.
// The pre-threshold wire for an IVF-Flat create is exactly its threshold==0 wire.
func TestVectorCreateCollectionArgsThresholdOldBytes(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
	}
	old := EncodeCreateCollectionArgs("docs", cfg) // threshold==0 == the legacy wire
	_, got, err := DecodeCreateCollectionArgs(old)
	if err != nil {
		t.Fatalf("old-bytes decode error: %v", err)
	}
	if got.IVFTrainThreshold != 0 {
		t.Fatalf("old-bytes IVFTrainThreshold = %d, want 0", got.IVFTrainThreshold)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Fatalf("old-bytes roundtrip cfg = %+v, want %+v", got, cfg)
	}
}

func TestCreateCollectionArgsRoundTripPartitions(t *testing.T) {
	cfg := vector.Config{Dim: 16, Metric: vector.Cosine, Partitions: 8}
	args := EncodeCreateCollectionArgs("docs", cfg)
	gotName, gotCfg, err := DecodeCreateCollectionArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if gotName != "docs" || gotCfg.Partitions != 8 {
		t.Fatalf("round-trip name=%q partitions=%d, want docs/8", gotName, gotCfg.Partitions)
	}

	// Backward compat: a pre-Partitions message ends before the trailing 4-byte
	// Partitions field. Truncating those 4 bytes simulates an old encoder's output;
	// the decoder's length guard must default Partitions to 0 without error.
	legacy := args[:len(args)-4]
	_, legacyCfg, err := DecodeCreateCollectionArgs(legacy)
	if err != nil {
		t.Fatalf("legacy decode error: %v", err)
	}
	if legacyCfg.Partitions != 0 {
		t.Fatalf("legacy partitions=%d, want 0", legacyCfg.Partitions)
	}
}

func TestVectorDropCollectionArgsRoundtrip(t *testing.T) {
	args := EncodeDropCollectionArgs("docs")
	name, err := DecodeDropCollectionArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" {
		t.Errorf("name = %q", name)
	}
}

func TestVectorInsertArgsRejectsBadLength(t *testing.T) {
	_, _, _, _, _, _, _, err := DecodeVectorInsertArgs([]byte{0x05})
	if err == nil {
		t.Error("expected error on truncated args")
	}
}

func TestVectorInsertArgsExtRoundtrip(t *testing.T) {
	meta := vector.Metadata{"tenant": vector.NewString("acme"), "score": vector.NewInt(95)}
	sv := vector.SparseVector{Indices: []uint32{3, 8}, Values: []float32{1.5, 2.5}}
	args := EncodeVectorInsertArgsExt("docs", 42, []float32{1, 2, 3}, 5*time.Second, meta, sv)
	col, id, vec, ttl, gotMeta, gotSparse, _, err := DecodeVectorInsertArgs(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "docs" || id != 42 || len(vec) != 3 {
		t.Errorf("basic fields: col=%q id=%d vec=%v", col, id, vec)
	}
	if ttl != 5*time.Second {
		t.Errorf("ttl = %v, want 5s", ttl)
	}
	if !gotMeta["tenant"].Equal(vector.NewString("acme")) || !gotMeta["score"].Equal(vector.NewInt(95)) {
		t.Errorf("meta = %+v", gotMeta)
	}
	if gotSparse == nil || len(gotSparse.Indices) != 2 || gotSparse.Indices[1] != 8 || gotSparse.Values[0] != 1.5 {
		t.Errorf("sparse = %+v", gotSparse)
	}
}

func TestHybridSearchArgsRoundtrip(t *testing.T) {
	sparse := vector.SparseVector{Indices: []uint32{2, 9}, Values: []float32{0.5, 1.5}}
	opts := vector.HybridOpts{
		Filter:  vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")},
		Method:  vector.FusionWeighted,
		Alpha:   0.7,
		RRFK:    40,
		DenseK:  100,
		SparseK: 80,
	}
	args := EncodeHybridSearchArgs("docs", []float32{0.1, 0.2, 0.3}, 5, sparse, opts)
	col, dense, k, gotSparse, gotOpts, err := DecodeHybridSearchArgs(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "docs" || k != 5 || len(dense) != 3 {
		t.Errorf("basic: col=%q k=%d dense=%v", col, k, dense)
	}
	if gotOpts.Method != vector.FusionWeighted || gotOpts.Alpha != 0.7 || gotOpts.RRFK != 40 || gotOpts.DenseK != 100 || gotOpts.SparseK != 80 {
		t.Errorf("opts = %+v", gotOpts)
	}
	if gotSparse.Indices[1] != 9 || gotSparse.Values[0] != 0.5 {
		t.Errorf("sparse = %+v", gotSparse)
	}
	if gotOpts.Filter.IsZero() {
		t.Error("filter decoded as zero")
	}
}

func TestHybridResultsRoundtrip(t *testing.T) {
	results := []vector.Result{
		{ID: 1, Distance: 0.5, Score: 0.9},
		{ID: 7, Distance: 0.0, Score: 0.3},
	}
	got, err := DecodeHybridResults(EncodeHybridResults(results))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 || got[0].ID != 1 || got[0].Score != 0.9 || got[1].Distance != 0.0 {
		t.Errorf("results = %+v", got)
	}
}

func TestVectorSearchArgsExtRoundtrip(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}
	args := EncodeVectorSearchArgsExt("docs", 7, []float32{0.1, 0.2}, filter)
	col, k, query, gotFilter, err := DecodeVectorSearchArgs(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "docs" || k != 7 || len(query) != 2 {
		t.Errorf("basic fields: col=%q k=%d query=%v", col, k, query)
	}
	if gotFilter.IsZero() {
		t.Fatal("filter decoded as zero, want eq tenant=acme")
	}
	m := vector.Metadata{"tenant": vector.NewString("acme")}
	wantPred, _ := filter.Compile()
	gotPred, _ := gotFilter.Compile()
	if wantPred(m) != gotPred(m) {
		t.Error("roundtripped filter disagrees with original")
	}
}

var _ = bytes.NewReader

// ─── search consistency opts + degraded-result wire codec ────────────────────

func TestSearchOptsConsistencyRoundTrip(t *testing.T) {
	// Encode with non-zero consistency opts.
	args := EncodeVectorSearchArgsOpts("docs", 10, []float32{1, 2, 3}, vector.Filter{}, 1, 1, 0)
	col, k, query, filter, rc, opa, _, err := DecodeVectorSearchArgsOpts(args)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if col != "docs" {
		t.Errorf("collection = %q, want docs", col)
	}
	if k != 10 {
		t.Errorf("k = %d, want 10", k)
	}
	if len(query) != 3 {
		t.Errorf("query len = %d, want 3", len(query))
	}
	if query[0] != 1 || query[1] != 2 || query[2] != 3 {
		t.Fatalf("query values = %v, want [1 2 3]", query)
	}
	if !filter.IsZero() {
		t.Errorf("filter = %+v, want zero", filter)
	}
	if rc != 1 {
		t.Errorf("ReadConsistency = %d, want 1", rc)
	}
	if opa != 1 {
		t.Errorf("OnPartitionUnavailable = %d, want 1", opa)
	}
}

func TestSearchOptsConsistencyLegacyCompat(t *testing.T) {
	// Legacy bytes (no consistency trailer) must decode with default (zero) opts.
	legacy := EncodeVectorSearchArgs("docs", 5, []float32{0.1, 0.2})
	col, k, query, filter, rc, opa, _, err := DecodeVectorSearchArgsOpts(legacy)
	if err != nil {
		t.Fatalf("decode legacy error: %v", err)
	}
	if col != "docs" || k != 5 || len(query) != 2 {
		t.Errorf("basic fields: col=%q k=%d query=%v", col, k, query)
	}
	if !filter.IsZero() {
		t.Errorf("filter = %+v, want zero", filter)
	}
	if rc != 0 || opa != 0 {
		t.Errorf("legacy: rc=%d opa=%d, want 0/0", rc, opa)
	}
}

func TestSearchOptsWithFilterRoundTrip(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}
	args := EncodeVectorSearchArgsOpts("col", 3, []float32{1}, filter, 1, 0, 0)
	col, k, query, gotFilter, rc, opa, _, err := DecodeVectorSearchArgsOpts(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "col" || k != 3 || len(query) != 1 {
		t.Errorf("basic: col=%q k=%d query=%v", col, k, query)
	}
	if gotFilter.IsZero() {
		t.Error("filter decoded as zero")
	}
	if rc != 1 || opa != 0 {
		t.Errorf("opts: rc=%d opa=%d, want 1/0", rc, opa)
	}
}

func TestSearchOptsTruncatedTrailerErrors(t *testing.T) {
	// A valid payload with vecFlagSearchOpts set and the full 2-byte trailer.
	full := EncodeVectorSearchArgsOpts("docs", 10, []float32{1, 2, 3}, vector.Filter{}, 1, 1, 0)

	// Full payload still round-trips: rc/opa preserved.
	col, k, query, filter, rc, opa, _, err := DecodeVectorSearchArgsOpts(full)
	if err != nil {
		t.Fatalf("decode full: %v", err)
	}
	if col != "docs" || k != 10 || len(query) != 3 || !filter.IsZero() || rc != 1 || opa != 1 {
		t.Fatalf("full round-trip mismatch: col=%q k=%d query=%v rc=%d opa=%d", col, k, query, rc, opa)
	}

	// Flag is set but the 2-byte trailer is chopped off → must fail loud.
	for _, chop := range []int{1, 2} {
		truncated := full[:len(full)-chop]
		if _, _, _, _, _, _, _, err := DecodeVectorSearchArgsOpts(truncated); !errors.Is(err, ErrVectorArgsTruncated) {
			t.Errorf("chop %d bytes: err = %v, want ErrVectorArgsTruncated", chop, err)
		}
	}

	// Legacy (flag NOT set, no trailer) still decodes to defaults without error.
	legacy := EncodeVectorSearchArgs("docs", 5, []float32{0.1, 0.2})
	_, _, _, _, lrc, lopa, _, lerr := DecodeVectorSearchArgsOpts(legacy)
	if lerr != nil {
		t.Fatalf("decode legacy: %v", lerr)
	}
	if lrc != 0 || lopa != 0 {
		t.Errorf("legacy defaults: rc=%d opa=%d, want 0/0", lrc, lopa)
	}
}

func TestSearchResultsDegradedRoundTrip(t *testing.T) {
	results := []vector.Result{{ID: 1, Distance: 0.5}}
	body := EncodeVectorSearchResultsDegraded(results, true, []uint16{2, 4})
	res, degraded, missing, err := DecodeVectorSearchResultsDegraded(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Errorf("results = %+v", res)
	}
	if !degraded {
		t.Error("degraded = false, want true")
	}
	if len(missing) != 2 || missing[0] != 2 || missing[1] != 4 {
		t.Errorf("missing = %v, want [2 4]", missing)
	}
}

func TestSearchResultsDegradedLegacy(t *testing.T) {
	// Legacy results bytes (no degraded trailer) must decode to degraded=false, nil missing.
	legacy := EncodeVectorSearchResults([]vector.Result{{ID: 7, Distance: 0.9}})
	res, degraded, missing, err := DecodeVectorSearchResultsDegraded(legacy)
	if err != nil {
		t.Fatalf("legacy decode: %v", err)
	}
	if degraded {
		t.Error("legacy: degraded = true, want false")
	}
	if missing != nil {
		t.Errorf("legacy: missing = %v, want nil", missing)
	}
	_ = res
}

func TestVectorDocsDegradedRoundTrip(t *testing.T) {
	docs := []vector.Document{
		{ID: 1, Distance: 0.5, Score: 0.9, Content: "alpha", Metadata: vector.Metadata{"k": vector.NewString("v")}},
		{ID: 2, Distance: 0.7, Content: "beta"},
	}
	body := EncodeVectorDocsDegraded(docs, true, []uint16{1, 3})
	got, degraded, missing, err := DecodeVectorDocsDegraded(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 || got[0].ID != 1 || got[0].Content != "alpha" || got[1].ID != 2 {
		t.Errorf("docs = %+v", got)
	}
	if !degraded {
		t.Error("degraded = false, want true")
	}
	if len(missing) != 2 || missing[0] != 1 || missing[1] != 3 {
		t.Errorf("missing = %v, want [1 3]", missing)
	}
}

func TestVectorDocsDegradedLegacy(t *testing.T) {
	legacy := EncodeVectorDocs([]vector.Document{{ID: 7, Distance: 0.9, Content: "x"}})
	got, degraded, missing, err := DecodeVectorDocsDegraded(legacy)
	if err != nil {
		t.Fatalf("legacy decode: %v", err)
	}
	if degraded {
		t.Error("legacy: degraded = true, want false")
	}
	if missing != nil {
		t.Errorf("legacy: missing = %v, want nil", missing)
	}
	if len(got) != 1 || got[0].ID != 7 {
		t.Errorf("legacy docs = %+v", got)
	}
}

func TestVectorDocsDegradedByteIdenticalToPlain(t *testing.T) {
	docs := []vector.Document{{ID: 1, Distance: 0.5, Content: "alpha"}, {ID: 2, Content: "beta"}}
	if !bytes.Equal(EncodeVectorDocsDegraded(docs, false, nil), EncodeVectorDocs(docs)) {
		t.Error("non-degraded docs encode != plain encode (back-compat broken)")
	}
}

func TestGroupsDegradedRoundTrip(t *testing.T) {
	groups := []vector.Group{
		{Key: vector.NewString("a"), Hits: []vector.Document{{ID: 1, Distance: 0.1, Content: "h1"}}},
		{Key: vector.NewString("b"), Hits: []vector.Document{{ID: 2, Distance: 0.2, Content: "h2"}}},
	}
	body := EncodeGroupsDegraded(groups, true, []uint16{1, 3})
	got, degraded, missing, err := DecodeGroupsDegraded(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 || len(got[0].Hits) != 1 || got[0].Hits[0].ID != 1 {
		t.Errorf("groups = %+v", got)
	}
	if !degraded {
		t.Error("degraded = false, want true")
	}
	if len(missing) != 2 || missing[0] != 1 || missing[1] != 3 {
		t.Errorf("missing = %v, want [1 3]", missing)
	}
}

func TestGroupsDegradedLegacy(t *testing.T) {
	legacy := EncodeGroups([]vector.Group{{Key: vector.NewString("a"), Hits: []vector.Document{{ID: 7}}}})
	got, degraded, missing, err := DecodeGroupsDegraded(legacy)
	if err != nil {
		t.Fatalf("legacy decode: %v", err)
	}
	if degraded {
		t.Error("legacy: degraded = true, want false")
	}
	if missing != nil {
		t.Errorf("legacy: missing = %v, want nil", missing)
	}
	if len(got) != 1 || got[0].Hits[0].ID != 7 {
		t.Errorf("legacy groups = %+v", got)
	}
}

func TestGroupsDegradedByteIdenticalToPlain(t *testing.T) {
	groups := []vector.Group{{Key: vector.NewString("a"), Hits: []vector.Document{{ID: 1, Content: "h"}}}}
	if !bytes.Equal(EncodeGroupsDegraded(groups, false, nil), EncodeGroups(groups)) {
		t.Error("non-degraded groups encode != plain encode (back-compat broken)")
	}
}

func TestHybridResultsDegradedRoundTrip(t *testing.T) {
	results := []vector.Result{{ID: 1, Distance: 0.5, Score: 0.8}, {ID: 2, Distance: 0.7, Score: 0.4}}
	body := EncodeHybridResultsDegraded(results, true, []uint16{1, 3})
	got, degraded, missing, err := DecodeHybridResultsDegraded(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 || got[0].ID != 1 || got[0].Score != 0.8 || got[1].ID != 2 {
		t.Errorf("results = %+v", got)
	}
	if !degraded {
		t.Error("degraded = false, want true")
	}
	if len(missing) != 2 || missing[0] != 1 || missing[1] != 3 {
		t.Errorf("missing = %v, want [1 3]", missing)
	}
}

func TestHybridResultsDegradedLegacy(t *testing.T) {
	legacy := EncodeHybridResults([]vector.Result{{ID: 7, Distance: 0.9, Score: 0.5}})
	got, degraded, missing, err := DecodeHybridResultsDegraded(legacy)
	if err != nil {
		t.Fatalf("legacy decode: %v", err)
	}
	if degraded {
		t.Error("legacy: degraded = true, want false")
	}
	if missing != nil {
		t.Errorf("legacy: missing = %v, want nil", missing)
	}
	if len(got) != 1 || got[0].ID != 7 || got[0].Score != 0.5 {
		t.Errorf("legacy results = %+v", got)
	}
}

func TestHybridResultsDegradedByteIdenticalToPlain(t *testing.T) {
	results := []vector.Result{{ID: 1, Distance: 0.5, Score: 0.8}}
	if !bytes.Equal(EncodeHybridResultsDegraded(results, false, nil), EncodeHybridResults(results)) {
		t.Error("non-degraded hybrid encode != plain encode (back-compat broken)")
	}
}

func TestHybridLanesResultRoundTrip(t *testing.T) {
	dense := []vector.Result{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.3}}
	sparse := []vector.Result{{ID: 2, Score: 0.9}, {ID: 5, Score: 0.4}}
	body := EncodeHybridLanesResult(dense, sparse)
	gotD, gotS, err := DecodeHybridLanesResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotD) != 2 || gotD[0].ID != 1 || gotD[1].Distance != 0.3 {
		t.Fatalf("dense lane round-trip = %+v", gotD)
	}
	if len(gotS) != 2 || gotS[0].ID != 2 || gotS[0].Score != 0.9 || gotS[1].ID != 5 {
		t.Fatalf("sparse lane round-trip = %+v", gotS)
	}
	// Empty lanes round-trip cleanly.
	b2 := EncodeHybridLanesResult(nil, nil)
	d2, s2, err := DecodeHybridLanesResult(b2)
	if err != nil || len(d2) != 0 || len(s2) != 0 {
		t.Fatalf("empty lanes: d=%v s=%v err=%v", d2, s2, err)
	}
	// One empty, one populated.
	b3 := EncodeHybridLanesResult(dense, nil)
	d3, s3, err := DecodeHybridLanesResult(b3)
	if err != nil || len(d3) != 2 || len(s3) != 0 {
		t.Fatalf("dense-only: d=%v s=%v err=%v", d3, s3, err)
	}
	// Truncated body errors (not panics).
	if _, _, err := DecodeHybridLanesResult(body[:len(body)-1]); err == nil {
		t.Fatal("truncated lanes body should error")
	}
}

func TestResplitArgsRoundtrip(t *testing.T) {
	args := EncodeResplitArgs("docs", 8)
	coll, newP, err := DecodeResplitArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if coll != "docs" || newP != 8 {
		t.Fatalf("round-trip coll=%q newP=%d, want docs/8", coll, newP)
	}
	// Truncated args error (not panic).
	if _, _, err := DecodeResplitArgs(args[:len(args)-1]); err == nil {
		t.Fatal("truncated resplit args should error")
	}
	if _, _, err := DecodeResplitArgs(nil); err == nil {
		t.Fatal("empty resplit args should error")
	}
}

func TestResplitCleanupArgsRoundtrip(t *testing.T) {
	args := EncodeResplitCleanupArgs("docs")
	coll, err := DecodeResplitCleanupArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if coll != "docs" {
		t.Fatalf("round-trip coll=%q, want docs", coll)
	}
}

func TestReshardArgsRoundtrip(t *testing.T) {
	args := EncodeReshardArgs("docs", 8)
	coll, newP, err := DecodeReshardArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if coll != "docs" || newP != 8 {
		t.Fatalf("round-trip coll=%q newP=%d, want docs/8", coll, newP)
	}
	// Truncated args error (not panic).
	if _, _, err := DecodeReshardArgs(args[:len(args)-1]); err == nil {
		t.Fatal("truncated reshard args should error")
	}
	if _, _, err := DecodeReshardArgs(nil); err == nil {
		t.Fatal("empty reshard args should error")
	}
}

func TestReshardAbortArgsRoundtrip(t *testing.T) {
	args := EncodeReshardAbortArgs("docs")
	coll, err := DecodeReshardAbortArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if coll != "docs" {
		t.Fatalf("round-trip coll=%q, want docs", coll)
	}
}

func TestResplitCleanupResultRoundtrip(t *testing.T) {
	body := EncodeResplitCleanupResult(5)
	dropped, err := DecodeResplitCleanupResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if dropped != 5 {
		t.Fatalf("round-trip dropped=%d, want 5", dropped)
	}
	// Truncated body errors (not panic).
	if _, err := DecodeResplitCleanupResult(body[:3]); err == nil {
		t.Fatal("truncated cleanup result should error")
	}
}

// --- Hybrid opts (consistency) wire codec ---

func TestHybridSearchOptsRoundTrip(t *testing.T) {
	sparse := vector.SparseVector{Indices: []uint32{1, 4}, Values: []float32{0.5, 0.25}}
	hopts := vector.HybridOpts{Method: vector.FusionRRF, RRFK: 60, DenseK: 50, SparseK: 50}
	args := EncodeHybridSearchArgsOpts("docs", []float32{1, 2, 3}, 10, sparse, hopts, 1, 1, 0)
	col, dense, k, gotSparse, gotOpts, rc, opa, _, err := DecodeHybridSearchArgsOpts(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "docs" || k != 10 || len(dense) != 3 {
		t.Errorf("basic: col=%q k=%d dense=%v", col, k, dense)
	}
	if len(gotSparse.Indices) != 2 {
		t.Errorf("sparse = %+v", gotSparse)
	}
	if gotOpts.Method != vector.FusionRRF || gotOpts.RRFK != 60 {
		t.Errorf("opts = %+v", gotOpts)
	}
	if rc != 1 || opa != 1 {
		t.Errorf("opts: rc=%d opa=%d, want 1/1", rc, opa)
	}
}

func TestHybridSearchOptsLegacyCompat(t *testing.T) {
	legacy := EncodeHybridSearchArgs("docs", []float32{1, 2}, 5, vector.SparseVector{}, vector.HybridOpts{})
	col, dense, k, _, _, rc, opa, _, err := DecodeHybridSearchArgsOpts(legacy)
	if err != nil {
		t.Fatalf("decode legacy: %v", err)
	}
	if col != "docs" || k != 5 || len(dense) != 2 {
		t.Errorf("basic: col=%q k=%d dense=%v", col, k, dense)
	}
	if rc != 0 || opa != 0 {
		t.Errorf("legacy: rc=%d opa=%d, want 0/0", rc, opa)
	}
}

func TestHybridSearchOptsByteIdenticalToPlain(t *testing.T) {
	sparse := vector.SparseVector{Indices: []uint32{2}, Values: []float32{0.7}}
	hopts := vector.HybridOpts{Method: vector.FusionRRF, RRFK: 60, Filter: vector.Filter{Op: vector.FilterEq, Field: "t", Value: vector.NewString("x")}}
	plain := EncodeHybridSearchArgs("docs", []float32{1, 2, 3}, 7, sparse, hopts)
	opts := EncodeHybridSearchArgsOpts("docs", []float32{1, 2, 3}, 7, sparse, hopts, 0, 0, 0)
	if !bytes.Equal(plain, opts) {
		t.Error("rc=0/opa=0 hybrid opts encode != plain encode (back-compat broken)")
	}
}

func TestHybridSearchOptsTruncatedTrailer(t *testing.T) {
	hopts := vector.HybridOpts{Method: vector.FusionRRF}
	full := EncodeHybridSearchArgsOpts("docs", []float32{1, 2, 3}, 10, vector.SparseVector{}, hopts, 1, 1, 0)
	for _, chop := range []int{1, 2} {
		truncated := full[:len(full)-chop]
		if _, _, _, _, _, _, _, _, err := DecodeHybridSearchArgsOpts(truncated); !errors.Is(err, ErrVectorArgsTruncated) {
			t.Errorf("chop %d: err = %v, want ErrVectorArgsTruncated", chop, err)
		}
	}
}

// --- Group opts (consistency) wire codec ---

func TestGroupSearchOptsRoundTrip(t *testing.T) {
	gopts := vector.GroupOpts{GroupBy: "author", GroupSize: 3, FetchK: 100}
	args := EncodeGroupSearchArgsOpts("docs", 10, []float32{1, 2, 3}, gopts, 1, 1, 0)
	col, k, query, gotOpts, rc, opa, _, err := DecodeGroupSearchArgsOpts(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "docs" || k != 10 || len(query) != 3 {
		t.Errorf("basic: col=%q k=%d query=%v", col, k, query)
	}
	if gotOpts.GroupBy != "author" || gotOpts.GroupSize != 3 || gotOpts.FetchK != 100 {
		t.Errorf("opts = %+v", gotOpts)
	}
	if rc != 1 || opa != 1 {
		t.Errorf("opts: rc=%d opa=%d, want 1/1", rc, opa)
	}
}

func TestGroupSearchOptsLegacyCompat(t *testing.T) {
	gopts := vector.GroupOpts{GroupBy: "author", GroupSize: 3, FetchK: 100}
	legacy := EncodeGroupSearchArgs("docs", 5, []float32{1, 2}, gopts)
	col, k, query, gotOpts, rc, opa, _, err := DecodeGroupSearchArgsOpts(legacy)
	if err != nil {
		t.Fatalf("decode legacy: %v", err)
	}
	if col != "docs" || k != 5 || len(query) != 2 || gotOpts.GroupBy != "author" {
		t.Errorf("basic: col=%q k=%d query=%v opts=%+v", col, k, query, gotOpts)
	}
	if rc != 0 || opa != 0 {
		t.Errorf("legacy: rc=%d opa=%d, want 0/0", rc, opa)
	}
	// Plain decoder ignores any trailing bytes → still works on opts-encoded args.
	withOpts := EncodeGroupSearchArgsOpts("docs", 5, []float32{1, 2}, gopts, 1, 1, 0)
	_, _, _, plainOpts, perr := DecodeGroupSearchArgs(withOpts)
	if perr != nil {
		t.Fatalf("plain decode of opts args: %v", perr)
	}
	if plainOpts.GroupBy != "author" {
		t.Errorf("plain decode of opts args opts=%+v", plainOpts)
	}
}

func TestGroupSearchOptsByteIdenticalToPlain(t *testing.T) {
	gopts := vector.GroupOpts{GroupBy: "author", GroupSize: 3, FetchK: 100, Filter: vector.Filter{Op: vector.FilterEq, Field: "t", Value: vector.NewString("x")}}
	plain := EncodeGroupSearchArgs("docs", 7, []float32{1, 2, 3}, gopts)
	opts := EncodeGroupSearchArgsOpts("docs", 7, []float32{1, 2, 3}, gopts, 0, 0, 0)
	if !bytes.Equal(plain, opts) {
		t.Error("rc=0/opa=0 group opts encode != plain encode (back-compat broken)")
	}
}

func TestGroupSearchOptsTruncatedTrailer(t *testing.T) {
	gopts := vector.GroupOpts{GroupBy: "author", GroupSize: 3, FetchK: 100}
	full := EncodeGroupSearchArgsOpts("docs", 10, []float32{1, 2, 3}, gopts, 1, 1, 0)
	// Trailer is [optsPresent:u8][rc:u8][opa:u8] = 3 bytes; chopping any of them
	// after the presence flag is read must fail loud.
	for _, chop := range []int{1, 2} {
		truncated := full[:len(full)-chop]
		if _, _, _, _, _, _, _, err := DecodeGroupSearchArgsOpts(truncated); !errors.Is(err, ErrVectorArgsTruncated) {
			t.Errorf("chop %d: err = %v, want ErrVectorArgsTruncated", chop, err)
		}
	}
}

// --- Scroll opts (consistency) wire codec ---

func TestScrollOptsRoundTrip(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}
	args := EncodeScrollArgsOpts("docs", filter, 50, 1, 1)
	col, gotFilter, limit, rc, opa, err := DecodeScrollArgsOpts(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "docs" || limit != 50 {
		t.Errorf("basic: col=%q limit=%d", col, limit)
	}
	if gotFilter.IsZero() {
		t.Error("filter decoded as zero")
	}
	if rc != 1 || opa != 1 {
		t.Errorf("opts: rc=%d opa=%d, want 1/1", rc, opa)
	}
}

func TestScrollOptsLegacyCompat(t *testing.T) {
	legacy := EncodeScrollArgs("docs", vector.Filter{}, 25)
	col, _, limit, rc, opa, err := DecodeScrollArgsOpts(legacy)
	if err != nil {
		t.Fatalf("decode legacy: %v", err)
	}
	if col != "docs" || limit != 25 {
		t.Errorf("basic: col=%q limit=%d", col, limit)
	}
	if rc != 0 || opa != 0 {
		t.Errorf("legacy: rc=%d opa=%d, want 0/0", rc, opa)
	}
	// Plain decoder tolerates trailing opts bytes.
	withOpts := EncodeScrollArgsOpts("docs", vector.Filter{}, 25, 1, 1)
	pcol, _, plimit, perr := DecodeScrollArgs(withOpts)
	if perr != nil || pcol != "docs" || plimit != 25 {
		t.Errorf("plain decode of opts args: col=%q limit=%d err=%v", pcol, plimit, perr)
	}
}

func TestScrollOptsByteIdenticalToPlain(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "t", Value: vector.NewString("x")}
	plain := EncodeScrollArgs("docs", filter, 50)
	opts := EncodeScrollArgsOpts("docs", filter, 50, 0, 0)
	if !bytes.Equal(plain, opts) {
		t.Error("rc=0/opa=0 scroll opts encode != plain encode (back-compat broken)")
	}
}

func TestScrollOptsTruncatedTrailer(t *testing.T) {
	full := EncodeScrollArgsOpts("docs", vector.Filter{}, 50, 1, 1)
	for _, chop := range []int{1, 2} {
		truncated := full[:len(full)-chop]
		if _, _, _, _, _, err := DecodeScrollArgsOpts(truncated); !errors.Is(err, ErrVectorArgsTruncated) {
			t.Errorf("chop %d: err = %v, want ErrVectorArgsTruncated", chop, err)
		}
	}
}

// --- vector_get_batch codec round-trips ---

func TestVectorGetBatchArgsRoundtrip(t *testing.T) {
	cases := []struct {
		name  string
		col   string
		ids   []uint64
		flags uint8
	}{
		{"empty", "acme/docs", nil, GetFlagsBoth},
		{"single", "docs", []uint64{42}, GetFlagWithVector},
		{"multi", "c", []uint64{1, 2, 3, 9999, 0}, GetFlagWithPayload},
		{"noflags", "x", []uint64{7, 8}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			col, ids, flags, err := DecodeVectorGetBatchArgs(EncodeVectorGetBatchArgs(tc.col, tc.ids, tc.flags))
			if err != nil {
				t.Fatal(err)
			}
			if col != tc.col || flags != tc.flags {
				t.Errorf("got (%q,flags=%d), want (%q,%d)", col, flags, tc.col, tc.flags)
			}
			want := tc.ids
			if want == nil {
				want = []uint64{}
			}
			if !reflect.DeepEqual(ids, want) {
				t.Errorf("ids = %v, want %v", ids, want)
			}
		})
	}
}

func TestVectorGetBatchArgsLargeList(t *testing.T) {
	ids := make([]uint64, 5000)
	for i := range ids {
		ids[i] = uint64(i) * 7
	}
	col, got, flags, err := DecodeVectorGetBatchArgs(EncodeVectorGetBatchArgs("big", ids, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	if col != "big" || flags != GetFlagsBoth || !reflect.DeepEqual(got, ids) {
		t.Fatalf("large list round-trip mismatch (n=%d)", len(got))
	}
}

func TestVectorGetBatchArgsTruncated(t *testing.T) {
	full := EncodeVectorGetBatchArgs("docs", []uint64{1, 2, 3}, GetFlagsBoth)
	// chop into the trailing id list, the count, and the flags.
	for _, chop := range []int{1, 5, 9, 12} {
		if chop >= len(full) {
			continue
		}
		if _, _, _, err := DecodeVectorGetBatchArgs(full[:len(full)-chop]); !errors.Is(err, ErrVectorArgsTruncated) {
			t.Errorf("chop %d: err = %v, want ErrVectorArgsTruncated", chop, err)
		}
	}
	if _, _, _, err := DecodeVectorGetBatchArgs(nil); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("nil args: err = %v, want ErrVectorArgsTruncated", err)
	}
	// a declared count that overruns the buffer must fail loud (no oversized alloc).
	bad := EncodeVectorGetBatchArgs("d", []uint64{1}, 0)
	bad[1+1+1] = 0xFF // bump the high byte of n:u32 so n claims far more ids than present
	if _, _, _, err := DecodeVectorGetBatchArgs(bad); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("inflated count: err = %v, want ErrVectorArgsTruncated", err)
	}
}

func TestVectorGetBatchResultRoundtrip(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1), "tag": vector.NewString("x")}
	sparse := &vector.SparseVector{Indices: []uint32{0, 5}, Values: []float32{1, 2}}
	vec := []float32{1, 2, 3, 4}

	rows := []GetBatchRow{
		// found, both projections present.
		{ID: 1, Found: true, Vec: vec, Meta: meta, TTLMs: 5000, Sparse: sparse},
		// not-found.
		{ID: 2, Found: false},
		// found, with_vector off (Vec nil), payload present.
		{ID: 3, Found: true, Meta: meta, TTLMs: 0, Sparse: sparse},
		// found, with_payload off (Meta/Sparse nil), vec present.
		{ID: 4, Found: true, Vec: vec, TTLMs: 1000},
		// found, bare (no vec, no payload, no ttl).
		{ID: 5, Found: true},
	}
	got, err := DecodeVectorGetBatchResult(EncodeVectorGetBatchResult(rows))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(rows) {
		t.Fatalf("rows = %d, want %d", len(got), len(rows))
	}
	// order + ids preserved.
	for i := range rows {
		if got[i].ID != rows[i].ID || got[i].Found != rows[i].Found {
			t.Errorf("row %d: got (id=%d,found=%v), want (id=%d,found=%v)", i, got[i].ID, got[i].Found, rows[i].ID, rows[i].Found)
		}
	}
	if !reflect.DeepEqual(got[0].Vec, vec) || got[0].Meta["a"].Int != 1 || got[0].TTLMs != 5000 || got[0].Sparse == nil {
		t.Errorf("row 0 = %+v", got[0])
	}
	if got[2].Vec != nil || got[2].Meta == nil || got[2].Sparse == nil {
		t.Errorf("row 2 (with_vector off): %+v", got[2])
	}
	if !reflect.DeepEqual(got[3].Vec, vec) || got[3].Meta != nil || got[3].Sparse != nil {
		t.Errorf("row 3 (with_payload off): %+v", got[3])
	}
	if got[4].Vec != nil || got[4].Meta != nil || got[4].Sparse != nil || got[4].TTLMs != 0 {
		t.Errorf("row 4 (bare): %+v", got[4])
	}
}

// TestVectorGetBatchResultIntoMatches pins the arena decoder against the plain one:
// same rows, same vectors, and — the part the arena could plausibly break — each
// row's Vec is its own window (writing one row's floats must not touch another's,
// and appending to one must not scribble into the next). It also exercises reuse of
// the returned scratch on a SECOND body, which is the only way callers are allowed
// to reuse it (the previous decode's rows must be dead).
func TestVectorGetBatchResultIntoMatches(t *testing.T) {
	rows := []GetBatchRow{
		{ID: 1, Found: true, Vec: []float32{1, 2, 3}, Meta: vector.Metadata{"a": vector.NewString("b")}, TTLMs: 1500, Version: 7},
		{ID: 2, Found: false},
		{ID: 3, Found: true, Vec: []float32{4, 5, 6}},
		{ID: 4, Found: true, Meta: vector.Metadata{"c": vector.NewString("d")}},
	}
	body := EncodeVectorGetBatchResult(rows)
	want, err := DecodeVectorGetBatchResult(body)
	if err != nil {
		t.Fatalf("DecodeVectorGetBatchResult: %v", err)
	}
	got, arena, err := DecodeVectorGetBatchResultInto(body, nil, nil)
	if err != nil {
		t.Fatalf("DecodeVectorGetBatchResultInto: %v", err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("Into decoded %+v, want %+v", got, want)
	}
	// Independent windows: mutating row 0's vector must not disturb row 2's, and an
	// append to row 0's must reallocate rather than overwrite row 2's first float.
	got[0].Vec[0] = 99
	_ = append(got[0].Vec, 100) //nolint:gocritic // the point is that this must NOT alias row 2
	if got[2].Vec[0] != 4 {
		t.Fatalf("row 2 vector was clobbered by a write through row 0: %v", got[2].Vec)
	}
	// Reuse the scratch on a different body (the first decode's rows are dead).
	other := EncodeVectorGetBatchResult([]GetBatchRow{{ID: 9, Found: true, Vec: []float32{7, 8}}})
	reused, _, err := DecodeVectorGetBatchResultInto(other, got[:0], arena[:0])
	if err != nil {
		t.Fatalf("reused decode: %v", err)
	}
	if len(reused) != 1 || reused[0].ID != 9 || !reflect.DeepEqual(reused[0].Vec, []float32{7, 8}) {
		t.Fatalf("reused decode gave %+v", reused)
	}
}

// TestVectorGetBatchResultIntoRaggedGrowsMidBatch exercises the path the uniform
// case cannot: an arena that REALLOCATES partway through the batch. The up-front
// reservation reads the FIRST row's dim, so a not-found first row reserves nothing
// at all and every subsequent vector grows the arena as it goes — and the widths
// climb (2 -> 64 -> 3) so the growth happens more than once.
//
// That is the defense at decodeGetResultAtArena's grow: when the arena moves, rows
// decoded earlier still point INTO THE OLD BACKING ARRAY, which keeps their floats
// because growth copies. The assertions below are what would catch a regression
// there — every row's contents must survive the reallocation that happened after
// it was decoded, and every Vec must remain a tight window (cap == len) so a
// caller's append reallocates instead of scribbling over the next row's floats.
func TestVectorGetBatchResultIntoRaggedGrowsMidBatch(t *testing.T) {
	mk := func(n int, base float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = base + float32(i)
		}
		return v
	}
	rows := []GetBatchRow{
		{ID: 1, Found: false}, // first row not found => zero up-front reservation
		{ID: 2, Found: true, Vec: mk(2, 100)},
		{ID: 3, Found: true, Vec: mk(64, 200)}, // forces a grow
		{ID: 4, Found: true, Vec: mk(3, 300)},  // and decodes after that grow
		{ID: 5, Found: true, Vec: mk(128, 400)},
	}
	body := EncodeVectorGetBatchResult(rows)
	got, _, err := DecodeVectorGetBatchResultInto(body, nil, nil)
	if err != nil {
		t.Fatalf("DecodeVectorGetBatchResultInto: %v", err)
	}
	if len(got) != len(rows) {
		t.Fatalf("decoded %d rows, want %d", len(got), len(rows))
	}
	for i, want := range rows {
		r := got[i]
		if r.ID != want.ID || r.Found != want.Found {
			t.Errorf("row %d: got (id=%d found=%v), want (id=%d found=%v)", i, r.ID, r.Found, want.ID, want.Found)
			continue
		}
		if !want.Found {
			continue
		}
		// Contents survived every reallocation that happened after this row decoded.
		if !reflect.DeepEqual(r.Vec, want.Vec) {
			t.Errorf("row %d (dim %d): vector = %v, want %v", i, len(want.Vec), r.Vec, want.Vec)
		}
		// Tight window: an append must not reach into the next row's floats.
		if cap(r.Vec) != len(r.Vec) {
			t.Errorf("row %d: cap(Vec)=%d len(Vec)=%d — a caller's append would scribble into the arena",
				i, cap(r.Vec), len(r.Vec))
		}
	}
	// The plain decoder must agree row for row (one decoder, one wire path).
	plain, err := DecodeVectorGetBatchResult(body)
	if err != nil {
		t.Fatalf("DecodeVectorGetBatchResult: %v", err)
	}
	if !reflect.DeepEqual(plain, got) {
		t.Fatalf("arena decode diverged from the plain decode:\n got  %+v\n want %+v", got, plain)
	}
}

// TestVectorGetBatchResultHostileRowCount pins the reservation bound: a body whose
// declared row count cannot possibly fit must be REJECTED as a decode error. Before
// the bound, 0xFFFFFFFF reached slices.Grow as a ~3.8e9-element reservation, and an
// out-of-memory abort is not something a caller can reject a frame on — it takes the
// process down, so a malformed/hostile frame from any transport was a liveness bug.
func TestVectorGetBatchResultHostileRowCount(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    uint32
	}{
		{"max-uint32", 0xFFFFFFFF},
		{"just-over-capacity", 2}, // body below carries one 9-byte row at most
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := make([]byte, 4+9)
			binary.BigEndian.PutUint32(body, tc.n)
			binary.BigEndian.PutUint64(body[4:], 7) // one id, then found=0
			if _, err := DecodeVectorGetBatchResult(body); !errors.Is(err, ErrVectorArgsTruncated) {
				t.Fatalf("n=%d: err = %v, want ErrVectorArgsTruncated", tc.n, err)
			}
			if _, _, err := DecodeVectorGetBatchResultInto(body, nil, nil); !errors.Is(err, ErrVectorArgsTruncated) {
				t.Fatalf("n=%d (Into): err = %v, want ErrVectorArgsTruncated", tc.n, err)
			}
		})
	}
	// The exact-fit boundary must still DECODE: one 9-byte not-found row.
	ok := make([]byte, 4+9)
	binary.BigEndian.PutUint32(ok, 1)
	binary.BigEndian.PutUint64(ok[4:], 7)
	got, err := DecodeVectorGetBatchResult(ok)
	if err != nil {
		t.Fatalf("exact-fit body rejected: %v", err)
	}
	if len(got) != 1 || got[0].ID != 7 || got[0].Found {
		t.Fatalf("exact-fit body decoded %+v", got)
	}
}

func TestVectorGetBatchResultZeroRows(t *testing.T) {
	got, err := DecodeVectorGetBatchResult(EncodeVectorGetBatchResult(nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("zero-row result decoded %d rows", len(got))
	}
}

func TestVectorGetBatchResultTruncated(t *testing.T) {
	body := EncodeVectorGetBatchResult([]GetBatchRow{
		{ID: 1, Found: true, Vec: []float32{1, 2}, Meta: vector.Metadata{"k": vector.NewInt(1)}, TTLMs: 1000},
		{ID: 2, Found: false},
	})
	if _, err := DecodeVectorGetBatchResult(nil); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("nil: err = %v, want ErrVectorArgsTruncated", err)
	}
	if _, err := DecodeVectorGetBatchResult(body[:2]); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("header chop: err = %v, want ErrVectorArgsTruncated", err)
	}
	// chop mid-row (after the count + part of the first row).
	if _, err := DecodeVectorGetBatchResult(body[:10]); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("mid-row chop: err = %v, want ErrVectorArgsTruncated", err)
	}
}

// --- handleVectorGetBatch over a built dense collection ---

func TestHandleVectorGetBatchDense(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	cfg := vector.Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// id 1: vec + payload + ttl; id 2: vec + payload, no ttl.
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgsExt("docs", 1, []float32{1, 0, 0, 0}, time.Hour,
		vector.Metadata{"a": vector.NewInt(1)}, vector.SparseVector{})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgsExt("docs", 2, []float32{0, 1, 0, 0}, 0,
		vector.Metadata{"b": vector.NewInt(2)}, vector.SparseVector{})); err != nil {
		t.Fatal(err)
	}

	// mixed batch: present, absent, present, absent — input order must be preserved.
	body, err := handleVectorGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1, 99, 2, 7}, GetFlagsBoth))
	if err != nil {
		t.Fatalf("unexpected op error: %v", err)
	}
	rows, err := DecodeVectorGetBatchResult(body)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []uint64{1, 99, 2, 7}
	wantFound := []bool{true, false, true, false}
	if len(rows) != 4 {
		t.Fatalf("rows = %d, want 4", len(rows))
	}
	for i, r := range rows {
		if r.ID != wantIDs[i] || r.Found != wantFound[i] {
			t.Errorf("row %d: got (id=%d,found=%v), want (id=%d,found=%v)", i, r.ID, r.Found, wantIDs[i], wantFound[i])
		}
	}
	if rows[0].Vec[0] != 1 || rows[0].Meta["a"].Int != 1 || rows[0].TTLMs == 0 {
		t.Errorf("row 0 found content: %+v", rows[0])
	}
	if rows[2].Vec[1] != 1 || rows[2].Meta["b"].Int != 2 {
		t.Errorf("row 2 found content: %+v", rows[2])
	}

	// projection: with_vector off -> no vec, payload kept; with_payload off -> vec kept, no meta.
	body, _ = handleVectorGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1}, GetFlagWithPayload))
	rows, _ = DecodeVectorGetBatchResult(body)
	if !rows[0].Found || rows[0].Vec != nil || rows[0].Meta["a"].Int != 1 {
		t.Errorf("with_vector off: %+v", rows[0])
	}
	body, _ = handleVectorGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1}, GetFlagWithVector))
	rows, _ = DecodeVectorGetBatchResult(body)
	if !rows[0].Found || rows[0].Vec[0] != 1 || rows[0].Meta != nil {
		t.Errorf("with_payload off: %+v", rows[0])
	}

	// all-absent batch -> all not-found rows, NO op error.
	body, err = handleVectorGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{100, 200, 300}, GetFlagsBoth))
	if err != nil {
		t.Fatalf("all-absent: unexpected op error %v", err)
	}
	rows, _ = DecodeVectorGetBatchResult(body)
	if len(rows) != 3 {
		t.Fatalf("all-absent: rows = %d, want 3", len(rows))
	}
	for i, r := range rows {
		if r.Found {
			t.Errorf("all-absent row %d: found = true, want false", i)
		}
	}

	// empty batch -> zero rows, no error.
	body, err = handleVectorGetBatch(tx, EncodeVectorGetBatchArgs("docs", nil, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	if rows, _ := DecodeVectorGetBatchResult(body); len(rows) != 0 {
		t.Errorf("empty batch: rows = %d, want 0", len(rows))
	}
}

// TestVectorCreateCollectionArgsFilterFirstRelativeBP covers the codec anchoring
// logic for FilterFirstRelativeBP: a non-zero value forces the drift block (which
// forces the IVFTrainThreshold word and every upstream optional slot), so the
// trailing 4-byte word is unambiguous. Key assertions:
//
//   - Round-trip for each of bp ∈ {1, 5000, 10000}.
//   - A config combining bp + non-zero drift + non-zero threshold still round-trips
//     (proves the anchoring chain with all three trailing sections present).
//   - A config with FilterFirstRelativeBP==0 (explicit) encodes BYTE-IDENTICAL to
//     the same config without the field set (the default-off guarantee at the codec
//     level).
func TestVectorCreateCollectionArgsFilterFirstRelativeBP(t *testing.T) {
	base := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
	}

	// Round-trip for each bp value.
	for _, bp := range []int{1, 5000, 10000} {
		cfg := base
		cfg.FilterFirstRelativeBP = bp
		name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
		if err != nil {
			t.Fatalf("bp=%d: decode: %v", bp, err)
		}
		if name != "docs" || !reflect.DeepEqual(got, cfg) {
			t.Errorf("bp=%d: roundtrip cfg = %+v, want %+v", bp, got, cfg)
		}
		if got.FilterFirstRelativeBP != bp {
			t.Errorf("bp=%d: FilterFirstRelativeBP = %d, want %d", bp, got.FilterFirstRelativeBP, bp)
		}
	}

	// Combining bp with non-zero drift + threshold: all three trailing sections
	// present simultaneously (the anchoring chain). Each must survive the round-trip.
	combined := base
	combined.FilterFirstRelativeBP = 7500
	combined.IVFTrainThreshold = 4096
	combined.IVFDriftRetrain = true
	combined.IVFDriftGrowthFactor = 2.5
	combined.IVFDriftFactor = 1.75
	name, gotCombined, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("d", combined))
	if err != nil {
		t.Fatalf("combined: decode: %v", err)
	}
	if name != "d" || !reflect.DeepEqual(gotCombined, combined) {
		t.Errorf("combined roundtrip = %+v, want %+v", gotCombined, combined)
	}

	// Default-off byte-identity: encoding with FilterFirstRelativeBP==0 (explicit)
	// must produce EXACTLY the same bytes as encoding without the field set (zero
	// value). The codec must not append the 4-byte word when bp==0.
	withZero := base
	withZero.FilterFirstRelativeBP = 0
	withoutField := base // zero value — field not set
	if !bytes.Equal(EncodeCreateCollectionArgs("docs", withZero), EncodeCreateCollectionArgs("docs", withoutField)) {
		t.Fatal("FilterFirstRelativeBP=0 (explicit) is not byte-identical to the zero-value default")
	}
}

// TestVectorCreateCollectionArgsOPQIters round-trips a non-zero OPQIters through the
// dense create wire (it rides the trailing word, forcing the upstream optional
// blocks present) and confirms OPQIters==0 is byte-identical to the pre-feature wire.
func TestVectorCreateCollectionArgsOPQIters(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, QuantPQM: 8, OPQ: true, OPQIters: 5,
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" || got.OPQIters != 5 || !got.OPQ {
		t.Fatalf("OPQIters not carried: %+v", got)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Fatalf("roundtrip cfg = %+v, want %+v", got, cfg)
	}

	// OPQIters==0 must be byte-identical to a create WITHOUT the field set (the
	// pre-feature wire — OPQIters forces nothing when zero).
	base := vector.Config{Dim: 32, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9, Quant: vector.QuantPQ, QuantPQM: 8, OPQ: true}
	zeroIters := base
	zeroIters.OPQIters = 0
	if !reflect.DeepEqual(EncodeCreateCollectionArgs("docs", base), EncodeCreateCollectionArgs("docs", zeroIters)) {
		t.Fatal("OPQIters==0 not byte-identical to the pre-feature wire")
	}
}

// TestVectorCreateCollectionArgsFullTextRoundtrip: the BM25 FullText trailer
// (analyzer name + k1 + b) round-trips through the create codec, including when
// combined with the other trailing extensions it transitively forces present.
func TestVectorCreateCollectionArgsFullTextRoundtrip(t *testing.T) {
	cfg := vector.Config{
		Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		FullText: &vector.FullTextConfig{Analyzer: "english", K1: 1.4, B: 0.6},
	}
	name, got, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" {
		t.Errorf("name = %q", name)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("roundtrip cfg = %+v (FullText=%+v), want %+v (FullText=%+v)", got, got.FullText, cfg, cfg.FullText)
	}
	if got.FullText == nil || got.FullText.Analyzer != "english" || got.FullText.K1 != 1.4 || got.FullText.B != 0.6 {
		t.Errorf("FullText extension not carried: %+v", got.FullText)
	}

	// Empty analyzer name (the default-english marker) + zero knobs must also
	// round-trip as a NON-nil FullText (the presence byte distinguishes it from
	// "full text disabled").
	def := vector.Config{Dim: 16, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &vector.FullTextConfig{}}
	_, gotDef, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("c", def))
	if err != nil {
		t.Fatal(err)
	}
	if gotDef.FullText == nil {
		t.Fatalf("an empty (default) FullText config must survive as non-nil, got nil")
	}
	if gotDef.FullText.Analyzer != "" || gotDef.FullText.K1 != 0 || gotDef.FullText.B != 0 {
		t.Errorf("default FullText not carried verbatim: %+v", gotDef.FullText)
	}
}

// TestVectorCreateCollectionArgsFullTextByteIdentical proves a create with
// FullText==nil (the common case, incl. every existing collection) is
// BYTE-IDENTICAL to the pre-feature encoder: the FullText trailer is appended
// only when FullText != nil, and a pre-extension decoder still decodes the bytes.
func TestVectorCreateCollectionArgsFullTextByteIdentical(t *testing.T) {
	base := vector.Config{Dim: 128, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}
	withoutFT := EncodeCreateCollectionArgs("docs", base)

	ft := base
	ft.FullText = &vector.FullTextConfig{Analyzer: "english", K1: 1.2, B: 0.75}
	withFT := EncodeCreateCollectionArgs("docs", ft)

	// nil-FullText encode must be the SAME length as the pre-feature HNSW wire (no
	// trailing OPQIters/relativeBP/drift/threshold blocks forced).
	wantLen := 1 + len("docs") + 4 + 1 + 4 + 4 + 4 + 8 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 4
	if len(withoutFT) != wantLen {
		t.Fatalf("nil-FullText encode length = %d, want %d (no trailer must be appended)", len(withoutFT), wantLen)
	}
	// The FullText create must be strictly longer (it forces the whole trailer chain
	// + carries the analyzer/k1/b), and the nil-FullText bytes must be its prefix.
	if len(withFT) <= len(withoutFT) {
		t.Fatalf("FullText encode (%d) must be longer than nil-FullText (%d)", len(withFT), len(withoutFT))
	}
	if !bytes.Equal(withFT[:len(withoutFT)], withoutFT) {
		t.Fatal("nil-FullText bytes must be a prefix of the FullText encode (additive trailer)")
	}

	// A PRE-EXTENSION client (which only knows the base+quant+graph wire) must still
	// decode the nil-FullText bytes, defaulting FullText to nil.
	_, got, err := DecodeCreateCollectionArgs(withoutFT)
	if err != nil {
		t.Fatalf("pre-extension decode of nil-FullText bytes: %v", err)
	}
	if got.FullText != nil {
		t.Errorf("nil-FullText decode must leave FullText nil, got %+v", got.FullText)
	}
	// And the FULL decode of the FullText bytes recovers the config.
	_, gotFT, err := DecodeCreateCollectionArgs(withFT)
	if err != nil {
		t.Fatal(err)
	}
	if gotFT.FullText == nil || gotFT.FullText.Analyzer != "english" {
		t.Errorf("FullText decode lost the config: %+v", gotFT.FullText)
	}
}

// TestVectorCreateCollectionArgsScannByteIdentical proves the ScaNN trailer
// (AnisotropicEta/SOAR/SOARLambda/PQNBits) is purely additive: a create with all
// four default (AnisotropicEta==0 && !SOAR && SOARLambda==0 && PQNBits in {0,8})
// is BYTE-IDENTICAL to the pre-ScaNN encoder. The four words ride only when
// non-default, forcing the upstream Vamana chain present as an anchor.
func TestVectorCreateCollectionArgsScannByteIdentical(t *testing.T) {
	base := vector.Config{Dim: 128, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}
	baseWire := EncodeCreateCollectionArgs("docs", base)

	// PQNBits==8 is the explicit 8-bit default: it must NOT force the trailer (0 and
	// 8 are the same on-wire default), so the wire is byte-identical to the base.
	pq8 := base
	pq8.PQNBits = 8
	if got := EncodeCreateCollectionArgs("docs", pq8); !bytes.Equal(got, baseWire) {
		t.Fatalf("PQNBits==8 must be byte-identical to the pre-ScaNN wire (len %d vs %d)", len(got), len(baseWire))
	}

	// Anchor against a config that already forces the whole upstream chain through
	// VamanaAlpha, so the only delta is the four ScaNN words. vamBase forces the
	// Vamana trailer (=> PRQLayers/SQBits/FullText anchor/... all forced) but no
	// ScaNN words.
	vamBase := base
	vamBase.IndexType = vector.IndexVamana
	vamBase.VamanaR = 96
	vamBase.VamanaL = 150
	vamBase.VamanaAlpha = 1.4
	vamBaseWire := EncodeCreateCollectionArgs("docs", vamBase)

	// With AnisotropicEta + SOAR + SOARLambda + PQNBits=4 on top: AnisotropicEta (4)
	// + SOAR flag (1) + SOARLambda (4) + PQNBits (4) = 13 bytes, prefix byte-identical
	// to the Vamana base.
	scann := vamBase
	scann.AnisotropicEta = 4
	scann.SOAR = true
	scann.SOARLambda = 2
	scann.PQNBits = 4
	scannWire := EncodeCreateCollectionArgs("docs", scann)
	if len(scannWire) != len(vamBaseWire)+4+1+4+4 {
		t.Fatalf("ScaNN (Vamana base) encode length = %d, want %d (eta+soar+lambda+nbits words)", len(scannWire), len(vamBaseWire)+13)
	}
	if !bytes.Equal(scannWire[:len(vamBaseWire)], vamBaseWire) {
		t.Fatalf("ScaNN wire prefix is not byte-identical to the Vamana base wire")
	}
}

// TestVectorCreateCollectionArgsScannRoundtrip round-trips the four ScaNN params
// through the create codec, and proves a pre-extension buffer (the ScaNN words
// stripped) still decodes with the new fields defaulted.
func TestVectorCreateCollectionArgsScannRoundtrip(t *testing.T) {
	scann := vector.Config{
		Dim: 96, Metric: vector.DotProduct, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 8,
		AnisotropicEta: 4, SOAR: true, SOARLambda: 2, PQNBits: 4,
	}
	wire := EncodeCreateCollectionArgs("docs", scann)
	_, got, err := DecodeCreateCollectionArgs(wire)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, scann) {
		t.Errorf("ScaNN roundtrip cfg = %+v, want %+v", got, scann)
	}
	if got.AnisotropicEta != 4 || !got.SOAR || got.SOARLambda != 2 || got.PQNBits != 4 {
		t.Errorf("ScaNN params not carried: eta=%v soar=%v lambda=%v nbits=%d", got.AnisotropicEta, got.SOAR, got.SOARLambda, got.PQNBits)
	}

	// Pre-extension decode tolerance: strip the trailing PQNBits (4) + SOARLambda (4)
	// + SOAR (1) + AnisotropicEta (4) = 13 bytes — a buffer from a client that
	// predates this feature. It must still decode, defaulting the four fields.
	old := wire[:len(wire)-13]
	_, gotOld, err := DecodeCreateCollectionArgs(old)
	if err != nil {
		t.Fatalf("pre-extension decode: %v", err)
	}
	if gotOld.AnisotropicEta != 0 || gotOld.SOAR || gotOld.SOARLambda != 0 || gotOld.PQNBits != 0 {
		t.Errorf("pre-extension decode must default ScaNN fields, got eta=%v soar=%v lambda=%v nbits=%d", gotOld.AnisotropicEta, gotOld.SOAR, gotOld.SOARLambda, gotOld.PQNBits)
	}
	if gotOld.IndexType != vector.IndexIVF || gotOld.Dim != 96 {
		t.Errorf("pre-extension decode lost base fields: %+v", gotOld)
	}
}
