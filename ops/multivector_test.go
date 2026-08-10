// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

func TestMVCreateArgsRoundtrip(t *testing.T) {
	cfg := vector.MultiVectorConfig{
		Dim: 16, M: 8, EfConstruction: 100, EfSearch: 50, Seed: 7,
		Quant: vector.QuantBQ1, RescoreFactor: 4, Persistent: true,
	}
	name, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/docs" || got != cfg {
		t.Errorf("roundtrip = %q %+v, want %+v", name, got, cfg)
	}
}

// TestMVCreateArgsIVFRoundtrip proves an IVF / IVF-PQ inner-index MV config
// round-trips through the create codec into the right MultiVectorConfig.
func TestMVCreateArgsIVFRoundtrip(t *testing.T) {
	cfg := vector.MultiVectorConfig{
		Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantNone, RescoreFactor: 3, Partitions: 4,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true, OPQ: true, IVFTrainThreshold: 1000,
	}
	name, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/mv", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/mv" || got != cfg {
		t.Errorf("IVF roundtrip = %q %+v, want %+v", name, got, cfg)
	}
	if got.IndexType != vector.IndexIVF || got.IVFNlist != 64 || got.IVFNprobe != 12 ||
		!got.IVFPQ || got.IVFPQM != 8 || !got.IVFRerank || !got.OPQ || got.IVFTrainThreshold != 1000 {
		t.Errorf("IVF fields not carried: %+v", got)
	}
}

// TestMVCreateArgsDriftRetrainRoundtrip: the MV IVF drift-retrain knobs round-trip
// through the MV create codec, and a no-drift IVF create is byte-identical (the
// 17-byte drift block is appended only when non-default).
func TestMVCreateArgsDriftRetrainRoundtrip(t *testing.T) {
	// Use a full IVF-PQ config (IVFPQ set) so the PQ sub-block is genuinely present —
	// the MV decoder reads the PQ sub-block whenever IndexType==IVF, so the encode and
	// decode stay symmetric (mirrors TestMVCreateArgsIVFRoundtrip).
	cfg := vector.MultiVectorConfig{
		Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantNone, RescoreFactor: 3, Partitions: 4,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true,
		IVFTrainThreshold: 1000,
		IVFDriftRetrain:   true, IVFDriftGrowthFactor: 2.5, IVFDriftFactor: 1.75,
	}
	name, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/mv", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/mv" || got != cfg {
		t.Errorf("drift roundtrip = %q %+v, want %+v", name, got, cfg)
	}
	if !got.IVFDriftRetrain || got.IVFDriftGrowthFactor != 2.5 || got.IVFDriftFactor != 1.75 {
		t.Errorf("drift knobs not carried: %+v", got)
	}

	// No-drift IVF create: must NOT append the drift block (byte-identical prefix).
	noDrift := cfg
	noDrift.IVFDriftRetrain = false
	noDrift.IVFDriftGrowthFactor = 0
	noDrift.IVFDriftFactor = 0
	ndBytes := EncodeMVCreateArgs("acme/mv", noDrift)
	drBytes := EncodeMVCreateArgs("acme/mv", cfg)
	// The drift create appends the 17-byte drift block AND forces the 1-byte PQDropVecs
	// anchor (the no-drift create here omits it since PQDropVecs=false), so the delta is
	// 18 bytes. The key invariant is that adding drift only grows the tail.
	if len(drBytes) != len(ndBytes)+1+1+8+8 {
		t.Fatalf("drift encode len = %d, want no-drift len %d + 18 (pqDrop anchor + drift block)", len(drBytes), len(ndBytes))
	}
	// The no-drift create round-trips with the drift fields defaulted to zero.
	_, ndGot, err := DecodeMVCreateArgs(ndBytes)
	if err != nil {
		t.Fatal(err)
	}
	if ndGot.IVFDriftRetrain || ndGot.IVFDriftGrowthFactor != 0 || ndGot.IVFDriftFactor != 0 {
		t.Errorf("no-drift decode must default drift fields, got %+v", ndGot)
	}
}

// TestMVCreateArgsIVFByteIdentical proves a default (HNSW) MV create is
// byte-identical to the pre-IVF encoder: the IVF block is appended ONLY when
// non-default, so an HNSW config encodes exactly as it did before the IVF fields.
func TestMVCreateArgsIVFByteIdentical(t *testing.T) {
	hnsw := vector.MultiVectorConfig{
		Dim: 16, M: 8, EfConstruction: 100, EfSearch: 50, Seed: 7,
		Quant: vector.QuantBQ1, RescoreFactor: 4, Persistent: true, Partitions: 8,
	}
	full := EncodeMVCreateArgs("docs", hnsw)
	// The pre-IVF wire ends at the partitions u32 — fixed length, no IVF trailer.
	wantLen := 1 + len("docs") + 4 + 4 + 4 + 4 + 8 + 1 + 4 + 1 + 4
	if len(full) != wantLen {
		t.Fatalf("HNSW encode length = %d, want %d (no IVF block for a default config)", len(full), wantLen)
	}
	ivf := hnsw
	ivf.IndexType = vector.IndexIVF
	ivfFull := EncodeMVCreateArgs("docs", ivf)
	// IVF-Flat adds the 9-byte IVF block (indexType:1 + nlist:4 + nprobe:4) only.
	if len(ivfFull) != wantLen+9 {
		t.Fatalf("IVF encode length = %d, want %d", len(ivfFull), wantLen+9)
	}
	if !bytes.Equal(ivfFull[:wantLen], full) {
		t.Fatal("IVF wire prefix is not byte-identical to the HNSW wire")
	}
}

func TestMVCreateArgsPartitionsRoundtrip(t *testing.T) {
	cfg := vector.MultiVectorConfig{
		Dim: 16, M: 8, EfConstruction: 100, EfSearch: 50, Seed: 7,
		Quant: vector.QuantBQ1, RescoreFactor: 4, Persistent: true, Partitions: 8,
	}
	payload := EncodeMVCreateArgs("docs", cfg)
	name, got, err := DecodeMVCreateArgs(payload)
	if err != nil {
		t.Fatal(err)
	}
	if name != "docs" {
		t.Errorf("name = %q, want %q", name, "docs")
	}
	if got.Partitions != 8 {
		t.Errorf("Partitions = %d, want 8", got.Partitions)
	}

	// Backward compatibility: an old-format payload lacks the trailing
	// partitions u32. Truncate it off and confirm it decodes to Partitions=0.
	old := payload[:len(payload)-4]
	_, gotOld, err := DecodeMVCreateArgs(old)
	if err != nil {
		t.Fatal(err)
	}
	if gotOld.Partitions != 0 {
		t.Errorf("old-format Partitions = %d, want 0", gotOld.Partitions)
	}
}

func TestMVGetConfigArgsRoundtrip(t *testing.T) {
	got, err := DecodeMVGetConfigArgs(EncodeMVGetConfigArgs("acme/docs"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "acme/docs" {
		t.Errorf("roundtrip = %q, want %q", got, "acme/docs")
	}
	if _, err := DecodeMVGetConfigArgs(nil); err == nil {
		t.Error("decode empty: want truncation error, got nil")
	}
}

func TestMVAddArgsRoundtrip(t *testing.T) {
	tokens := [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	meta := vector.Metadata{"doc": vector.NewInt(3)}
	name, id, got, gotMeta, err := DecodeMVAddArgs(EncodeMVAddArgs("c", 42, tokens, meta))
	if err != nil {
		t.Fatal(err)
	}
	if name != "c" || id != 42 || len(got) != 3 || got[2][2] != 9 {
		t.Errorf("roundtrip = %q %d %v", name, id, got)
	}
	if gotMeta["doc"].Int != 3 {
		t.Errorf("meta = %+v", gotMeta)
	}
}

func TestMVSearchArgsRoundtrip(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	name, got, k, cpt, err := DecodeMVSearchArgs(EncodeMVSearchArgs("c", q, 5, 128))
	if err != nil {
		t.Fatal(err)
	}
	if name != "c" || k != 5 || cpt != 128 || len(got) != 2 || got[1][1] != 1 {
		t.Errorf("roundtrip = %q k=%d cpt=%d %v", name, k, cpt, got)
	}
}

func TestMVResultsRoundtrip(t *testing.T) {
	in := []vector.MultiResult{
		{ID: 1, Score: 1.5, Metadata: vector.Metadata{"x": vector.NewString("a")}},
		{ID: 2, Score: 0.25},
	}
	got, err := DecodeMVResults(EncodeMVResults(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].ID != 1 || got[0].Score != 1.5 || got[0].Metadata["x"].Str != "a" || got[1].ID != 2 {
		t.Errorf("roundtrip = %+v", got)
	}
}

func TestMVScanArgsRoundtrip(t *testing.T) {
	got, err := DecodeMVScanArgs(EncodeMVScanArgs("acme/docs"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "acme/docs" {
		t.Errorf("roundtrip = %q, want %q", got, "acme/docs")
	}
	if _, err := DecodeMVScanArgs(nil); err == nil {
		t.Error("decode empty: want truncation error, got nil")
	}
}

func TestMVScanResultRoundtrip(t *testing.T) {
	in := []vector.MultiScanRecord{
		{
			ID:       1,
			Tokens:   [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}},
			Metadata: vector.Metadata{"docid": vector.NewInt(1)},
		},
		{
			ID:     2,
			Tokens: [][]float32{{9, 10, 11, 12}}, // distinct token count, no metadata
		},
	}
	got, err := DecodeMVScanResult(EncodeMVScanResult(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("roundtrip count = %d, want 2", len(got))
	}
	if got[0].ID != 1 || len(got[0].Tokens) != 2 || got[0].Tokens[1][3] != 8 {
		t.Errorf("rec0 = %+v", got[0])
	}
	if got[0].Metadata["docid"].Int != 1 {
		t.Errorf("rec0 metadata = %+v", got[0].Metadata)
	}
	if got[1].ID != 2 || len(got[1].Tokens) != 1 || got[1].Tokens[0][0] != 9 {
		t.Errorf("rec1 = %+v", got[1])
	}
	if got[1].Metadata != nil {
		t.Errorf("rec1 metadata = %+v, want nil", got[1].Metadata)
	}

	// Empty result round-trips to an empty (non-nil) slice.
	empty, err := DecodeMVScanResult(EncodeMVScanResult(nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(empty) != 0 {
		t.Errorf("empty scan = %+v, want 0 records", empty)
	}

	// A record with zero tokens encodes/decodes safely.
	zero, err := DecodeMVScanResult(EncodeMVScanResult([]vector.MultiScanRecord{{ID: 7}}))
	if err != nil {
		t.Fatal(err)
	}
	if len(zero) != 1 || zero[0].ID != 7 || len(zero[0].Tokens) != 0 {
		t.Errorf("zero-token record = %+v", zero)
	}
}

// TestMVViaDispatch drives the full op path: create → add two docs → search →
// delete, asserting the MaxSim winner.
func TestMVViaDispatch(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("docs", vector.MultiVectorConfig{Dim: 4})); err != nil {
		t.Fatal(err)
	}
	// doc 1 has a token aligned with the query; doc 2 does not.
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", 1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, vector.Metadata{"d": vector.NewInt(1)})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", 2, [][]float32{{0, 0, 1, 0}}, nil)); err != nil {
		t.Fatal(err)
	}

	body, err := handleMVSearch(tx, EncodeMVSearchArgs("docs", [][]float32{{1, 0, 0, 0}}, 2, 50))
	if err != nil {
		t.Fatal(err)
	}
	res, err := DecodeMVResults(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != 1 {
		t.Fatalf("search = %+v, want doc 1 first", res)
	}
	if res[0].Metadata["d"].Int != 1 {
		t.Errorf("winner metadata = %+v", res[0].Metadata)
	}

	// Delete doc 1; doc 2 remains.
	db, err := handleMVDelete(tx, EncodeMVDeleteArgs("docs", 1))
	if err != nil {
		t.Fatal(err)
	}
	if len(db) == 0 || db[0] != 1 {
		t.Errorf("delete = %v, want [1]", db)
	}
	body, _ = handleMVSearch(tx, EncodeMVSearchArgs("docs", [][]float32{{1, 0, 0, 0}}, 5, 50))
	res, _ = DecodeMVResults(body)
	if len(res) != 1 || res[0].ID != 2 {
		t.Errorf("after delete = %+v, want only doc 2", res)
	}
}

func TestMVResultsDegradedRoundTrip(t *testing.T) {
	results := []vector.MultiResult{
		{ID: 1, Score: 0.9, Metadata: vector.Metadata{"k": vector.NewString("v")}},
		{ID: 2, Score: 0.4},
	}
	body := EncodeMVResultsDegraded(results, true, []uint16{1, 3})
	got, degraded, missing, err := DecodeMVResultsDegraded(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 || got[0].ID != 1 || got[0].Score != 0.9 || got[1].ID != 2 {
		t.Errorf("results = %+v", got)
	}
	if !degraded {
		t.Error("degraded = false, want true")
	}
	if len(missing) != 2 || missing[0] != 1 || missing[1] != 3 {
		t.Errorf("missing = %v, want [1 3]", missing)
	}
}

func TestMVResultsDegradedLegacy(t *testing.T) {
	legacy := EncodeMVResults([]vector.MultiResult{{ID: 7, Score: 0.5}})
	got, degraded, missing, err := DecodeMVResultsDegraded(legacy)
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

func TestMVResultsDegradedByteIdenticalToPlain(t *testing.T) {
	results := []vector.MultiResult{{ID: 1, Score: 0.9}}
	if !bytes.Equal(EncodeMVResultsDegraded(results, false, nil), EncodeMVResults(results)) {
		t.Error("non-degraded mv encode != plain encode (back-compat broken)")
	}
}

// --- MV search opts (consistency) wire codec ---

func TestMVSearchOptsRoundTrip(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	args := EncodeMVSearchArgsOpts("c", q, 5, 128, 1, 1, 0)
	name, got, k, cpt, rc, opa, _, err := DecodeMVSearchArgsOpts(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if name != "c" || k != 5 || cpt != 128 || len(got) != 2 || got[1][1] != 1 {
		t.Errorf("roundtrip = %q k=%d cpt=%d %v", name, k, cpt, got)
	}
	if rc != 1 || opa != 1 {
		t.Errorf("opts: rc=%d opa=%d, want 1/1", rc, opa)
	}
}

func TestMVSearchOptsLegacyCompat(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	legacy := EncodeMVSearchArgs("c", q, 5, 128)
	name, got, k, cpt, rc, opa, _, err := DecodeMVSearchArgsOpts(legacy)
	if err != nil {
		t.Fatalf("decode legacy: %v", err)
	}
	if name != "c" || k != 5 || cpt != 128 || len(got) != 2 {
		t.Errorf("roundtrip = %q k=%d cpt=%d %v", name, k, cpt, got)
	}
	if rc != 0 || opa != 0 {
		t.Errorf("legacy: rc=%d opa=%d, want 0/0", rc, opa)
	}
	// Plain decoder tolerates trailing opts bytes.
	withOpts := EncodeMVSearchArgsOpts("c", q, 5, 128, 1, 1, 0)
	pname, _, pk, pcpt, perr := DecodeMVSearchArgs(withOpts)
	if perr != nil || pname != "c" || pk != 5 || pcpt != 128 {
		t.Errorf("plain decode of opts args: name=%q k=%d cpt=%d err=%v", pname, pk, pcpt, perr)
	}
}

func TestMVSearchOptsByteIdenticalToPlain(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	plain := EncodeMVSearchArgs("c", q, 5, 128)
	opts := EncodeMVSearchArgsOpts("c", q, 5, 128, 0, 0, 0)
	if !bytes.Equal(plain, opts) {
		t.Error("rc=0/opa=0 mv opts encode != plain encode (back-compat broken)")
	}
}

func TestMVSearchOptsTruncatedTrailer(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	full := EncodeMVSearchArgsOpts("c", q, 5, 128, 1, 1, 0)
	for _, chop := range []int{1, 2} {
		truncated := full[:len(full)-chop]
		if _, _, _, _, _, _, _, err := DecodeMVSearchArgsOpts(truncated); !errors.Is(err, errVectorArgsTruncated) {
			t.Errorf("chop %d: err = %v, want errVectorArgsTruncated", chop, err)
		}
	}
}

// --- MV search filter wire codec (payload filter on the wire) ---

func mvTestFilter() vector.Filter {
	return vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}
}

// TestMVSearchFilterRoundTrip exercises every combination of {filter, rc, opa}:
// filter-only, rc-only, opa-only, all-three, none — and asserts the filter (and
// rc/opa) decode back unchanged.
func TestMVSearchFilterRoundTrip(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	f := mvTestFilter()
	cases := []struct {
		name           string
		filter         vector.Filter
		rc, opa        uint8
		wantFilterZero bool
	}{
		{"none", vector.Filter{}, 0, 0, true},
		{"filter-only", f, 0, 0, false},
		{"rc-only", vector.Filter{}, 1, 0, true},
		{"opa-only", vector.Filter{}, 0, 1, true},
		{"all-three", f, 1, 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			args := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, tc.rc, tc.opa, tc.filter, 0)
			name, got, k, cpt, rc, opa, gotFilter, _, err := DecodeMVSearchArgsOptsFilter(args)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if name != "c" || k != 5 || cpt != 128 || len(got) != 2 || got[1][1] != 1 {
				t.Errorf("base roundtrip = %q k=%d cpt=%d %v", name, k, cpt, got)
			}
			if rc != tc.rc || opa != tc.opa {
				t.Errorf("rc=%d opa=%d, want %d/%d", rc, opa, tc.rc, tc.opa)
			}
			if gotFilter.IsZero() != tc.wantFilterZero {
				t.Errorf("filter.IsZero()=%v, want %v (filter=%+v)", gotFilter.IsZero(), tc.wantFilterZero, gotFilter)
			}
			if !tc.wantFilterZero {
				if gotFilter.Op != tc.filter.Op || gotFilter.Field != tc.filter.Field || gotFilter.Value.Str != tc.filter.Value.Str {
					t.Errorf("filter = %+v, want %+v", gotFilter, tc.filter)
				}
			}
		})
	}
}

// TestMVSearchFilterByteIdenticalToPlain is the CRITICAL backward-compat check:
// a no-filter/no-rc/no-opa EncodeMVSearchArgsOptsFilter must be BYTE-IDENTICAL to
// the unchanged base EncodeMVSearchArgs (and to EncodeMVSearchArgsOpts with 0/0).
func TestMVSearchFilterByteIdenticalToPlain(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	plain := EncodeMVSearchArgs("c", q, 5, 128)

	withFilterZero := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, 0, 0, vector.Filter{}, 0)
	if !bytes.Equal(plain, withFilterZero) {
		t.Errorf("zero-filter/0/0 encode != plain encode (back-compat broken)\n plain=%x\n opts =%x", plain, withFilterZero)
	}

	// EncodeMVSearchArgsOpts (the rc/opa-only entry point) must also stay
	// byte-identical to plain when rc==0 && opa==0 (it now delegates through the
	// filter encoder).
	if !bytes.Equal(plain, EncodeMVSearchArgsOpts("c", q, 5, 128, 0, 0, 0)) {
		t.Error("EncodeMVSearchArgsOpts(0,0, 0) != plain encode (back-compat broken)")
	}

	// And the legacy rc/opa-only trailer must be byte-identical too (marker==1,
	// the same "[1][rc][opa]" the pre-filter encoder produced).
	legacyOpts := append(append([]byte(nil), plain...), 1, 1, 1)
	if !bytes.Equal(legacyOpts, EncodeMVSearchArgsOptsFilter("c", q, 5, 128, 1, 1, vector.Filter{}, 0)) {
		t.Error("rc/opa-only trailer != legacy [1][rc][opa] form (wire incompat)")
	}
}

// TestMVSearchFilterLegacyDecode: an OLD-format payload (no trailer) decodes via
// the new filter-aware decoder with a zero filter (forward-read tolerance), and
// the plain DecodeMVSearchArgs still drops a filter-carrying trailer.
func TestMVSearchFilterLegacyDecode(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	legacy := EncodeMVSearchArgs("c", q, 5, 128)
	name, got, k, cpt, rc, opa, filter, _, err := DecodeMVSearchArgsOptsFilter(legacy)
	if err != nil {
		t.Fatalf("decode legacy: %v", err)
	}
	if name != "c" || k != 5 || cpt != 128 || len(got) != 2 || rc != 0 || opa != 0 || !filter.IsZero() {
		t.Errorf("legacy decode = %q k=%d cpt=%d rc=%d opa=%d filterZero=%v", name, k, cpt, rc, opa, filter.IsZero())
	}

	// Plain decoder ignores a filter-carrying trailer (trailing-bytes contract).
	withFilter := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, 0, 0, mvTestFilter(), 0)
	pname, _, pk, pcpt, perr := DecodeMVSearchArgs(withFilter)
	if perr != nil || pname != "c" || pk != 5 || pcpt != 128 {
		t.Errorf("plain decode of filter args: name=%q k=%d cpt=%d err=%v", pname, pk, pcpt, perr)
	}

	// And the rc/opa-only Opts decoder skips the filter block and still reads
	// rc/opa correctly (the ReadConsistencyOf peek relies on this).
	bothArgs := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, 2, 0, mvTestFilter(), 0)
	_, _, _, _, prc, popa, _, derr := DecodeMVSearchArgsOpts(bothArgs)
	if derr != nil || prc != 2 || popa != 0 {
		t.Errorf("opts decode over filter trailer: rc=%d opa=%d err=%v, want rc=2", prc, popa, derr)
	}
}

// TestMVSearchFilterMalformed: a corrupt filter JSON in the trailer must fail
// loud (an error), NOT be silently dropped to "match all".
func TestMVSearchFilterMalformed(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	base := EncodeMVSearchArgs("c", q, 5, 128)
	// marker=mvTrailerFilter, [filterLen=3]["{[}"] — invalid JSON.
	bad := append(append([]byte(nil), base...), mvTrailerFilter, 0, 0, 0, 3)
	bad = append(bad, '{', '[', '}')
	if _, _, _, _, _, _, _, _, err := DecodeMVSearchArgsOptsFilter(bad); err == nil {
		t.Fatal("malformed filter JSON decoded without error (should fail loud)")
	}
	// The rc/opa Opts decoder shares the path, so it must also surface the error.
	if _, _, _, _, _, _, _, err := DecodeMVSearchArgsOpts(bad); err == nil {
		t.Fatal("malformed filter JSON: DecodeMVSearchArgsOpts did not fail loud")
	}
}

// TestMVSearchFilterTruncated: a filter trailer that promises more bytes than
// present is a truncation error.
func TestMVSearchFilterTruncated(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	full := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, 1, 1, mvTestFilter(), 0)
	for _, chop := range []int{1, 5} {
		truncated := full[:len(full)-chop]
		if _, _, _, _, _, _, _, _, err := DecodeMVSearchArgsOptsFilter(truncated); !errors.Is(err, errVectorArgsTruncated) {
			t.Errorf("chop %d: err = %v, want errVectorArgsTruncated", chop, err)
		}
	}
}

// TestMVSearchReadConsistencyPeekWithFilter confirms ReadConsistencyOf still reads
// rc out of a filter-carrying MV payload (the peek delegates to the full decode).
func TestMVSearchReadConsistencyPeekWithFilter(t *testing.T) {
	q := [][]float32{{1, 0}, {0, 1}}
	args := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, ConsistencyLinearizable, 0, mvTestFilter(), 0)
	rc, ok := ReadConsistencyOf("vector_mv_search", args)
	if !ok || rc != ConsistencyLinearizable {
		t.Errorf("ReadConsistencyOf = (%d,%v), want (%d,true)", rc, ok, ConsistencyLinearizable)
	}
	// rc==0 with a filter present must still peek as 0/ok (not misclassified).
	args0 := EncodeMVSearchArgsOptsFilter("c", q, 5, 128, 0, 0, mvTestFilter(), 0)
	rc0, ok0 := ReadConsistencyOf("vector_mv_search", args0)
	if !ok0 || rc0 != 0 {
		t.Errorf("ReadConsistencyOf (filter-only) = (%d,%v), want (0,true)", rc0, ok0)
	}
}

// TestMVSearchFilterViaDispatch is the end-to-end builtin-level test: a filtered
// MV search dispatched through handleMVSearch returns ONLY matching docs.
func TestMVSearchFilterViaDispatch(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("docs", vector.MultiVectorConfig{Dim: 4})); err != nil {
		t.Fatal(err)
	}
	// Both docs have a token aligned with the query; only doc 1 matches the filter.
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", 1, [][]float32{{1, 0, 0, 0}}, vector.Metadata{"tenant": vector.NewString("acme")})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", 2, [][]float32{{1, 0, 0, 0}}, vector.Metadata{"tenant": vector.NewString("other")})); err != nil {
		t.Fatal(err)
	}

	// No filter: both docs returned.
	body, err := handleMVSearch(tx, EncodeMVSearchArgs("docs", [][]float32{{1, 0, 0, 0}}, 10, 50))
	if err != nil {
		t.Fatal(err)
	}
	res, _ := DecodeMVResults(body)
	if len(res) != 2 {
		t.Fatalf("no-filter search = %+v, want both docs", res)
	}

	// With filter tenant==acme: only doc 1.
	args := EncodeMVSearchArgsOptsFilter("docs", [][]float32{{1, 0, 0, 0}}, 10, 50, 0, 0, mvTestFilter(), 0)
	body, err = handleMVSearch(tx, args)
	if err != nil {
		t.Fatal(err)
	}
	res, err = DecodeMVResults(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("filtered search = %+v, want only doc 1", res)
	}
}

// TestMVScrollArgsRoundtrip round-trips the cursor/rc/opa/filter through
// EncodeMVScrollArgsOpts/DecodeMVScrollArgsOpts.
func TestMVScrollArgsRoundtrip(t *testing.T) {
	const (
		wantCol     = "acme/docs"
		wantLimit   = 42
		wantAfterID = uint64(123456789)
		wantRC      = ConsistencyLinearizable
		wantOPA     = uint8(1)
	)
	wantFilter := mvTestFilter()
	args := EncodeMVScrollArgsOptsBounded(wantCol, wantFilter, wantLimit, wantRC, wantOPA, wantAfterID, true, 0)
	col, filter, limit, rc, opa, afterID, hasAfter, _, err := DecodeMVScrollArgsOpts(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != wantCol || limit != wantLimit || rc != wantRC || opa != wantOPA || afterID != wantAfterID || !hasAfter {
		t.Fatalf("decode = (%q,%d,%d,%d,%d,%v), want (%q,%d,%d,%d,%d,true)",
			col, limit, rc, opa, afterID, hasAfter, wantCol, wantLimit, wantRC, wantOPA, wantAfterID)
	}
	if filter.Op != wantFilter.Op || filter.Field != wantFilter.Field || filter.Value.Str != wantFilter.Value.Str {
		t.Fatalf("filter = %+v, want %+v", filter, wantFilter)
	}
}

// TestMVScrollArgsByteIdenticalLegacy proves the no-cursor/no-rc/no-opa form is
// BYTE-IDENTICAL to the base EncodeMVScrollArgs (so the wire is unchanged for the
// legacy single-shard path).
func TestMVScrollArgsByteIdenticalLegacy(t *testing.T) {
	// No filter, no cursor, no opts.
	base := EncodeMVScrollArgs("docs", vector.Filter{}, 10)
	opts := EncodeMVScrollArgsOptsBounded("docs", vector.Filter{}, 10, 0, 0, 0, false, 0)
	if !bytes.Equal(base, opts) {
		t.Fatalf("no-trailer Opts not byte-identical:\n base=%v\n opts=%v", base, opts)
	}
	// With a filter (in the base block), still byte-identical when no cursor/opts.
	baseF := EncodeMVScrollArgs("docs", mvTestFilter(), 10)
	optsF := EncodeMVScrollArgsOptsBounded("docs", mvTestFilter(), 10, 0, 0, 0, false, 0)
	if !bytes.Equal(baseF, optsF) {
		t.Fatalf("no-trailer Opts (filtered) not byte-identical:\n base=%v\n opts=%v", baseF, optsF)
	}
	// Decoding the base must yield hasAfter=false, rc=0, opa=0.
	col, _, limit, rc, opa, afterID, hasAfter, _, err := DecodeMVScrollArgsOpts(base)
	if err != nil || col != "docs" || limit != 10 || rc != 0 || opa != 0 || afterID != 0 || hasAfter {
		t.Fatalf("decode base = (%q,%d,%d,%d,%d,%v,err=%v), want (docs,10,0,0,0,false,nil)",
			col, limit, rc, opa, afterID, hasAfter, err)
	}
}

// TestMVScrollArgsMalformed: a present cursor/opts marker with a truncated block
// is corruption — fail loud (no silent drop).
func TestMVScrollArgsMalformed(t *testing.T) {
	base := EncodeMVScrollArgs("docs", vector.Filter{}, 10)
	// marker promises a cursor (afterID:u64) but only 3 bytes follow.
	badCursor := append(append([]byte(nil), base...), mvScrollCursor, 1, 2, 3)
	if _, _, _, _, _, _, _, _, err := DecodeMVScrollArgsOpts(badCursor); !errors.Is(err, errVectorArgsTruncated) {
		t.Fatalf("truncated cursor: err = %v, want errVectorArgsTruncated", err)
	}
	// marker promises opts (rc,opa) but only 1 byte follows.
	badOpts := append(append([]byte(nil), base...), mvScrollOpts, 2)
	if _, _, _, _, _, _, _, _, err := DecodeMVScrollArgsOpts(badOpts); !errors.Is(err, errVectorArgsTruncated) {
		t.Fatalf("truncated opts: err = %v, want errVectorArgsTruncated", err)
	}
	// A malformed filter JSON in the base block fails loud.
	bad := []byte{4}
	bad = append(bad, "docs"...)
	bad = append(bad, 0, 0, 0, 10) // limit
	bad = append(bad, 0, 0, 0, 3)  // filterLen=3
	bad = append(bad, '{', '[', '}')
	if _, _, _, _, _, _, _, _, err := DecodeMVScrollArgsOpts(bad); err == nil {
		t.Fatal("malformed filter JSON decoded without error (should fail loud)")
	}
}

// TestMVScrollReadConsistencyPeek confirms ReadConsistencyOf("vector_mv_scroll", ...)
// reads rc out of the scroll trailer (the critical shard-barrier-arming coverage)
// and that legacy (no-trailer) args peek as (0,true).
func TestMVScrollReadConsistencyPeek(t *testing.T) {
	// rc=2 with a cursor + filter present.
	args := EncodeMVScrollArgsOptsBounded("docs", mvTestFilter(), 10, ConsistencyLinearizable, 0, 7, true, 0)
	if rc, ok := ReadConsistencyOf("vector_mv_scroll", args); !ok || rc != ConsistencyLinearizable {
		t.Errorf("ReadConsistencyOf(vector_mv_scroll, rc=2) = (%d,%v), want (2,true)", rc, ok)
	}
	// Legacy / no-trailer args peek as (0,true).
	legacy := EncodeMVScrollArgs("docs", vector.Filter{}, 10)
	if rc, ok := ReadConsistencyOf("vector_mv_scroll", legacy); !ok || rc != ConsistencyAnyReplica {
		t.Errorf("ReadConsistencyOf(vector_mv_scroll, legacy) = (%d,%v), want (0,true)", rc, ok)
	}
}

// TestMVScrollViaDispatch is the builtin-level test: a cursor-aware MV scroll
// dispatched through handleMVScroll pages id-ascending with the filter applied.
func TestMVScrollViaDispatch(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("docs", vector.MultiVectorConfig{Dim: 4})); err != nil {
		t.Fatal(err)
	}
	for _, id := range []uint64{3, 1, 2} {
		if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", id, [][]float32{{1, 0, 0, 0}}, nil)); err != nil {
			t.Fatal(err)
		}
	}
	// Page 1: limit 2, no cursor ⇒ {1,2}.
	body, err := handleMVScroll(tx, EncodeMVScrollArgsOptsBounded("docs", vector.Filter{}, 2, 0, 0, 0, false, 0))
	if err != nil {
		t.Fatal(err)
	}
	docs, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 || docs[0].ID != 1 || docs[1].ID != 2 {
		t.Fatalf("page1 = %+v, want ids {1,2}", docs)
	}
	// Page 2: cursor afterID=2 ⇒ {3}.
	body, err = handleMVScroll(tx, EncodeMVScrollArgsOptsBounded("docs", vector.Filter{}, 2, 0, 0, 2, true, 0))
	if err != nil {
		t.Fatal(err)
	}
	docs, err = DecodeVectorDocs(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != 3 {
		t.Fatalf("page2 = %+v, want id {3}", docs)
	}
}

// ---- MV batch get (vector_mv_get_batch) ----

// TestMVGetBatchResultRoundtrip encodes a mix of found/not-found rows with
// various projections and decodes them back, asserting ids + order + the
// token-matrix / meta projection survive the round-trip (MV has NO ttl). The MV
// clone of TestNamedGetBatchResultRoundtrip.
func TestMVGetBatchResultRoundtrip(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1), "tag": vector.NewString("x")}
	toks := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}

	rows := []MVGetBatchRow{
		{ID: 1, Found: true, Tokens: toks, Meta: meta}, // both projections
		{ID: 2, Found: false},                          // not-found
		{ID: 3, Found: true, Meta: meta},               // with_vector off (no tokens)
		{ID: 4, Found: true, Tokens: toks},             // with_payload off (no meta)
		{ID: 5, Found: true},                           // bare
	}
	got, err := DecodeMVGetBatchResult(EncodeMVGetBatchResult(rows))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(rows) {
		t.Fatalf("rows = %d, want %d", len(got), len(rows))
	}
	for i := range rows {
		if got[i].ID != rows[i].ID || got[i].Found != rows[i].Found {
			t.Errorf("row %d: got (id=%d,found=%v), want (id=%d,found=%v)", i, got[i].ID, got[i].Found, rows[i].ID, rows[i].Found)
		}
	}
	if !reflect.DeepEqual(got[0].Tokens, toks) || got[0].Meta["a"].Int != 1 {
		t.Errorf("row 0 = %+v", got[0])
	}
	if len(got[2].Tokens) != 0 || got[2].Meta == nil {
		t.Errorf("row 2 (with_vector off): %+v", got[2])
	}
	if !reflect.DeepEqual(got[3].Tokens, toks) || got[3].Meta != nil {
		t.Errorf("row 3 (with_payload off): %+v", got[3])
	}
	if len(got[4].Tokens) != 0 || got[4].Meta != nil {
		t.Errorf("row 4 (bare): %+v", got[4])
	}

	// zero rows.
	if z, err := DecodeMVGetBatchResult(EncodeMVGetBatchResult(nil)); err != nil || len(z) != 0 {
		t.Errorf("zero-row result: rows=%d err=%v", len(z), err)
	}
	// truncation (fail-loud).
	body := EncodeMVGetBatchResult(rows)
	if _, err := DecodeMVGetBatchResult(nil); !errors.Is(err, errVectorArgsTruncated) {
		t.Errorf("nil: err = %v, want errVectorArgsTruncated", err)
	}
	if _, err := DecodeMVGetBatchResult(body[:2]); !errors.Is(err, errVectorArgsTruncated) {
		t.Errorf("header chop: err = %v, want errVectorArgsTruncated", err)
	}
	if _, err := DecodeMVGetBatchResult(body[:10]); !errors.Is(err, errVectorArgsTruncated) {
		t.Errorf("mid-row chop: err = %v, want errVectorArgsTruncated", err)
	}
}

// TestMVGetBatchTwoRowOffsetAdvance is the regression guard for the
// decodeMVGetResultAt missing-offset-advance bug: row 0 carries BOTH tokens AND a
// meta payload, so the decoder must advance off past the meta bytes (off += mlen)
// before reading row 1. If that advance is missing, row 1's id + record are read
// from the MIDDLE of row 0's meta JSON — yielding a wrong id, garbage token
// count, and likely a huge allocation. This test asserts row 1 decodes EXACTLY
// (id, tokens, meta) after a meta-bearing row 0, which only holds when the offset
// advances correctly. (The single-get decoder tolerated the bug because off is
// unused after one record; the batch decoder reuses the …At helper per row.)
func TestMVGetBatchTwoRowOffsetAdvance(t *testing.T) {
	meta0 := vector.Metadata{"k": vector.NewString("a-longish-payload-value-to-shift-offset")}
	meta1 := vector.Metadata{"k2": vector.NewInt(42)}
	toks0 := [][]float32{{1, 1, 1, 1}, {2, 2, 2, 2}}
	toks1 := [][]float32{{9, 8, 7, 6}}
	rows := []MVGetBatchRow{
		{ID: 10, Found: true, Tokens: toks0, Meta: meta0},
		{ID: 11, Found: true, Tokens: toks1, Meta: meta1},
	}
	got, err := DecodeMVGetBatchResult(EncodeMVGetBatchResult(rows))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("rows = %d, want 2", len(got))
	}
	// Row 1 must be intact — the load-bearing assertion for the offset advance.
	if got[1].ID != 11 {
		t.Fatalf("row 1 id = %d, want 11 (missing off += mlen reads row 1 from mid-payload)", got[1].ID)
	}
	if !got[1].Found || !reflect.DeepEqual(got[1].Tokens, toks1) {
		t.Errorf("row 1 tokens = %+v, want %+v", got[1].Tokens, toks1)
	}
	if got[1].Meta["k2"].Int != 42 {
		t.Errorf("row 1 meta = %+v, want k2=42", got[1].Meta)
	}
	// Row 0 also intact.
	if got[0].ID != 10 || !reflect.DeepEqual(got[0].Tokens, toks0) || got[0].Meta["k"].Str != "a-longish-payload-value-to-shift-offset" {
		t.Errorf("row 0 = %+v", got[0])
	}
}

// TestHandleMVGetBatch drives handleMVGetBatch over a built MV collection: a
// mixed present/absent batch preserves input order, a partial miss is a found=0
// row (never an op error), and the projection flags gate the token matrix and the
// payload. Args reuse EncodeVectorGetBatchArgs verbatim. The MV clone of
// TestHandleNamedGetBatch (no ttl).
func TestHandleMVGetBatch(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("docs", vector.MultiVectorConfig{Dim: 4})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", 1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, vector.Metadata{"a": vector.NewInt(1)})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("docs", 2, [][]float32{{0, 0, 1, 0}}, vector.Metadata{"b": vector.NewInt(2)})); err != nil {
		t.Fatal(err)
	}

	// mixed batch: present, absent, present, absent — input order preserved.
	body, err := handleMVGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1, 99, 2, 7}, GetFlagsBoth))
	if err != nil {
		t.Fatalf("unexpected op error: %v", err)
	}
	rows, err := DecodeMVGetBatchResult(body)
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
	if len(rows[0].Tokens) != 2 || rows[0].Tokens[0][0] != 1 || rows[0].Meta["a"].Int != 1 {
		t.Errorf("row 0 found content: %+v", rows[0])
	}
	if len(rows[2].Tokens) != 1 || rows[2].Tokens[0][2] != 1 || rows[2].Meta["b"].Int != 2 {
		t.Errorf("row 2 found content: %+v", rows[2])
	}

	// projection: with_vector off -> no tokens, payload kept; with_payload off -> tokens kept, no meta.
	body, _ = handleMVGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1}, getFlagWithPayload))
	rows, _ = DecodeMVGetBatchResult(body)
	if !rows[0].Found || len(rows[0].Tokens) != 0 || rows[0].Meta["a"].Int != 1 {
		t.Errorf("with_vector off: %+v", rows[0])
	}
	body, _ = handleMVGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1}, getFlagWithVector))
	rows, _ = DecodeMVGetBatchResult(body)
	if !rows[0].Found || len(rows[0].Tokens) != 2 || rows[0].Meta != nil {
		t.Errorf("with_payload off: %+v", rows[0])
	}

	// all-absent batch -> all not-found rows, NO op error.
	body, err = handleMVGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{100, 200, 300}, GetFlagsBoth))
	if err != nil {
		t.Fatalf("all-absent: unexpected op error %v", err)
	}
	rows, _ = DecodeMVGetBatchResult(body)
	if len(rows) != 3 {
		t.Fatalf("all-absent: rows = %d, want 3", len(rows))
	}
	for i, r := range rows {
		if r.Found {
			t.Errorf("all-absent row %d: found = true, want false", i)
		}
	}

	// empty batch -> zero rows, no error.
	body, err = handleMVGetBatch(tx, EncodeVectorGetBatchArgs("docs", nil, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	if rows, _ := DecodeMVGetBatchResult(body); len(rows) != 0 {
		t.Fatalf("empty batch: rows = %d, want 0", len(rows))
	}
}

// TestMVCreateArgsFilterFirstRelativeBP mirrors TestVectorCreateCollectionArgsFilterFirstRelativeBP
// for the MV codec: non-zero bp forces the drift block which forces the OPQ/IVFTrainThreshold
// anchor chain, anchoring the trailing 4-byte word unambiguously.
func TestMVCreateArgsFilterFirstRelativeBP(t *testing.T) {
	// Use a full IVF-PQ config (IVFPQ=true) so the decoder's IVF-gated PQ sub-block
	// read (6 bytes) is satisfied by a real PQ block — the MV decoder uses a length
	// guard (not a flag) for the PQ sub-block, so bp-only (no-IVFPQ) IVF configs
	// are indistinguishable at the decoder when trailing bytes follow the IVF header.
	// This mirrors the pattern used by TestMVCreateArgsDriftRetrainRoundtrip.
	base := vector.MultiVectorConfig{
		Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true,
		IVFTrainThreshold: 1000,
	}

	// Round-trip for each bp value.
	for _, bp := range []int{1, 5000, 10000} {
		cfg := base
		cfg.FilterFirstRelativeBP = bp
		name, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/mv", cfg))
		if err != nil {
			t.Fatalf("bp=%d: decode: %v", bp, err)
		}
		if name != "acme/mv" || got != cfg {
			t.Errorf("bp=%d: roundtrip cfg = %+v, want %+v", bp, got, cfg)
		}
		if got.FilterFirstRelativeBP != bp {
			t.Errorf("bp=%d: FilterFirstRelativeBP = %d, want %d", bp, got.FilterFirstRelativeBP, bp)
		}
	}

	// Combining bp with non-zero drift + threshold + IVFPQ proves all three trailing
	// sections are present simultaneously (anchoring chain).
	combined := base
	combined.FilterFirstRelativeBP = 7500
	combined.IVFDriftRetrain = true
	combined.IVFDriftGrowthFactor = 2.5
	combined.IVFDriftFactor = 1.75
	name, gotCombined, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/mv", combined))
	if err != nil {
		t.Fatalf("combined: decode: %v", err)
	}
	if name != "acme/mv" || gotCombined != combined {
		t.Errorf("combined roundtrip = %+v, want %+v", gotCombined, combined)
	}

	// Default-off byte-identity: FilterFirstRelativeBP==0 (explicit) must encode
	// BYTE-IDENTICAL to the zero-value default (no trailing 4-byte word appended).
	withZero := base
	withZero.FilterFirstRelativeBP = 0
	withoutField := base
	if !bytes.Equal(EncodeMVCreateArgs("acme/mv", withZero), EncodeMVCreateArgs("acme/mv", withoutField)) {
		t.Fatal("MV FilterFirstRelativeBP=0 (explicit) is not byte-identical to the zero-value default")
	}
}

// TestMVCreateArgsOPQIters round-trips a non-zero OPQIters through the MV create wire
// and confirms OPQIters==0 is byte-identical to the pre-feature wire.
func TestMVCreateArgsOPQIters(t *testing.T) {
	cfg := vector.MultiVectorConfig{Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9, Quant: vector.QuantPQ, OPQ: true, OPQIters: 5}
	name, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/mv", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/mv" || got.OPQIters != 5 || !got.OPQ {
		t.Fatalf("OPQIters not carried: %+v", got)
	}
	base := vector.MultiVectorConfig{Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9, Quant: vector.QuantPQ, OPQ: true}
	zeroIters := base
	zeroIters.OPQIters = 0
	if !bytes.Equal(EncodeMVCreateArgs("acme/mv", base), EncodeMVCreateArgs("acme/mv", zeroIters)) {
		t.Fatal("OPQIters==0 not byte-identical to the pre-feature MV wire")
	}
}
