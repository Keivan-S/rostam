// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// geoBruteForce returns the set of ids whose ValueGeo "loc" metadata satisfies
// pred, computed independently over ALL points — the ground truth the geohash
// filter-first path must match exactly.
func geoBruteForce(metas map[uint64]Metadata, pred func(lat, lon float64) bool) []uint64 {
	var out []uint64
	for id, m := range metas {
		v, ok := m["loc"]
		if !ok || v.Kind != ValueGeo {
			continue
		}
		if pred(v.Lat, v.Lon) {
			out = append(out, id)
		}
	}
	return sortedIDs(out)
}

// buildGeoCorpus inserts n points with a ValueGeo "loc" field spread over a
// region around (centerLat, centerLon). Returns the hnsw, the per-id metadata,
// and the per-id vector for brute-force scoring.
func buildGeoCorpus(t *testing.T, n int, centerLat, centerLon, spreadDeg float64, seed int64) (*hnsw, map[uint64]Metadata, map[uint64][]float32) {
	t.Helper()
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(seed))
	metas := make(map[uint64]Metadata, n)
	corpus := make(map[uint64][]float32, n)
	for i := 1; i <= n; i++ {
		lat := centerLat + (rng.Float64()*2-1)*spreadDeg
		lon := centerLon + (rng.Float64()*2-1)*spreadDeg
		if lat > 89.9 {
			lat = 89.9
		}
		if lat < -89.9 {
			lat = -89.9
		}
		vec := []float32{float32(i), rng.Float32(), rng.Float32()}
		meta := Metadata{"loc": NewGeo(lat, lon)}
		if _, _, err := h.Insert(uint64(i), vec, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
		metas[uint64(i)] = meta
		corpus[uint64(i)] = vec
	}
	return h, metas, corpus
}

// TestGeohashEncodeCover proves the cell-level superset property: every point in
// a bbox maps to a cell contained in coverCells(bbox), and the encoding of a
// point lands in its expected grid cell.
func TestGeohashEncodeCover(t *testing.T) {
	// Encoding sanity: two nearby points share a cell; a far point does not.
	a := cellOf(48.8566, 2.3522)  // Paris
	b := cellOf(48.857, 2.353)    // ~100 m away
	c := cellOf(51.5074, -0.1278) // London
	if a != b {
		t.Errorf("nearby points landed in different cells: %v vs %v", a, b)
	}
	if a == c {
		t.Errorf("Paris and London unexpectedly share a cell %v", a)
	}

	// Superset at the cell level: sample many points inside a bbox and assert each
	// maps to a cell in the cover set.
	bbox := geoBBox{minLat: 48.80, minLon: 2.30, maxLat: 48.92, maxLon: 2.42}
	cells, ok := coverCells(bbox, nil)
	if !ok {
		t.Fatal("coverCells overflowed for a small bbox")
	}
	cover := make(map[geoCell]bool, len(cells))
	for _, cl := range cells {
		cover[cl] = true
	}
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 20000; i++ {
		lat := bbox.minLat + rng.Float64()*(bbox.maxLat-bbox.minLat)
		lon := bbox.minLon + rng.Float64()*(bbox.maxLon-bbox.minLon)
		if !cover[cellOf(lat, lon)] {
			t.Fatalf("point (%.6f,%.6f) inside bbox not covered by cover set", lat, lon)
		}
	}
	// Also include the exact corners.
	for _, corner := range [][2]float64{
		{bbox.minLat, bbox.minLon}, {bbox.minLat, bbox.maxLon},
		{bbox.maxLat, bbox.minLon}, {bbox.maxLat, bbox.maxLon},
	} {
		if !cover[cellOf(corner[0], corner[1])] {
			t.Fatalf("corner (%.6f,%.6f) not covered", corner[0], corner[1])
		}
	}
}

// haversineForward solves the direct geodesic problem on the sphere: the
// destination point reached from (lat,lon) by travelling r meters along the
// great circle at the given bearing (degrees). This places points on the TRUE
// haversine circle (not a flat-Earth approximation), so the bbox-containment
// test genuinely exercises the worst-case longitude extent at high latitude.
func haversineForward(lat, lon, r, bearing float64) (float64, float64) {
	const deg2rad = math.Pi / 180
	delta := r / earthRadiusM // angular distance
	phi1 := lat * deg2rad
	lam1 := lon * deg2rad
	th := bearing * deg2rad
	phi2 := math.Asin(math.Sin(phi1)*math.Cos(delta) + math.Cos(phi1)*math.Sin(delta)*math.Cos(th))
	lam2 := lam1 + math.Atan2(math.Sin(th)*math.Sin(delta)*math.Cos(phi1), math.Cos(delta)-math.Sin(phi1)*math.Sin(phi2))
	return phi2 / deg2rad, lam2 / deg2rad
}

// TestRadiusBBoxContainsCircle proves the circle→bbox over-estimate: every point
// on the TRUE haversine circle of radius R around the center is inside
// radiusBBox(center,R). Includes high-latitude cases where the longitude extent
// is widest (the worst case for under-cover).
func TestRadiusBBoxContainsCircle(t *testing.T) {
	cases := []struct {
		lat, lon, r float64
	}{
		{48.8566, 2.3522, 3000},   // Paris, 3 km
		{0, 0, 50000},             // equator, 50 km
		{60, 10, 20000},           // high latitude, 20 km
		{75, -40, 100000},         // 75°N, 100 km — wide lon extent, must not clip
		{-70, 120, 80000},         // 70°S, 80 km
		{-33.8688, 151.2093, 500}, // Sydney, 500 m
	}
	for _, c := range cases {
		bbox, ok := radiusBBox(c.lat, c.lon, c.r)
		if !ok {
			t.Fatalf("radiusBBox bailed for %v", c)
		}
		// Walk the TRUE circle in fine bearing steps; each boundary point must be
		// in the box (with a tiny epsilon for float error at the edge).
		const eps = 1e-9
		for bearing := 0.0; bearing < 360; bearing += 0.5 {
			plat, plon := haversineForward(c.lat, c.lon, c.r, bearing)
			if plat < bbox.minLat-eps || plat > bbox.maxLat+eps || plon < bbox.minLon-eps || plon > bbox.maxLon+eps {
				t.Fatalf("true circle point (%.6f,%.6f) bearing %.1f outside bbox %+v for %v", plat, plon, bearing, bbox, c)
			}
		}
	}
}

// TestGeoFilterFirstEqualsBruteForce is the correctness crux: for radius, box,
// and polygon filters that are selective enough to fire the geohash index, the
// filter-first search result EXACTLY equals brute-force ground truth computed
// with the Task-2 predicate over ALL points. Any under-cover/narrowing bug would
// drop a true match and fail here.
func TestGeoFilterFirstEqualsBruteForce(t *testing.T) {
	const n = 4000
	centerLat, centerLon := 48.8566, 2.3522 // Paris
	h, metas, corpus := buildGeoCorpus(t, n, centerLat, centerLon, 0.5, 99)
	const limit = 50_000
	q := []float32{2000, 0, 0}
	k := 30

	type tc struct {
		name string
		f    Filter
		pred func(lat, lon float64) bool
	}
	cases := []tc{
		{
			name: "radius",
			f: Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
				CenterLat: centerLat, CenterLon: centerLon, RadiusM: 8000,
			}},
			pred: func(lat, lon float64) bool {
				return haversineMeters(centerLat, centerLon, lat, lon) <= 8000
			},
		},
		{
			name: "box",
			f: Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{
				MinLat: 48.80, MinLon: 2.30, MaxLat: 48.90, MaxLon: 2.40,
			}},
			pred: func(lat, lon float64) bool {
				return pointInBox(lat, lon, 48.80, 2.30, 48.90, 2.40)
			},
		},
		{
			name: "polygon",
			f: Filter{Op: FilterGeoPolygon, Field: "loc", Geo: &GeoCondition{
				// A concave (arrow-ish) polygon around Paris center.
				Polygon: []float64{
					48.82, 2.30,
					48.90, 2.32,
					48.86, 2.36,
					48.90, 2.40,
					48.82, 2.42,
				},
			}},
			pred: func(lat, lon float64) bool {
				return pointInPolygon(lat, lon, []float64{
					48.82, 2.30, 48.90, 2.32, 48.86, 2.36, 48.90, 2.40, 48.82, 2.42,
				})
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// The geohash index must FIRE (ok==true) for these selective filters.
			cands, ok := h.payloadIdx.candidates(c.f, limit)
			if !ok {
				t.Fatalf("%s: candidates ok=false; expected geohash narrowing", c.name)
			}
			// The candidate superset must contain EVERY true match (superset invariant).
			candSet := make(map[uint32]bool, len(cands))
			for _, s := range cands {
				candSet[s] = true
			}
			truth := geoBruteForce(metas, c.pred)
			// Map true ids to slots and assert membership in the candidate set.
			for _, id := range truth {
				slot, ok := h.arena.Slot(id)
				if !ok {
					t.Fatalf("%s: id %d missing from arena", c.name, id)
				}
				if !candSet[slot] {
					t.Fatalf("%s: UNDER-COVER — true match id %d (slot %d) absent from geohash candidate set", c.name, id, slot)
				}
			}

			// Full search result == brute-force top-k.
			got := resultIDs(mustSearch(t, h, q, k, c.f))
			want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool {
				v, ok := m["loc"]
				return ok && v.Kind == ValueGeo && c.pred(v.Lat, v.Lon)
			})
			if !eqUint64(got, want) {
				t.Errorf("%s: filter-first result != brute force\n got=%v\nwant=%v", c.name, got, want)
			}
		})
	}
}

// TestGeoIndexOverflowFallsBack asserts a huge region whose covering-cell set
// exceeds the cap makes candidates() return ok=false (graph fallback), and that
// graph traversal still returns correct results.
func TestGeoIndexOverflowFallsBack(t *testing.T) {
	const n = 1500
	h, metas, corpus := buildGeoCorpus(t, n, 0, 0, 40, 5)
	const limit = 50_000

	// A radius spanning thousands of km → bbox spans many degrees → cover cells
	// blow past geoMaxCoverCells.
	huge := Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
		CenterLat: 0, CenterLon: 0, RadiusM: 3_000_000, // 3000 km
	}}
	if _, ok := h.payloadIdx.candidates(huge, limit); ok {
		t.Fatal("huge radius: expected candidates ok=false (overflow → graph), got ok=true")
	}

	// A huge bounding box too.
	hugeBox := Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{
		MinLat: -80, MinLon: -170, MaxLat: 80, MaxLon: 170,
	}}
	if _, ok := h.payloadIdx.candidates(hugeBox, limit); ok {
		t.Fatal("huge box: expected candidates ok=false (overflow → graph), got ok=true")
	}

	// Graph traversal must still be exact for the huge radius.
	q := []float32{750, 0, 0}
	k := 40
	got := resultIDs(mustSearch(t, h, q, k, huge))
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool {
		v, ok := m["loc"]
		return ok && v.Kind == ValueGeo && haversineMeters(0, 0, v.Lat, v.Lon) <= 3_000_000
	})
	if !eqUint64(got, want) {
		t.Errorf("graph fallback for huge radius != brute force\n got=%v\nwant=%v", got, want)
	}
}

// TestGeoIndexReindexReclaimSnapshot verifies the geohash index stays correct
// after Delete→Reclaim (incremental drop + reuse) and after Snapshot→Restore
// (full rebuild), with geo search still exact.
func TestGeoIndexReindexReclaimSnapshot(t *testing.T) {
	const n = 600
	centerLat, centerLon := 40.0, -74.0 // NYC-ish
	h, metas, corpus := buildGeoCorpus(t, n, centerLat, centerLon, 0.4, 13)
	const limit = 50_000
	q := []float32{300, 0, 0}
	k := 25

	radius := Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
		CenterLat: centerLat, CenterLon: centerLon, RadiusM: 6000,
	}}
	pred := func(lat, lon float64) bool {
		return haversineMeters(centerLat, centerLon, lat, lon) <= 6000
	}
	matchMeta := func(m Metadata) bool {
		v, ok := m["loc"]
		return ok && v.Kind == ValueGeo && pred(v.Lat, v.Lon)
	}

	// Delete the first 200 ids, reclaim, reinsert 200 new ids reusing slots.
	for i := 1; i <= 200; i++ {
		h.Delete(uint64(i), CASCond{})
		delete(metas, uint64(i))
		delete(corpus, uint64(i))
	}
	h.Reclaim()
	rng := rand.New(rand.NewSource(21))
	for i := n + 1; i <= n+200; i++ {
		lat := centerLat + (rng.Float64()*2-1)*0.4
		lon := centerLon + (rng.Float64()*2-1)*0.4
		vec := []float32{float32(i), rng.Float32(), rng.Float32()}
		meta := Metadata{"loc": NewGeo(lat, lon)}
		if _, _, err := h.Insert(uint64(i), vec, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
		metas[uint64(i)] = meta
		corpus[uint64(i)] = vec
	}

	// After Reclaim + reuse, the geohash filter-first must still fire and be exact.
	if _, ok := h.payloadIdx.candidates(radius, limit); !ok {
		t.Fatal("after reclaim: geohash candidates ok=false")
	}
	got := resultIDs(mustSearch(t, h, q, k, radius))
	want := bruteForceFiltered(corpus, metas, q, k, matchMeta)
	if !eqUint64(got, want) {
		t.Errorf("after reclaim: geo search != brute force\n got=%v\nwant=%v", got, want)
	}

	// Snapshot → Restore (rebuild path) and re-check.
	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	dst, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatal(err)
	}
	if _, ok := dst.payloadIdx.candidates(radius, limit); !ok {
		t.Fatal("after restore: geohash candidates ok=false (rebuild lost the geo index)")
	}
	got2 := resultIDs(mustSearch(t, dst, q, k, radius))
	if !eqUint64(got2, want) {
		t.Errorf("after restore: geo search != brute force\n got=%v\nwant=%v", got2, want)
	}
}

// TestGeoFieldDoesNotBreakScalarIndex verifies a collection carrying BOTH geo
// fields and scalar/range fields keeps the scalar eq/range index working (no
// regression from the geo path), and that geo values never leak into the scalar
// structures.
func TestGeoFieldDoesNotBreakScalarIndex(t *testing.T) {
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	const limit = 50_000
	for i := 1; i <= 200; i++ {
		meta := Metadata{
			"loc":   NewGeo(48.85+float64(i)*0.001, 2.35+float64(i)*0.001),
			"cat":   NewInt(int64(i % 4)),
			"price": NewFloat(float64(i)),
		}
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	// Geo values must NOT appear in the scalar field index.
	if _, ok := h.payloadIdx.fields["loc"]; ok {
		t.Error("geo field 'loc' leaked into the scalar eq/range index")
	}
	if _, ok := h.payloadIdx.geo["loc"]; !ok {
		t.Error("geo field 'loc' missing from the geohash index")
	}

	// Scalar equality still narrows.
	eq := Filter{Op: FilterEq, Field: "cat", Value: NewInt(2)}
	cands, ok := h.payloadIdx.candidates(eq, limit)
	if !ok {
		t.Fatal("scalar eq: candidates ok=false (geo path broke scalar index)")
	}
	for _, slot := range cands {
		id := h.arena.ID(slot)
		if id%4 != 2 {
			t.Errorf("cat==2 candidate id %d has cat %d", id, id%4)
		}
	}

	// Scalar range still narrows.
	rng := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterGte, Field: "price", Value: NewFloat(50)},
		{Op: FilterLt, Field: "price", Value: NewFloat(60)},
	}}
	if _, ok := h.payloadIdx.candidates(rng, limit); !ok {
		t.Fatal("scalar range: candidates ok=false (geo path broke range index)")
	}

	// And(geo, scalar-eq) narrows on both — assert it fires and stays exact.
	mixed := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{MinLat: 48.85, MinLon: 2.35, MaxLat: 48.95, MaxLon: 2.45}},
		{Op: FilterEq, Field: "cat", Value: NewInt(1)},
	}}
	mcands, ok := h.payloadIdx.candidates(mixed, limit)
	if !ok {
		t.Fatal("And(geo, eq): candidates ok=false")
	}
	for _, slot := range mcands {
		id := h.arena.ID(slot)
		if id%4 != 1 {
			t.Errorf("And(geo, cat==1) candidate id %d has cat %d", id, id%4)
		}
	}
}
