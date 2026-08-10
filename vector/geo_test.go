// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

// TestHaversineMeters checks the great-circle distance helper against a known
// fixture (Paris -> London ~343 km) plus the degenerate zero-distance case.
func TestHaversineMeters(t *testing.T) {
	const (
		parisLat, parisLon   = 48.8566, 2.3522
		londonLat, londonLon = 51.5074, -0.1278
	)
	d := haversineMeters(parisLat, parisLon, londonLat, londonLon)
	// Reference distance is ~343 km. Allow a small tolerance for the spherical
	// approximation.
	const wantKm = 343.0
	if math.Abs(d/1000-wantKm) > 5 {
		t.Errorf("haversine(Paris,London) = %.1f m (%.1f km), want ~%.0f km", d, d/1000, wantKm)
	}
	// Same point -> zero distance.
	if z := haversineMeters(parisLat, parisLon, parisLat, parisLon); z != 0 {
		t.Errorf("haversine(p,p) = %v, want 0", z)
	}
	// Symmetric.
	rev := haversineMeters(londonLat, londonLon, parisLat, parisLon)
	if math.Abs(rev-d) > 1e-6 {
		t.Errorf("haversine not symmetric: %v vs %v", d, rev)
	}
	// Antipodal points: half the great circle ≈ π·R, and must NOT be NaN (the
	// `a = min(1, a)` clamp guards against float rounding pushing a above 1).
	anti := haversineMeters(0, 0, 0, 180)
	if math.IsNaN(anti) {
		t.Fatal("haversine(antipodal) = NaN (a-clamp missing)")
	}
	wantAnti := math.Pi * earthRadiusM
	if math.Abs(anti-wantAnti) > 1 {
		t.Errorf("haversine(antipodal) = %.1f m, want ~%.1f m", anti, wantAnti)
	}
}

func TestPointInBox(t *testing.T) {
	// Box: SW (10,20) -> NE (30,40).
	const minLat, minLon, maxLat, maxLon = 10.0, 20.0, 30.0, 40.0
	cases := []struct {
		name     string
		lat, lon float64
		want     bool
	}{
		{"interior", 20, 30, true},
		{"outside north", 31, 30, false},
		{"outside east", 20, 41, false},
		{"outside south", 9, 30, false},
		{"outside west", 20, 19, false},
		{"on SW corner inclusive", 10, 20, true},
		{"on NE corner inclusive", 30, 40, true},
		{"on south edge inclusive", 10, 30, true},
		{"on east edge inclusive", 20, 40, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := pointInBox(c.lat, c.lon, minLat, minLon, maxLat, maxLon); got != c.want {
				t.Errorf("pointInBox(%v,%v) = %v, want %v", c.lat, c.lon, got, c.want)
			}
		})
	}
}

// concavePoly is an L-shaped concave polygon. Flat lat,lon ring (lat is the Y
// axis, lon is the X axis):
//
//	(0,0) - (0,10) - (10,10) - (10,6) - (4,6) - (4,0) - back to (0,0)
//
// The shape is a full lower band (lat 0..4 across lon 0..10) plus a right tower
// (lon 6..10 up to lat 10). The upper-LEFT rectangle lat∈(4,10], lon∈[0,6) is
// the "notch" — cut OUT of the shape, so points there are OUTSIDE. A convex
// hull would (wrongly) include the notch; even-odd ray-casting excludes it.
var concavePoly = []float64{
	0, 0,
	0, 10,
	10, 10,
	10, 6,
	4, 6,
	4, 0,
}

func TestPointInPolygonConcave(t *testing.T) {
	cases := []struct {
		name     string
		lat, lon float64
		want     bool
	}{
		{"inside lower band", 2, 5, true},
		{"inside right tower high", 8, 8, true},
		{"inside lower band right", 2, 9, true},
		{"in the notch (upper-left) -> outside", 8, 2, false}, // concavity proof: even-odd
		{"far outside", 50, 50, false},
		{"below shape", -1, 5, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := pointInPolygon(c.lat, c.lon, concavePoly); got != c.want {
				t.Errorf("pointInPolygon(%v,%v) = %v, want %v", c.lat, c.lon, got, c.want)
			}
		})
	}
}

// TestPointInPolygonOnEdge documents the (consistent) on-edge behavior of the
// ray-casting implementation. On-edge results are deterministic; we only assert
// that evaluation does not panic and is stable across repeated calls.
func TestPointInPolygonOnEdge(t *testing.T) {
	square := []float64{0, 0, 0, 10, 10, 10, 10, 0}
	a := pointInPolygon(5, 0, square) // on the west edge
	b := pointInPolygon(5, 0, square)
	if a != b {
		t.Errorf("on-edge classification not deterministic: %v vs %v", a, b)
	}
	// Clear interior / exterior are unambiguous regardless of edge convention.
	if !pointInPolygon(5, 5, square) {
		t.Error("center of square should be inside")
	}
	if pointInPolygon(5, 20, square) {
		t.Error("far point should be outside")
	}
}
