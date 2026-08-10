// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// earthRadiusM is the mean Earth radius used for the spherical (haversine)
// great-circle distance. The geo conventions are WGS84 degrees but distance is
// a spherical approximation (geodesic/ellipsoidal distance is a documented
// non-goal).
const earthRadiusM = 6_371_000.0

// haversineMeters returns the great-circle distance in meters between two
// WGS84 points given in degrees, using the haversine formula on a sphere of
// radius earthRadiusM.
func haversineMeters(lat1, lon1, lat2, lon2 float64) float64 {
	const deg2rad = math.Pi / 180
	φ1 := lat1 * deg2rad
	φ2 := lat2 * deg2rad
	dφ := (lat2 - lat1) * deg2rad
	dλ := (lon2 - lon1) * deg2rad

	sinDφ := math.Sin(dφ / 2)
	sinDλ := math.Sin(dλ / 2)
	a := sinDφ*sinDφ + math.Cos(φ1)*math.Cos(φ2)*sinDλ*sinDλ
	// Clamp against float rounding pushing a slightly above 1 for near-antipodal
	// points, which would make sqrt(1-a) NaN (a can never legitimately exceed 1).
	a = math.Min(1, a)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return earthRadiusM * c
}

// pointInBox reports whether (lat,lon) lies within the axis-aligned bounding
// box [minLat,maxLat] x [minLon,maxLon]. Bounds are INCLUSIVE on all edges and
// corners. Antimeridian-crossing boxes (minLon > maxLon) are a documented
// non-goal and are rejected at compile time, so this assumes minLon <= maxLon.
func pointInBox(lat, lon, minLat, minLon, maxLat, maxLon float64) bool {
	return lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon
}

// pointInPolygon reports whether (lat,lon) lies inside the simple polygon whose
// exterior ring is the flat slice poly = [lat0,lon0, lat1,lon1, ...] (the ring
// is implicitly closed; the last vertex connects back to the first). It uses
// the even-odd (ray-casting) rule and therefore handles concave polygons
// correctly. Treating lat as the Y axis and lon as the X axis, it casts a ray
// in the +X (increasing lon) direction and counts edge crossings.
//
// On-edge behavior: a point lying exactly on an edge is classified by the
// strict ">" / half-open crossing test below — it is deterministic but
// convention-dependent (a vertex/edge may count for one bordering polygon and
// not the other). Callers needing robust boundary semantics should not rely on
// the on-edge result. The caller guarantees len(poly) is even and >= 6.
func pointInPolygon(lat, lon float64, poly []float64) bool {
	inside := false
	n := len(poly) / 2
	// j is the previous vertex index (wraps to the last vertex for i == 0),
	// forming the edge j->i.
	j := n - 1
	for i := 0; i < n; i++ {
		yi, xi := poly[2*i], poly[2*i+1] // vertex i: lat=y, lon=x
		yj, xj := poly[2*j], poly[2*j+1] // vertex j
		// Does the edge (j->i) straddle the horizontal ray at y=lat, and is the
		// crossing point to the right (greater lon) of the test point?
		if (yi > lat) != (yj > lat) {
			xCross := (xj-xi)*(lat-yi)/(yj-yi) + xi
			if lon < xCross {
				inside = !inside
			}
		}
		j = i
	}
	return inside
}
