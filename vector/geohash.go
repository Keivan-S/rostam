// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// Geohash-prefix spatial index support.
//
// A geohash recursively bisects the lat [-90,90] / lon [-180,180] rectangle,
// interleaving lon and lat bits (lon first), and groups every 5 bits into a
// base-32 character. We bucket geo points by a FIXED-LENGTH geohash prefix
// (the "cell"); all points whose prefix matches share a candidate posting list.
//
// CELL SIZE (the chosen precision is geohashPrecision = 5 characters = 25 bits,
// 13 lon-bits and 12 lat-bits):
//
//	lon cell width  = 360 / 2^13 = 0.043945 deg  (~4.9 km at the equator)
//	lat cell height = 180 / 2^12 = 0.043945 deg  (~4.9 km everywhere)
//
// So a precision-5 cell is about 4.9 km x 4.9 km near the equator (narrower in
// lon toward the poles). This is the standard "geohash length 5 ≈ 4.9km" figure.
//
// SUPERSET INVARIANT: candidate generation reduces a query region to its
// lat/lon bounding box, then enumerates EVERY cell of the fixed grid that the
// bbox touches (rounding the box outward to whole cells). The union of those
// cells' posting lists is therefore a superset of every point inside the bbox,
// hence a superset of every true predicate match (the predicate re-checks for
// exactness). Over-cover is always safe; we never under-cover.

const (
	// geohashPrecision is the fixed geohash prefix length (characters) used by
	// the spatial index. 5 chars = 25 bits ≈ 4.9 km cells (see file comment).
	geohashPrecision = 5

	// geohashBits is the total number of grid bits at geohashPrecision.
	geohashBits = geohashPrecision * 5 // 25

	// lonBits / latBits split geohashBits. Geohash interleaves lon first, so with
	// an odd total bit count lon gets the extra bit: lon=13, lat=12.
	lonBits = (geohashBits + 1) / 2 // 13
	latBits = geohashBits / 2       // 12

	// lonCells / latCells are the number of grid columns / rows at this precision.
	lonCells = 1 << lonBits // 8192
	latCells = 1 << latBits // 4096

	// lonCellDeg / latCellDeg are the cell dimensions in degrees.
	lonCellDeg = 360.0 / float64(lonCells)
	latCellDeg = 180.0 / float64(latCells)

	// geoMaxCoverCells caps the number of covering cells the candidate path will
	// enumerate. Beyond this the query region is too large for the index to be
	// selective, so candidates() bails to ok=false → graph traversal. This bounds
	// both the enumeration work and the candidate-union size.
	geoMaxCoverCells = 4096
)

// geoCell is the integer (col,row) coordinate of a grid cell at the index
// precision. col indexes lon in [0,lonCells), row indexes lat in [0,latCells).
// A (col,row) pair uniquely identifies the geohash prefix, so it is the map key
// of the spatial index (cheaper than the base-32 string and equally exact).
type geoCell struct {
	col uint32 // lon bucket
	row uint32 // lat bucket
}

// lonCol maps a longitude in [-180,180] to its grid column [0,lonCells).
// Values are clamped into range; the right edge (lon == 180) maps to the last
// column rather than overflowing.
func lonCol(lon float64) uint32 {
	if lon <= -180 {
		return 0
	}
	if lon >= 180 {
		return lonCells - 1
	}
	c := int64(math.Floor((lon + 180) / lonCellDeg))
	if c < 0 {
		return 0
	}
	if c >= lonCells {
		return lonCells - 1
	}
	return uint32(c)
}

// latRow maps a latitude in [-90,90] to its grid row [0,latCells), clamped.
func latRow(lat float64) uint32 {
	if lat <= -90 {
		return 0
	}
	if lat >= 90 {
		return latCells - 1
	}
	r := int64(math.Floor((lat + 90) / latCellDeg))
	if r < 0 {
		return 0
	}
	if r >= latCells {
		return latCells - 1
	}
	return uint32(r)
}

// cellOf returns the grid cell that contains the point (lat,lon).
func cellOf(lat, lon float64) geoCell {
	return geoCell{col: lonCol(lon), row: latRow(lat)}
}

// geoBBox is an axis-aligned lat/lon bounding box (degrees), SW->NE, used for
// candidate cell enumeration. It is always non-antimeridian-crossing
// (minLon <= maxLon) because callers either build it from a non-crossing box or
// bail before producing one that would cross.
type geoBBox struct {
	minLat, minLon, maxLat, maxLon float64
}

// coverCells enumerates every grid cell that the bbox touches, appending them to
// dst. It rounds the box OUTWARD to whole cells (floor on the min corner via the
// containing cell, ceil on the max corner via the containing cell of the max
// coordinate), guaranteeing the returned set is a superset of every point in the
// box (the SUPERSET INVARIANT at the cell level). Returns ok=false if the number
// of covering cells would exceed geoMaxCoverCells (the overflow bail) — the
// caller then falls back to graph traversal.
//
// The caller guarantees the bbox has been clamped to valid ranges
// (lat in [-90,90], lon in [-180,180]) and does not cross the antimeridian.
func coverCells(b geoBBox, dst []geoCell) ([]geoCell, bool) {
	colLo := lonCol(b.minLon)
	colHi := lonCol(b.maxLon)
	rowLo := latRow(b.minLat)
	rowHi := latRow(b.maxLat)
	// lonCol/latRow already floor to the containing cell, so colLo/rowLo are the
	// cells holding the min corner and colHi/rowHi the cells holding the max
	// corner. Because the grid columns/rows are monotonic in lon/lat, every cell
	// in [colLo,colHi] x [rowLo,rowHi] is touched by (or fully inside) the box,
	// and no point of the box falls outside that inclusive rectangle.
	ncols := int(colHi-colLo) + 1
	nrows := int(rowHi-rowLo) + 1
	if ncols*nrows > geoMaxCoverCells {
		return dst, false
	}
	for c := colLo; c <= colHi; c++ {
		for r := rowLo; r <= rowHi; r++ {
			dst = append(dst, geoCell{col: c, row: r})
		}
	}
	return dst, true
}

// radiusBBox returns a lat/lon bounding box that FULLY CONTAINS the haversine
// circle of radius radiusM (meters) centered at (lat,lon), or ok=false if the
// query is unsafe to index (radius reaches/crosses a pole, or the lon span would
// wrap the antimeridian) — in which case the caller bails to graph traversal so
// correctness is preserved (a documented over-cover-or-bail policy).
//
// THE SUPERSET MATH. The box half-extents must be a conservative OVER-estimate
// so the box never clips the circle:
//
//   - Latitude: 1 deg latitude ≈ earthRadiusM * pi/180 meters everywhere
//     (a meridian is a great circle). dLat = radiusM / metersPerLatDeg, where
//     metersPerLatDeg = earthRadiusM * pi/180. No latitude dependence.
//
//   - Longitude: 1 deg longitude ≈ metersPerLatDeg * cos(lat) meters, which
//     SHRINKS toward the poles, so dLon = radiusM / (metersPerLatDeg * cos(lat))
//     GROWS. To guarantee the box contains the whole circle we must use the
//     SMALLEST cos(lat) over the circle's latitude span (the latitude nearest a
//     pole), giving the LARGEST dLon. We use cos at maxAbsLat = the box's lat
//     edge farthest from the equator (|lat| + dLat), which is a safe upper bound
//     on dLon for every point of the circle. cos is evaluated on the clamped
//     [0,90) magnitude. If that latitude reaches a pole (>= 90) cos -> 0 and dLon
//     -> infinity, so we BAIL (graph fallback) rather than risk under-cover.
func radiusBBox(lat, lon, radiusM float64) (geoBBox, bool) {
	const deg2rad = math.Pi / 180
	metersPerLatDeg := earthRadiusM * deg2rad // meters per degree of latitude

	dLat := radiusM / metersPerLatDeg
	minLat := lat - dLat
	maxLat := lat + dLat
	// If the circle reaches a pole the longitude bbox degenerates (all lons are
	// within radius near a pole). Bail to graph traversal — pole handling is a
	// documented non-goal for the index.
	if minLat <= -90 || maxLat >= 90 {
		return geoBBox{}, false
	}

	// Largest |lat| on the box's lat span gives the smallest cos -> largest dLon.
	maxAbsLat := math.Max(math.Abs(minLat), math.Abs(maxLat))
	cosLat := math.Cos(maxAbsLat * deg2rad)
	if cosLat <= 0 {
		return geoBBox{}, false // degenerate near a pole
	}
	dLon := radiusM / (metersPerLatDeg * cosLat)
	// A safe over-estimate that nonetheless spans >= 180 deg of longitude could
	// wrap the antimeridian; bail rather than build a crossing box (non-goal).
	if dLon >= 180 {
		return geoBBox{}, false
	}
	minLon := lon - dLon
	maxLon := lon + dLon
	if minLon < -180 || maxLon > 180 {
		return geoBBox{}, false // would cross the antimeridian; bail to graph
	}
	return geoBBox{minLat: minLat, minLon: minLon, maxLat: maxLat, maxLon: maxLon}, true
}

// polygonBBox returns the axis-aligned bounding box of a flat lat,lon,... ring.
// The caller guarantees len(poly) is even and >= 6 (validated at compile time).
func polygonBBox(poly []float64) geoBBox {
	minLat, minLon := poly[0], poly[1]
	maxLat, maxLon := poly[0], poly[1]
	for i := 0; i < len(poly); i += 2 {
		lat, lon := poly[i], poly[i+1]
		if lat < minLat {
			minLat = lat
		}
		if lat > maxLat {
			maxLat = lat
		}
		if lon < minLon {
			minLon = lon
		}
		if lon > maxLon {
			maxLon = lon
		}
	}
	return geoBBox{minLat: minLat, minLon: minLon, maxLat: maxLat, maxLon: maxLon}
}
