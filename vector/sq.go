// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// trainedSQ is a TRAINED, metric-agnostic scalar quantizer. Unlike the legacy
// fixed-scale sq8 (a hardcoded 1/127 Cosine fast-path), trainedSQ learns a
// per-dimension [min, max] range from a build sample and quantizes each
// component to a bits-wide level inside its own range. Distance is computed
// asymmetrically under the index's ACTUAL metric (Cosine / L2 / DotProduct):
// the query stays in exact float32, the stored code is dequantized, and the
// configured distFunc is applied. Because HNSW always rescores the final
// candidate set on exact float32 (see hnsw.rescore), this approximate distance
// only needs to ORDER graph traversal — correctness here is RECALL, not
// bit-exact distance.
//
// bits is 8 (one byte per dimension, levelMax = 255). Later work
// generalizes bits to {4, 6, 8}; the level math already reads levelMax, so only
// the byte packing (CodeLen / level read-write) changes there.
//
// An UNTRAINED trainedSQ (min == nil) has no learned ranges: Encode is a no-op
// and trained() is false, so the HNSW owner navigates on EXACT float32 until the
// build-time auto-train swaps in learned ranges (mirrors the pqQuantizer
// untrained=exact policy).
type trainedSQ struct {
	dim    int
	bits   int
	metric Metric
	// min/max are the learned per-dimension ranges (len == dim each), or nil when
	// untrained. inv[i] = 1/(max[i]-min[i]) precomputed for Encode; for a constant
	// dimension (max==min) the range is widened to sqEps so inv is finite and the
	// level collapses to 0 (no NaN/Inf).
	min  []float32
	max  []float32
	inv  []float32
	dist distFunc
}

// sqEps is the floor range used for a constant dimension (max == min in the
// training sample): it keeps 1/(max-min) finite and maps every value of that
// dimension to level 0 rather than producing NaN/Inf.
const sqEps = float32(1e-6)

// newTrainedSQ returns an UNTRAINED trained scalar quantizer for dim/bits/metric.
// bits must be 8 (the factory enforces SQBits ∈ {0,8}). The learned
// ranges are populated later by setRanges (the build-time auto-train) or
// loadRanges (snapshot/persist restore); until then trained() is false and the
// owner uses exact float32.
func newTrainedSQ(dim, bits int, metric Metric) *trainedSQ {
	if bits <= 0 {
		bits = 8
	}
	return &trainedSQ{dim: dim, bits: bits, metric: metric, dist: pickDist(metric)}
}

// levelMax returns the maximum quantization level (2^bits - 1). For bits == 8
// this is 255 (one byte per dimension); 15 for bits == 4; 63 for bits == 6.
func (q *trainedSQ) levelMax() float32 { return float32(int(1<<uint(q.bits)) - 1) }

// CodeLen returns the encoded length in bytes: ceil(dim*bits/8). For bits == 8
// this is one byte per dimension (the Task-1 fast path); for 4/6 the dim
// bits-wide levels are packed LSB-first into a bit stream, so the final byte may
// be partial (e.g. dim=10 at 6-bit ⇒ 60 bits ⇒ 8 bytes, top 4 bits of the last
// byte unused).
func (q *trainedSQ) CodeLen() int { return (q.dim*q.bits + 7) / 8 }

// packLevel writes a single bits-wide level into dst at logical dimension index i
// using an LSB-first bit stream: dimension i occupies bits [i*bits, (i+1)*bits),
// little-endian within and across bytes. Only valid for the generic (non-8) path;
// the 8-bit path writes dst[i] directly. level must already be clamped to
// [0, levelMax].
func (q *trainedSQ) packLevel(dst []byte, i int, level uint32) {
	bitPos := i * q.bits
	for b := 0; b < q.bits; b++ {
		if level&(1<<uint(b)) != 0 {
			p := bitPos + b
			dst[p>>3] |= 1 << uint(p&7)
		}
	}
}

// unpackLevel reads the bits-wide level at logical dimension index i from the
// LSB-first bit stream written by packLevel. Inverse of packLevel; only used on
// the generic (non-8) path.
func (q *trainedSQ) unpackLevel(code []byte, i int) uint32 {
	bitPos := i * q.bits
	var level uint32
	for b := 0; b < q.bits; b++ {
		p := bitPos + b
		if code[p>>3]&(1<<uint(p&7)) != 0 {
			level |= 1 << uint(b)
		}
	}
	return level
}

// trained reports whether the per-dimension ranges have been learned. An
// untrained quantizer cannot Encode or score; the HNSW owner checks this and
// navigates on EXACT float32 until the build-time train hook swaps ranges in
// (mirrors pqQuantizer.trained()).
func (q *trainedSQ) trained() bool { return q.min != nil }

// setRanges installs the learned per-dimension ranges (from trainSQ) and
// precomputes the inverse spans. Called once by the build-time auto-train under
// h.mu (write lock). After it returns trained() is true.
func (q *trainedSQ) setRanges(min, max []float32) {
	q.min = min
	q.max = max
	q.inv = make([]float32, q.dim)
	for i := 0; i < q.dim; i++ {
		span := max[i] - min[i]
		if span < sqEps {
			span = sqEps
		}
		q.inv[i] = 1.0 / span
	}
}

// loadRangesBits rebuilds the quantizer from serialized ranges (snapshot/persist
// restore) plus an explicit bit-depth override from the
// serialized block. Defense-in-depth: even if the rebuilt config carried the
// wrong SQBits (e.g. a snapshot that predates carrying SQBits in its per-
// collection config), the codes still decode correctly because the serialized
// bits — written alongside the ranges by the SAME trained quantizer — win. A
// non-positive bits is ignored (keeps the factory default), and a valid value
// always AGREES with cfg.SQBits on a correctly-configured restore.
func (q *trainedSQ) loadRangesBits(min, max []float32, bits int) {
	if bits > 0 {
		q.bits = bits
	}
	q.setRanges(min, max)
}

// Encode writes the per-dimension level code for vec into dst (len == CodeLen()).
// SAFE before training: an untrained quantizer (nil ranges) leaves dst untouched
// (the arena's auto-encode writes a zero placeholder, never scored — the scorer
// uses exact float32 until trained), mirroring pqQuantizer.Encode.
func (q *trainedSQ) Encode(dst []byte, vec []float32) {
	if q.min == nil {
		return
	}
	lvlMax := q.levelMax()
	if q.bits == 8 {
		// Byte-aligned fast path (one byte per dimension): byte-identical and
		// speed-equivalent. Do NOT route 8-bit through the generic bit
		// stream — existing QuantSQ-8 codes/wire must stay unchanged.
		for i := 0; i < q.dim; i++ {
			dst[i] = byte(q.level(vec[i], i, lvlMax)) //nolint:gosec // clamped to [0,255]
		}
		return
	}
	// Generic 4/6-bit path: clear the (partial) code then pack levels LSB-first.
	for i := range dst {
		dst[i] = 0
	}
	for i := 0; i < q.dim; i++ {
		q.packLevel(dst, i, q.level(vec[i], i, lvlMax))
	}
}

// level quantizes component vec[i] of dimension i to its bits-wide level:
// clamp(round((x - min[i]) * inv[i] * levelMax), 0, levelMax). Shared by the
// 8-bit fast path and the generic packer.
func (q *trainedSQ) level(x float32, i int, lvlMax float32) uint32 {
	l := math.Round(float64((x - q.min[i]) * q.inv[i] * lvlMax))
	if l < 0 {
		l = 0
	} else if l > float64(lvlMax) {
		l = float64(lvlMax)
	}
	return uint32(l) //nolint:gosec // l is clamped to [0, levelMax]
}

// deq dequantizes level for dimension i back to an approximate float32:
// min[i] + level/levelMax * (max[i]-min[i]). For a constant dimension the
// learned range is degenerate (max==min) so this returns min[i] exactly. The
// level is the bits-wide quantization level (0..levelMax) regardless of bit-depth.
func (q *trainedSQ) deq(level uint32, i int) float32 {
	return q.min[i] + float32(level)/q.levelMax()*(q.max[i]-q.min[i])
}

// readLevel extracts the bits-wide level for dimension i from a code: the
// byte-aligned fast path for 8-bit (code[i]), the LSB-first bit stream otherwise.
func (q *trainedSQ) readLevel(code []byte, i int) uint32 {
	if q.bits == 8 {
		return uint32(code[i])
	}
	return q.unpackLevel(code, i)
}

// sqQuery is the per-search handle produced by PrepareQuery: the exact float32
// query (asymmetric distance — the query keeps full precision) plus a reusable
// per-search scratch buffer the Distance loop dequantizes the code into. The
// scratch is owned by the single search goroutine, so it is allocation-free
// across the thousands of Distance calls in one traversal without racing.
type sqQuery struct {
	q       []float32
	scratch []float32
}

// PrepareQuery returns the asymmetric query handle: the exact float32 query and a
// dim-sized scratch buffer for the per-call dequantize. The caller must not
// mutate q for the search's duration.
func (q *trainedSQ) PrepareQuery(query []float32) queryCode {
	return &sqQuery{q: query, scratch: make([]float32, q.dim)}
}

// Distance returns the approximate distance between the prepared float32 query
// and a stored code under the index metric: the code is dequantized into the
// per-search scratch and the configured distFunc is applied (Cosine: 1-dot; L2:
// Σ(q-deq)²; DotProduct: -dot). Smaller = nearer, matching distFunc semantics.
// Asymmetric: the query side keeps full float32 precision.
func (q *trainedSQ) Distance(qc queryCode, code []byte) float32 {
	h := qc.(*sqQuery)
	s := h.scratch
	for i := 0; i < q.dim; i++ {
		s[i] = q.deq(q.readLevel(code, i), i)
	}
	return q.dist(h.q, s)
}

// CodeDistance returns the symmetric approximate distance between two stored
// codes: BOTH sides are dequantized and the configured distFunc is applied. Used
// by the quantized build path (neighbor selection over codes). It dequantizes
// inline into two stack-escaping-free local buffers per call; the build path is
// already O(n log n) distance calls, and v1 favors correctness/recall over the
// SIMD kernel (a follow-up).
func (q *trainedSQ) CodeDistance(a, b []byte) float32 {
	da := make([]float32, q.dim)
	db := make([]float32, q.dim)
	for i := 0; i < q.dim; i++ {
		da[i] = q.deq(q.readLevel(a, i), i)
		db[i] = q.deq(q.readLevel(b, i), i)
	}
	return q.dist(da, db)
}

// trainSQ learns the per-dimension [min, max] ranges over sample and returns a
// TRAINED trainedSQ for dim/bits/metric. A constant dimension (min==max across
// the whole sample) keeps min==max here; setRanges widens its span to sqEps so
// Encode never divides by zero. An empty sample yields a nil-range (untrained)
// quantizer (the caller skips training when there is nothing to learn from).
func trainSQ(sample [][]float32, dim, bits int, metric Metric) *trainedSQ {
	q := newTrainedSQ(dim, bits, metric)
	if len(sample) == 0 {
		return q
	}
	min := make([]float32, dim)
	max := make([]float32, dim)
	for i := 0; i < dim; i++ {
		min[i] = float32(math.Inf(1))
		max[i] = float32(math.Inf(-1))
	}
	for _, v := range sample {
		for i := 0; i < dim; i++ {
			x := v[i]
			if x < min[i] {
				min[i] = x
			}
			if x > max[i] {
				max[i] = x
			}
		}
	}
	q.setRanges(min, max)
	return q
}
