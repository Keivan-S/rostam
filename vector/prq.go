// SPDX-License-Identifier: Apache-2.0

package vector

// Product-Residual Quantization (PRQ): a stack of L product-quantizer layers in
// which each layer quantizes the RESIDUAL the previous layers could not represent.
// It extends the plain pq codec (pq.go) — every layer IS a plain m-subquantizer
// pq over the SAME dim — so a 1-layer PRQ is exactly a PQ.
//
// CODE LAYOUT: the per-vector code is L*m bytes, the L layers' m-byte sub-codes
// concatenated in layer order: code[layer*m : layer*m+m] is layer's PQ sub-code.
//
// OPQ PLACEMENT (applied ONCE, before layer 0): when OPQ is enabled the rotation R
// is learned once (it lives on layer 0's pq) and the input is rotated to Rx a
// SINGLE time at the front of encode/train/query. ALL residual layers operate in
// that rotated space (their codebooks are residual-rotated-space codebooks); only
// layers[0].rotation is non-nil, layers[1..].rotation stays nil so they never
// re-rotate. reconstruct concatenates+sums the per-layer reconstructions in
// rotated space and un-rotates by Rᵀ ONCE at the very end, landing back in the
// ORIGINAL space (mirrors pq.reconstruct's single Rᵀ).
//
// RESIDUAL CHAIN (train): layer 0 trains on the (rotated) sample; r0 = x_rot −
// reconstruct₀(code0) is the layer-0 residual in rotated space; layer 1 trains on
// {r0}; r1 = r0 − reconstruct₁(code1); … L layers. Each layer is a plain pq trained
// by trainPQ on its residual input WITHOUT its own rotation (rotation is layer 0's
// job alone) — so residual layers add finer detail in the already-rotated space.
//
// ADC (summed LUTs): the query is rotated ONCE, then one ADC LUT is built per layer
// over that layer's codebooks. Distance(code) = Σ_layer adc(qLUTs[layer], subcode).
// This is the ADDITIVE approximation — it IGNORES inter-layer cross-terms (the true
// distance to Σ_layer reconstrucion has cross-products between layers' centroids).
// This is intentional and sound here: HNSW always rescores the final candidate set
// on EXACT float32 (hnsw.rescore), so the approximate Distance only needs to ORDER
// graph traversal; the exact rescore fixes the final ranking. The summed-LUT form
// avoids the hard additive-quantizer cross-term bookkeeping entirely.
type prq struct {
	layers []*pq // L PQ layers; layers[0] owns the OPQ rotation R (others nil)
	l      int   // number of layers (== len(layers))
	m      int   // sub-quantizers per layer
	dim    int   // full vector dimension
	metric Metric
}

// defaultPRQLayers is the layer count used when Config.PRQLayers is 0.
const defaultPRQLayers = 2

// trainPRQ trains an L-layer m-subquantizer PRQ on vecs (the build sample) for
// dim/metric. seed/workers/opq/opqIters mirror trainPQ. OPQ (when on) is applied
// ONCE: layer 0 is trained via trainPQ with opq=true so it learns R and trains its
// codebooks on Rx; the residual chain is computed in that ROTATED space (so every
// residual layer's codebooks are rotated-space codebooks) and the residual layers
// are trained WITHOUT their own rotation. opq=false ⇒ every layer is a plain PQ in
// the original space (rotation nil throughout). Requires dim%m == 0 and m > 0 and
// l >= 1; returns the same errors as trainPQ for a bad m/dim.
func trainPRQ(vecs [][]float32, m, dim, l int, seed int64, metric Metric, workers int, opq bool, opqIters int) (*prq, error) {
	if l < 1 {
		l = defaultPRQLayers
	}
	p := &prq{l: l, m: m, dim: dim, metric: metric, layers: make([]*pq, 0, l)}

	// Layer 0: a full PQ (with OPQ when requested). trainPQ rotates vecs to Rx
	// internally and stores R on the returned codec; the residual chain below
	// works in that rotated space. A train error (bad m/dim) propagates verbatim.
	// PRQ trains isotropic (eta=1): anisotropic PQ is a Task-1 HNSW-PQ/IVF-PQ knob;
	// keeping PRQ on the existing L2 trainer preserves its byte-identical codebooks.
	layer0, err := trainPQ(vecs, m, dim, seed, metric, workers, opq, opqIters, 1, 8)
	if err != nil {
		return nil, err
	}
	p.layers = append(p.layers, layer0)

	// Residual chain. Work in ROTATED space when OPQ is on: rotate every sample
	// vector once (matching layer 0's R), then iteratively subtract each layer's
	// rotated-space reconstruction and train the next layer on the residual.
	// rotation == nil ⇒ rotated space IS the original space (no-op rotate).
	rot := layer0.rotation
	residual := make([][]float32, len(vecs))
	for i, v := range vecs {
		x := v
		if rot != nil {
			x = rotate(rot, v)
		}
		// r = x_rot − reconstruct_rotated(layer0.code(x))
		residual[i] = subVec(x, layer0.reconstructRotated(layer0.encode(v)))
	}

	for layer := 1; layer < l; layer++ {
		// Each residual layer is a PLAIN pq (no rotation of its own — OPQ is layer
		// 0's job). Train on the current residual set in rotated space.
		lp, lerr := trainPQ(residual, m, dim, seed+int64(layer)*0x9E3779B1, metric, workers, false, 0, 1, 8)
		if lerr != nil {
			return nil, lerr
		}
		p.layers = append(p.layers, lp)
		if layer == l-1 {
			break // no further residual needed after the last layer
		}
		// Subtract this layer's reconstruction to form the next residual (rotated
		// space; lp has no rotation so reconstruct == reconstructRotated).
		for i := range residual {
			residual[i] = subVec(residual[i], lp.reconstructRotated(lp.encode(residual[i])))
		}
	}
	return p, nil
}

// subVec returns a − b component-wise (fresh slice; len == len(a)). Used by the
// residual chain. len(a) == len(b) by construction (both dim-length).
func subVec(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

// CodeLen returns the encoded length in bytes: L*m (one byte per subspace per
// layer). The arena sizes the per-slot codes side-array to this.
func (p *prq) CodeLen() int { return p.l * p.m }

// rotation returns layer 0's OPQ rotation R (dim×dim flat, row-major), or nil when
// OPQ is off. Only layer 0 carries R (it is applied once). Used by the snapshot
// writers + the quantizer adapter.
func (p *prq) rotation() []float32 {
	if len(p.layers) == 0 {
		return nil
	}
	return p.layers[0].rotation
}

// encodeInto writes the L*m-byte PRQ code for vec into dst (len(dst) == CodeLen()).
// It rotates vec ONCE to Rx (layer 0's R) up front, then greedily encodes by layer
// 0, subtracts that layer's reconstruction, encodes the residual by layer 1, … —
// each layer's m sub-code bytes written into its slice of dst. The per-layer
// encode/reconstruct run in ROTATED space so dst[layer*m:] is layer's rotated-space
// sub-code (decoded the same way by reconstruct). Allocation: one rotated copy +
// one running residual buffer per call; the per-insert path is not on the hottest
// loop (build re-encodes in parallel, search rotates the query once).
func (p *prq) encodeInto(dst []byte, vec []float32) {
	rot := p.rotation()
	// x is the working vector in ROTATED space (== original when rot is nil).
	x := vec
	if rot != nil {
		x = rotate(rot, vec)
	}
	// running residual starts at x; each layer subtracts its rotated-space recon.
	residual := append([]float32(nil), x...)
	for layer := 0; layer < p.l; layer++ {
		lp := p.layers[layer]
		sub := dst[layer*p.m : layer*p.m+p.m]
		// Encode the residual by this layer WITHOUT re-rotating: residual already
		// lives in rotated space, and only layer 0 has a rotation. To skip the
		// per-layer rotation we encode against a rotation-free view of the codec.
		encodeRotatedInto(lp, sub, residual)
		if layer == p.l-1 {
			break
		}
		recon := lp.reconstructRotated(sub)
		for i := range residual {
			residual[i] -= recon[i]
		}
	}
}

// encodeRotatedInto encodes sub-vector residual (already in the codec's rotated
// space) into dst[:m] by nearest sub-centroid, WITHOUT applying lp.rotation. The
// PRQ residual chain is entirely in rotated space and only layer 0 carries R, so
// the per-layer encode must NOT re-rotate. This mirrors pq.encodeInto's inner loop
// minus the rotation prologue (which pq.encodeInto runs unconditionally when
// rotation != nil). Used only by prq.encodeInto.
func encodeRotatedInto(lp *pq, dst []byte, residual []float32) {
	dist := pickDistDim(L2, lp.dsub)
	for s := 0; s < lp.m; s++ {
		sub := residual[s*lp.dsub : s*lp.dsub+lp.dsub]
		cb := lp.codebooks[s]
		best := 0
		bestD := dist(sub, cb[0])
		for c := 1; c < len(cb); c++ {
			if d := dist(sub, cb[c]); d < bestD {
				bestD = d
				best = c
			}
		}
		dst[s] = byte(best) //nolint:gosec // best ∈ [0,255]: len(cb) <= 256
	}
}

// Encode writes the L*m-byte PRQ code for vec into a fresh slice and returns it.
func (p *prq) Encode(vec []float32) []byte {
	code := make([]byte, p.CodeLen())
	p.encodeInto(code, vec)
	return code
}

// reconstruct returns the APPROXIMATE float32 vector for a PRQ code: it sums the L
// layers' rotated-space sub-centroid reconstructions, then un-rotates by Rᵀ ONCE so
// the result lands back in the ORIGINAL space (matching pq.reconstruct's single
// Rᵀ). For a 1-layer PRQ with OPQ this is byte-equal to pq.reconstruct. The
// returned slice is freshly allocated.
func (p *prq) reconstruct(code []byte) []float32 {
	out := make([]float32, p.dim)
	for layer := 0; layer < p.l; layer++ {
		sub := code[layer*p.m : layer*p.m+p.m]
		recon := p.layers[layer].reconstructRotated(sub)
		for i := range out {
			out[i] += recon[i]
		}
	}
	// Un-rotate ONCE: the summed reconstruction is in rotated space; Rᵀ returns it
	// to the original space. nil rotation ⇒ no-op (already original space).
	if rot := p.rotation(); rot != nil {
		out = rotateT(rot, out)
	}
	return out
}

// prqLUTs is the per-search query handle PrepareQuery produces: one ADC LUT per
// layer (each [m*256]float32, built over that layer's codebooks). The query is
// rotated ONCE up front so every layer's LUT is computed against rotated-space
// codebooks (the rotation is an isometry, so the per-layer adc still matches that
// layer's metric contribution). Distance sums the per-layer adc lookups.
type prqLUTs struct {
	luts [][]float32 // [L][m*256]
}

// PrepareQuery builds the summed-LUT query handle: rotate q ONCE (layer 0's R),
// then one ADC LUT per layer over that layer's codebooks. The per-layer LUT is the
// SAME table pq.queryLUT builds (the metric contribution of the rotated query's
// sub-vectors against each layer's centroids), EXCEPT the rotation is applied here
// once rather than per layer, so each layer's LUT is built rotation-free.
func (p *prq) PrepareQuery(q []float32) *prqLUTs {
	rot := p.rotation()
	rq := q
	if rot != nil {
		rq = rotate(rot, q)
	}
	luts := make([][]float32, p.l)
	for layer := 0; layer < p.l; layer++ {
		luts[layer] = queryLUTRotated(p.layers[layer])(rq)
	}
	return &prqLUTs{luts: luts}
}

// queryLUTRotated returns a closure that fills lp's ADC LUT for an ALREADY-ROTATED
// query (no per-layer rotation). It mirrors pq.queryLUTInto's inner loop minus the
// rotation prologue. Returned as a closure so PrepareQuery can apply it per layer.
func queryLUTRotated(lp *pq) func(rq []float32) []float32 {
	return func(rq []float32) []float32 {
		lut := make([]float32, lp.lutLen())
		negDot := lp.metric != L2
		for s := 0; s < lp.m; s++ {
			sub := rq[s*lp.dsub : s*lp.dsub+lp.dsub]
			cb := lp.codebooks[s]
			base := s * pqCodebookSize
			for c := 0; c < len(cb); c++ {
				if negDot {
					lut[base+c] = -dotProduct(sub, cb[c])
				} else {
					lut[base+c] = l2Squared(sub, cb[c])
				}
			}
		}
		return lut
	}
}

// Distance is the summed-LUT ADC: Σ_layer adcRaw(qLUTs[layer], subcode_layer),
// plus the metric offset ONCE (1 for Cosine so the result is 1 − Σdot; 0 else).
// The offset is added a single time (not per layer) so the value tracks the metric
// distance; the per-layer adcRaw sums the raw LUT lookups without the offset.
// Smaller = nearer. INTER-LAYER CROSS-TERMS ARE IGNORED (the additive
// approximation) — sound because HNSW exact-rescores the shortlist (see the type
// doc). Lower = nearer, matching distFunc semantics.
func (p *prq) Distance(qc *prqLUTs, code []byte) float32 {
	sum := pqMetricOffset(p.metric)
	for layer := 0; layer < p.l; layer++ {
		sum += adcRaw(qc.luts[layer], code[layer*p.m:layer*p.m+p.m], p.m)
	}
	return sum
}

// adcRaw sums the m byte-indexed LUT lookups for one layer's sub-code WITHOUT the
// metric offset (the caller adds it once across layers). Mirrors pq.adc's inner
// bounds-check-eliminated loop. code is the layer's m-byte slice; m == the layer's
// subspace count; lut is the layer's [m*256]float32 ADC table.
func adcRaw(lut []float32, code []byte, m int) float32 {
	var sum float32
	code = code[:m]
	lut = lut[: m*pqCodebookSize : m*pqCodebookSize]
	for s, c := range code {
		sum += lut[s*pqCodebookSize+int(c)]
	}
	return sum
}

// CodeDistance is the symmetric approximate distance between two PRQ codes: it
// reconstructs code a into an approximate query (its summed rotated-space
// reconstruction, then Rᵀ to original space), builds the per-layer LUTs from it,
// and ADC-scores code b. This reuses the asymmetric summed-LUT path so the ordering
// matches Distance (the build path and search rank candidates the same way). It is
// the costlier-but-consistent of the two options (reconstruct-a + ADC-b vs.
// reconstruct-both + distFunc); consistency with Distance is what keeps the
// quantized build graph aligned with the search traversal, so it is preferred.
func (p *prq) CodeDistance(a, b []byte) float32 {
	return p.Distance(p.PrepareQuery(p.reconstruct(a)), b)
}
