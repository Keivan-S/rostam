// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"reflect"
	"testing"
)

// TestGetProjectedGatesCopies covers finding 022 at the index level: GetProjected
// must return exactly the projections requested — withVec gates the dense vector,
// withPayload gates meta + sparse — while version/ttl/ok always match Get. Run over
// every VectorIndex implementer (hnsw and ivf) through the shared build helper.
func TestGetProjectedGatesCopies(t *testing.T) {
	for name, idx := range getIntoIndexes(t) {
		t.Run(name, func(t *testing.T) {
			gp, okIface := idx.(projectedGetter)
			if !okIface {
				t.Fatalf("%s does not implement GetProjected", name)
			}
			// Reference full read (id 1 has payload + sparse).
			wVec, wMeta, wTTL, wSparse, wVer, wOK := idx.Get(1)
			if !wOK {
				t.Fatal("id 1 not found")
			}

			// both projections on → identical to Get.
			vec, meta, ttl, sparse, ver, okr := gp.GetProjected(1, true, true)
			if !okr || !reflect.DeepEqual(vec, wVec) || !reflect.DeepEqual(meta, wMeta) ||
				!reflect.DeepEqual(sparse, wSparse) || ttl != wTTL || ver != wVer {
				t.Errorf("both-on mismatch: vec=%v meta=%v sparse=%v ttl=%v ver=%d", vec, meta, sparse, ttl, ver)
			}

			// with_vector=false → no dense copy, payload retained.
			vec, meta, _, sparse, ver, okr = gp.GetProjected(1, false, true)
			if !okr || vec != nil || meta == nil || sparse == nil || ver != wVer {
				t.Errorf("vec-off: vec=%v meta=%v sparse=%v (want nil vec, kept payload)", vec, meta, sparse)
			}

			// with_payload=false → no meta/sparse clone, vector retained.
			vec, meta, _, sparse, ver, okr = gp.GetProjected(1, true, false)
			if !okr || vec == nil || meta != nil || sparse != nil || ver != wVer {
				t.Errorf("payload-off: vec=%v meta=%v sparse=%v (want kept vec, nil payload)", vec, meta, sparse)
			}

			// both off → only liveness + version + ttl, nothing copied.
			vec, meta, ttl, sparse, ver, okr = gp.GetProjected(1, false, false)
			if !okr || vec != nil || meta != nil || sparse != nil || ttl != wTTL || ver != wVer {
				t.Errorf("both-off: vec=%v meta=%v sparse=%v ttl=%v ver=%d", vec, meta, sparse, ttl, ver)
			}

			// miss → all nil, not found.
			if v, m, _, s, vr, o := gp.GetProjected(999, true, true); o || v != nil || m != nil || s != nil || vr != 0 {
				t.Errorf("miss: got ok=%v vec=%v meta=%v sparse=%v ver=%d", o, v, m, s, vr)
			}
		})
	}
}

// TestGetProjectedNoDenseAllocWhenVecOff confirms the batch fast path's core claim:
// a with_vector=false get on a bare point allocates NOTHING per call (the dense copy
// is skipped), whereas the full Get allocates the per-call []float32.
func TestGetProjectedNoDenseAllocWhenVecOff(t *testing.T) {
	for name, idx := range getIntoIndexes(t) {
		gp := idx.(projectedGetter)
		t.Run(name, func(t *testing.T) {
			// id 2 is a bare dense point (no payload/sparse/ttl), the zero-alloc target.
			projected := testing.AllocsPerRun(100, func() {
				_, _, _, _, _, _ = gp.GetProjected(2, false, false)
			})
			if projected != 0 {
				t.Errorf("GetProjected(with_vector=false) = %.1f allocs/op, want 0", projected)
			}
			get := testing.AllocsPerRun(100, func() {
				_, _, _, _, _, _ = idx.Get(2)
			})
			if get == 0 {
				t.Errorf("Get = %.1f allocs/op, want > 0 (the dense copy GetProjected elides)", get)
			}
		})
	}
}
