// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/authz"
)

// Regression suite for three defects in the JSON bulk-staging path, all
// reachable by an UNAUTHENTICATED client because ops.EncodeBulkStageArgs was
// evaluated as an ARGUMENT to a.call and therefore ran before a.authorize:
//
//  1. Unrecoverable OOM. The encoder sized its buffer
//     len(ids)*(8+dim*4) with dim taken from vecs[0] — two numbers the JSON body
//     chooses INDEPENDENTLY. One long vector plus many short points multiplies
//     out to a buffer no body could justify. A large enough request reaches
//     runtime.throw ("fatal error: out of memory"), which net/http's
//     per-connection recover cannot catch: one request kills the process.
//  2. Panic. A ragged body whose LATER vector is longer than vecs[0] wrote past
//     the buffer — index out of range, per request.
//  3. Silent corruption. A ragged body with a MIDDLE SHORT vector left the write
//     offset behind, shifting every following row: ids were read out of vector
//     bytes and points were stored under FABRICATED ids, with no error.
//
// Defect 3 is why the encoder — not the shard — has to reject raggedness.
// DecodeBulkStageArgs always materializes make([]float32, dim) per row, so a
// downstream dim check only ever sees uniform vectors: raggedness is destroyed
// by the encoder before any validation downstream can observe it.

// bulkJSON builds a /points/bulk body from explicit per-point vectors.
func bulkJSON(ids []uint64, vecs [][]float32) string {
	pts := make([]string, len(ids))
	for i := range ids {
		vb, _ := json.Marshal(vecs[i])
		pts[i] = fmt.Sprintf(`{"id":%d,"vector":%s}`, ids[i], vb)
	}
	return `{"points":[` + strings.Join(pts, ",") + `]}`
}

// TestBulkStageAnonymousAmplification is defect 1. An anonymous POST whose body
// is under a megabyte must be rejected with 401 having allocated a bounded
// amount — the encoder must not run before authorization, and must not size a
// buffer from two independently chosen numbers.
func TestBulkStageAnonymousAmplification(t *testing.T) {
	h, cleanup := newTestAPIOpts(t, Options{
		Authenticator: func(authz.AuthRequest) bool { return false },
	})
	defer cleanup()

	// One 20k-dim vector, then 20k points carrying no vector at all. The body is
	// ~890 KB; the old encoder multiplied out to 20000*(8+20000*4) ≈ 1.6 GB per
	// request, and the same shape scaled inside the 256 MiB body cap reached
	// hundreds of terabytes.
	const dim, extra = 20000, 20000
	var b strings.Builder
	b.WriteString(`{"points":[{"id":1,"vector":[`)
	for i := 0; i < dim; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteByte('1')
	}
	b.WriteString(`]}`)
	for i := 0; i < extra; i++ {
		fmt.Fprintf(&b, `,{"id":%d}`, i+2)
	}
	b.WriteString(`]}`)
	body := b.String()

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	r := httptest.NewRequest("POST", "/v1/collections/victim/points/bulk", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	runtime.ReadMemStats(&after)
	alloc := after.TotalAlloc - before.TotalAlloc

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d (%s), want 401", rec.Code, rec.Body.String())
	}
	// Generous next to the ~1.6 GB the defect churned, tight enough that any
	// reintroduced multiplication of dim by point count fails here.
	const budget = 64 << 20
	t.Logf("body=%dB status=%d allocated=%dB", len(body), rec.Code, alloc)
	if alloc > budget {
		t.Fatalf("anonymous %d-byte request allocated %d bytes (budget %d): "+
			"attacker-sized work is happening before authorization", len(body), alloc, budget)
	}
}

// TestBulkStageRaggedLongerIsRejected is defect 2: a later vector longer than
// the first used to write past the buffer and panic.
func TestBulkStageRaggedLongerIsRejected(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createCollP0(t, h, "ragged", 4)

	body := bulkJSON([]uint64{1, 2}, [][]float32{{1, 1, 1, 1}, {9, 9, 9, 9, 9, 9}})
	rec := do(t, h, "POST", "/v1/collections/ragged/points/bulk", body, nil) // panics on regression
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
	}
}

// TestBulkStageRaggedShortIsRejected is defect 3, the dangerous one: a MIDDLE
// short vector shifted every subsequent row, so ids were reconstructed out of
// vector bytes. The probe below produced ids [1 2 4674736414298996736] — a point
// stored under an id the client never sent, with no error returned.
func TestBulkStageRaggedShortIsRejected(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createCollP0(t, h, "shift", 4)

	body := bulkJSON([]uint64{1, 2, 3}, [][]float32{{1, 1, 1, 1}, {9, 9}, {7, 7, 7, 7}})
	rec := do(t, h, "POST", "/v1/collections/shift/points/bulk", body, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
	}

	// Nothing was staged: the build succeeds over an empty collection, and no
	// fabricated id is retrievable.
	if rec := do(t, h, "POST", "/v1/collections/shift/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build after a rejected ragged stage: %d %s", rec.Code, rec.Body.String())
	}
	for _, id := range []uint64{1, 2, 3, 4674736414298996736} {
		rec := do(t, h, "GET", fmt.Sprintf("/v1/collections/shift/points/%d", id), "", nil)
		if rec.Code == http.StatusOK {
			t.Fatalf("id %d is retrievable after a REJECTED ragged batch", id)
		}
	}
}

// TestBulkStageUniformWrongDimStillReachesTheStore checks the ragged rejection
// did not over-reach into a case that is NOT the encoder's business, and that
// the two independent checks compose the way they are supposed to.
//
// A UNIFORM batch whose dim is merely wrong for the collection is perfectly
// well-formed on the wire — one dim, every vector that length. The ENCODER must
// pass it through: whether it suits THIS collection needs the config, which the
// encoder does not have. The SHARD then rejects it at stage time, because that
// is where the config lives.
//
// The assertion is on WHICH layer refuses it, not merely that something did. An
// encoder that started rejecting uniform batches would still produce a 400 here
// and would look identical to a passing test, while having quietly taken over a
// decision it cannot make correctly.
func TestBulkStageUniformWrongDimStillReachesTheStore(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createCollP0(t, h, "wrongdim", 4)

	body := bulkJSON([]uint64{1, 2}, [][]float32{{1, 1, 1}, {2, 2, 2}}) // dim 3 vs 4
	rec := do(t, h, "POST", "/v1/collections/wrongdim/points/bulk", body, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("uniform wrong-dim gave %d (%s), want 400 from the shard", rec.Code, rec.Body.String())
	}
	// The shard names the collection's configured Dim; the encoder's ragged error
	// speaks only of the batch's own dim and cannot mention the collection.
	if !strings.Contains(rec.Body.String(), "collection Dim") {
		t.Fatalf("the rejection did not come from the shard — the encoder appears to "+
			"be deciding collection fit, which it cannot do: %s", rec.Body.String())
	}
	// Nothing was staged, so the build runs clean over an empty collection.
	if rec := do(t, h, "POST", "/v1/collections/wrongdim/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build after a rejected wrong-dim stage: %d %s", rec.Code, rec.Body.String())
	}
}

// TestBulkStageWellFormedStillWorks guards against over-rejection.
func TestBulkStageWellFormedStillWorks(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createCollP0(t, h, "ok", 4)

	body := bulkJSON([]uint64{1, 2, 3}, [][]float32{{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}})
	var res struct{ Staged int }
	rec := do(t, h, "POST", "/v1/collections/ok/points/bulk", body, &res)
	if rec.Code != http.StatusOK || res.Staged != 3 {
		t.Fatalf("well-formed stage: %d %s (staged=%d)", rec.Code, rec.Body.String(), res.Staged)
	}
	if rec := do(t, h, "POST", "/v1/collections/ok/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build: %d %s", rec.Code, rec.Body.String())
	}
	var hits struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/ok/points/search", `{"query":[2,2,2,2],"k":3}`, &hits)
	if len(hits.Results) != 3 || hits.Results[0].ID != 2 {
		t.Fatalf("search after bulk load returned %+v", hits.Results)
	}
}

func createCollP0(t *testing.T, h http.Handler, name string, dim int) {
	t.Helper()
	rec := do(t, h, "POST", "/v1/collections",
		fmt.Sprintf(`{"name":%q,"config":{"dim":%d,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, name, dim),
		nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create %s: %d %s", name, rec.Code, rec.Body.String())
	}
}
