// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"fmt"
	"net/http"
	"strings"
	"testing"
)

// createDocsCAS creates a 2-dim L2 collection for the CAS HTTP tests.
func createDocsCAS(t *testing.T, h http.Handler) {
	t.Helper()
	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":2,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPGetCarriesVersionAndCASConflict exercises the full HTTP→engine wire:
// a Get response carries the per-point version, a matching expected_version
// applies, and a mismatch is 409 Conflict.
func TestHTTPGetCarriesVersionAndCASConflict(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	// Insert id 1 (no CAS).
	if rec := do(t, h, "POST", "/v1/collections/docs/points", `{"id":1,"vector":[1,0]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("insert = %d (%s)", rec.Code, rec.Body)
	}
	// Get returns version 1.
	var gres struct {
		Found   bool   `json:"found"`
		Version uint64 `json:"version"`
	}
	if rec := do(t, h, "GET", "/v1/collections/docs/points/1", "", &gres); rec.Code != http.StatusOK {
		t.Fatalf("get = %d (%s)", rec.Code, rec.Body)
	}
	if !gres.Found || gres.Version != 1 {
		t.Fatalf("get found=%v version=%d, want true,1", gres.Found, gres.Version)
	}

	// CAS set_payload with matching expected_version → applied (200).
	rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload?expected_version=1", `{"k":{"kind":"string","str":"a"}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("matching CAS payload = %d (%s)", rec.Code, rec.Body)
	}
	// CAS set_payload with WRONG expected_version → 409 Conflict.
	rec = do(t, h, "POST", "/v1/collections/docs/points/1/payload?expected_version=999", `{"k":{"kind":"string","str":"b"}}`, nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("mismatched CAS payload = %d (%s), want 409", rec.Code, rec.Body)
	}
	// CAS delete with WRONG expected_version → 409; point survives.
	rec = do(t, h, "DELETE", "/v1/collections/docs/points/1?expected_version=999", "", nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("mismatched CAS delete = %d (%s), want 409", rec.Code, rec.Body)
	}
	if rec := do(t, h, "GET", "/v1/collections/docs/points/1", "", nil); rec.Code != http.StatusOK {
		t.Fatalf("point removed despite CAS conflict (get = %d)", rec.Code)
	}
}

// TestHTTPCASUpsertConflict covers the insert/upsert CAS body field: an
// expect-absent CAS (expected_version 0) on a LIVE id is 409.
func TestHTTPCASUpsertConflict(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	if rec := do(t, h, "POST", "/v1/collections/docs/points", `{"id":1,"vector":[1,0]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("insert = %d (%s)", rec.Code, rec.Body)
	}
	// expect-absent CAS on a live id → 409.
	rec := do(t, h, "POST", "/v1/collections/docs/points", `{"id":1,"vector":[2,0],"expected_version":0}`, nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("expect-absent CAS on live id = %d (%s), want 409", rec.Code, rec.Body)
	}
	// Matching expected_version upsert → 200.
	rec = do(t, h, "POST", "/v1/collections/docs/points", `{"id":1,"vector":[3,0],"upsert":true,"expected_version":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("matching CAS upsert = %d (%s), want 200", rec.Code, rec.Body)
	}
}

// TestHTTPNoCASUnchanged proves a request WITHOUT any expected_version behaves
// exactly as before: insert + delete succeed (no CAS enforcement).
func TestHTTPNoCASUnchanged(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	if rec := do(t, h, "POST", "/v1/collections/docs/points", `{"id":1,"vector":[1,0]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("insert = %d (%s)", rec.Code, rec.Body)
	}
	var dres struct {
		Deleted bool `json:"deleted"`
	}
	if rec := do(t, h, "DELETE", "/v1/collections/docs/points/1", "", &dres); rec.Code != http.StatusOK || !dres.Deleted {
		t.Fatalf("no-CAS delete = %d deleted=%v (%s)", rec.Code, dres.Deleted, rec.Body)
	}
}

// batchResults is the CAS-mode /points/batch response (per-id version + status).
type batchResults struct {
	Results []struct {
		ID      uint64 `json:"id"`
		Version uint64 `json:"version"`
		Status  string `json:"status"`
	} `json:"results"`
}

// TestHTTPBatchNoCASUnchanged proves a batch body WITHOUT any expected_version on
// any point behaves EXACTLY as before: the response is the legacy {"count":N}
// shape (no "results" array), HTTP 200.
func TestHTTPBatchNoCASUnchanged(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	body := `{"upsert":true,"points":[{"id":1,"vector":[1,0]},{"id":2,"vector":[2,0]}]}`
	var bres struct {
		Count   int   `json:"count"`
		Results []any `json:"results"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/points/batch", body, &bres)
	if rec.Code != http.StatusOK || bres.Count != 2 {
		t.Fatalf("no-CAS batch = %d count=%d (%s)", rec.Code, bres.Count, rec.Body)
	}
	if bres.Results != nil {
		t.Fatalf("no-CAS batch leaked a results array: %s", rec.Body)
	}
	// The legacy body is the bare {"count":N} object — no "results" key at all.
	if strings.Contains(rec.Body.String(), "results") {
		t.Fatalf("no-CAS batch body not byte-identical (has results key): %s", rec.Body)
	}
}

// TestHTTPBatchMixedCAS covers a batch with MIXED per-id expected_version: a
// matching precondition applies (and returns the bumped version), a mismatch is
// reported as a per-id conflict (no mutation — verified via a follow-up get),
// and a point without expected_version applies unconditionally. Overall 200.
func TestHTTPBatchMixedCAS(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	// Seed ids 1 and 2 at version 1 (no CAS).
	for _, id := range []int{1, 2} {
		body := fmt.Sprintf(`{"id":%d,"vector":[%d,0],"upsert":true}`, id, id)
		if rec := do(t, h, "POST", "/v1/collections/docs/points", body, nil); rec.Code != http.StatusOK {
			t.Fatalf("seed %d = %d (%s)", id, rec.Code, rec.Body)
		}
	}

	// Batch (upsert path = delete-then-insert, so an applied point's post-write
	// version is 1):
	//  id 1: expected_version 1 (matches)   → ok, post-write v1
	//  id 2: expected_version 999 (mismatch) → conflict, unchanged
	//  id 3: no expected_version             → unconditional upsert, v1
	body := `{"upsert":true,"points":[` +
		`{"id":1,"vector":[9,0],"expected_version":1},` +
		`{"id":2,"vector":[9,0],"expected_version":999},` +
		`{"id":3,"vector":[3,0]}` +
		`]}`
	var br batchResults
	rec := do(t, h, "POST", "/v1/collections/docs/points/batch", body, &br)
	if rec.Code != http.StatusOK {
		t.Fatalf("mixed CAS batch = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(br.Results) != 3 {
		t.Fatalf("results = %d, want 3 (%s)", len(br.Results), rec.Body)
	}
	want := map[uint64]struct {
		status  string
		version uint64
	}{
		1: {"ok", 1},
		2: {"conflict", 0},
		3: {"ok", 1},
	}
	for _, res := range br.Results {
		exp, ok := want[res.ID]
		if !ok {
			t.Fatalf("unexpected id %d in results", res.ID)
		}
		if res.Status != exp.status || res.Version != exp.version {
			t.Fatalf("id %d = {status:%q version:%d}, want {status:%q version:%d}",
				res.ID, res.Status, res.Version, exp.status, exp.version)
		}
	}

	// The conflicting point (id 2) must be UNCHANGED: still version 1 with its
	// original vector (a conflict mutated nothing).
	var g2 struct {
		Found   bool      `json:"found"`
		Vector  []float32 `json:"vector"`
		Version uint64    `json:"version"`
	}
	if rec := do(t, h, "GET", "/v1/collections/docs/points/2", "", &g2); rec.Code != http.StatusOK {
		t.Fatalf("get id 2 = %d (%s)", rec.Code, rec.Body)
	}
	if !g2.Found || g2.Version != 1 {
		t.Fatalf("conflicting id 2 mutated: found=%v version=%d, want true,1", g2.Found, g2.Version)
	}
	if len(g2.Vector) != 2 || g2.Vector[0] != 2 {
		t.Fatalf("conflicting id 2 vector mutated: %v, want [2 0]", g2.Vector)
	}
}

// TestHTTPBatchCASInsertIfAbsent covers expect-absent CAS (expected_version 0)
// in a batch: it creates a missing id at v1 and conflicts on a live id.
func TestHTTPBatchCASInsertIfAbsent(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	// Seed id 1 live (so expect-absent on it must conflict).
	if rec := do(t, h, "POST", "/v1/collections/docs/points", `{"id":1,"vector":[1,0]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("seed = %d (%s)", rec.Code, rec.Body)
	}

	// id 1 expect-absent (live → conflict); id 2 expect-absent (missing → create v1).
	body := `{"points":[` +
		`{"id":1,"vector":[5,0],"expected_version":0},` +
		`{"id":2,"vector":[5,0],"expected_version":0}` +
		`]}`
	var br batchResults
	rec := do(t, h, "POST", "/v1/collections/docs/points/batch", body, &br)
	if rec.Code != http.StatusOK || len(br.Results) != 2 {
		t.Fatalf("expect-absent batch = %d results=%d (%s)", rec.Code, len(br.Results), rec.Body)
	}
	got := map[uint64]string{}
	ver := map[uint64]uint64{}
	for _, res := range br.Results {
		got[res.ID] = res.Status
		ver[res.ID] = res.Version
	}
	if got[1] != "conflict" {
		t.Fatalf("id 1 expect-absent on live = %q, want conflict (%s)", got[1], rec.Body)
	}
	if got[2] != "ok" || ver[2] != 1 {
		t.Fatalf("id 2 expect-absent create = {status:%q version:%d}, want {ok,1} (%s)", got[2], ver[2], rec.Body)
	}
}

// TestHTTPBatchCASRealErrorFailsLoud proves a TRUE error (a bad/unknown
// collection) still fails the WHOLE batch loud even on the CAS path — it is NOT
// silently downgraded to a per-id conflict.
func TestHTTPBatchCASRealErrorFailsLoud(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createDocsCAS(t, h)

	// Target a collection that does not exist; the CAS path is engaged via
	// expected_version. The unknown collection is a real error → non-200, with the
	// legacy {"error","committed"} shape (NOT a per-id conflict).
	body := `{"points":[{"id":1,"vector":[1,0],"expected_version":1}]}`
	var er struct {
		Error     string `json:"error"`
		Committed int    `json:"committed"`
		Results   []any  `json:"results"`
	}
	rec := do(t, h, "POST", "/v1/collections/nope/points/batch", body, &er)
	if rec.Code == http.StatusOK {
		t.Fatalf("bad-collection batch = 200, want a loud failure (%s)", rec.Body)
	}
	if er.Error == "" {
		t.Fatalf("bad-collection batch did not report an error (%s)", rec.Body)
	}
	if er.Results != nil {
		t.Fatalf("real error leaked into a per-id results array (swallowed as conflict): %s", rec.Body)
	}
}
