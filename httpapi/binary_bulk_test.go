// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// binBody builds an "RVB1" binary bulk body. payloads may be nil (flag clear) or
// one JSON blob per point ("" = no payload for that point).
func binBody(flags uint32, ids []uint64, vecs [][]float32, payloads []string) []byte {
	dim := 0
	if len(vecs) > 0 {
		dim = len(vecs[0])
	}
	var b bytes.Buffer
	b.WriteString(binaryBulkMagic)
	_ = binary.Write(&b, binary.BigEndian, flags)
	_ = binary.Write(&b, binary.BigEndian, uint32(len(ids)))
	_ = binary.Write(&b, binary.BigEndian, uint32(dim))
	for i, id := range ids {
		_ = binary.Write(&b, binary.BigEndian, id)
		for _, f := range vecs[i] {
			_ = binary.Write(&b, binary.BigEndian, math.Float32bits(f))
		}
	}
	for _, p := range payloads {
		_ = binary.Write(&b, binary.BigEndian, uint32(len(p)))
		b.WriteString(p)
	}
	return b.Bytes()
}

// doBin issues a POST with the binary content type and returns the recorder.
func doBin(t *testing.T, h http.Handler, path string, body []byte, out any) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest("POST", path, bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/octet-stream")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if out != nil && rec.Body.Len() > 0 {
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			t.Fatalf("POST %s: decode response %q: %v", path, rec.Body.String(), err)
		}
	}
	return rec
}

// testPoints builds n deterministic dim-d points with ids 1..n.
func testPoints(n, dim int) ([]uint64, [][]float32) {
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(i) + float32(d)/8
		}
		vecs[i] = v
	}
	return ids, vecs
}

func createColl(t *testing.T, h http.Handler, name string, dim int) {
	t.Helper()
	rec := do(t, h, "POST", "/v1/collections",
		fmt.Sprintf(`{"name":%q,"config":{"dim":%d,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, name, dim),
		nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create %s: %d %s", name, rec.Code, rec.Body.String())
	}
}

// TestBinaryBulkMatchesJSONBulk is the anti-divergence test: staging the SAME
// points over the binary framing and over the JSON body must produce the same
// index — same staged count, same build, same search results in the same order.
func TestBinaryBulkMatchesJSONBulk(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	const n, dim = 60, 16
	ids, vecs := testPoints(n, dim)
	createColl(t, h, "jsoncoll", dim)
	createColl(t, h, "bincoll", dim)

	// JSON staging (unchanged path).
	pts := make([]string, n)
	for i := range ids {
		vb, _ := json.Marshal(vecs[i])
		pts[i] = fmt.Sprintf(`{"id":%d,"vector":%s}`, ids[i], vb)
	}
	var jres struct{ Staged int }
	rec := do(t, h, "POST", "/v1/collections/jsoncoll/points/bulk", `{"points":[`+strings.Join(pts, ",")+`]}`, &jres)
	if rec.Code != http.StatusOK {
		t.Fatalf("json stage: %d %s", rec.Code, rec.Body.String())
	}

	// Binary staging.
	var bres struct{ Staged int }
	rec = doBin(t, h, "/v1/collections/bincoll/points/bulk", binBody(0, ids, vecs, nil), &bres)
	if rec.Code != http.StatusOK {
		t.Fatalf("binary stage: %d %s", rec.Code, rec.Body.String())
	}
	if jres.Staged != n || bres.Staged != n {
		t.Fatalf("staged counts differ: json=%d binary=%d want %d", jres.Staged, bres.Staged, n)
	}

	for _, c := range []string{"jsoncoll", "bincoll"} {
		if rec := do(t, h, "POST", "/v1/collections/"+c+"/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
			t.Fatalf("build %s: %d %s", c, rec.Code, rec.Body.String())
		}
	}

	qb, _ := json.Marshal(vecs[7])
	query := fmt.Sprintf(`{"query":%s,"k":10}`, qb)
	var jhits, bhits struct {
		Results []struct {
			ID       uint64  `json:"id"`
			Distance float64 `json:"distance"`
		} `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/jsoncoll/points/search", query, &jhits)
	do(t, h, "POST", "/v1/collections/bincoll/points/search", query, &bhits)
	if len(jhits.Results) == 0 || len(jhits.Results) != len(bhits.Results) {
		t.Fatalf("result counts differ: json=%d binary=%d", len(jhits.Results), len(bhits.Results))
	}
	for i := range jhits.Results {
		if jhits.Results[i].ID != bhits.Results[i].ID || jhits.Results[i].Distance != bhits.Results[i].Distance {
			t.Fatalf("hit %d differs: json=%+v binary=%+v", i, jhits.Results[i], bhits.Results[i])
		}
	}
}

// TestBinaryBatchWithPayloads covers the filter-case path: the binary body
// carries a JSON payload blob per point on /points/batch, and a filtered search
// then matches on it.
func TestBinaryBatchWithPayloads(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	const n, dim = 40, 8
	ids, vecs := testPoints(n, dim)
	createColl(t, h, "pay", dim)

	payloads := make([]string, n)
	for i, id := range ids {
		payloads[i] = fmt.Sprintf(`{"id":{"kind":"int","int":%d}}`, id)
	}
	var res struct{ Count int }
	rec := doBin(t, h, "/v1/collections/pay/points/batch",
		binBody(binBulkFlagPayloads|binBulkFlagUpsert, ids, vecs, payloads), &res)
	if rec.Code != http.StatusOK {
		t.Fatalf("binary batch: %d %s", rec.Code, rec.Body.String())
	}
	if res.Count != n {
		t.Fatalf("count = %d, want %d", res.Count, n)
	}

	qb, _ := json.Marshal(vecs[0])
	var hits struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}
	body := fmt.Sprintf(`{"query":%s,"k":50,"filter":{"op":"gte","field":"id","value":{"kind":"int","int":30}}}`, qb)
	do(t, h, "POST", "/v1/collections/pay/points/search", body, &hits)
	if len(hits.Results) == 0 {
		t.Fatal("filtered search returned nothing; payloads did not land")
	}
	for _, hit := range hits.Results {
		if hit.ID < 30 {
			t.Fatalf("filtered search returned id %d < 30", hit.ID)
		}
	}
}

// TestBinaryBatchNoPayloadsMatchesJSON checks the no-payload binary batch body
// decodes into the same request the JSON batch body does.
func TestBinaryBatchNoPayloadsMatchesJSON(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	const n, dim = 20, 8
	ids, vecs := testPoints(n, dim)
	createColl(t, h, "bb", dim)

	var res struct{ Count int }
	rec := doBin(t, h, "/v1/collections/bb/points/batch", binBody(binBulkFlagUpsert, ids, vecs, nil), &res)
	if rec.Code != http.StatusOK || res.Count != n {
		t.Fatalf("binary batch: %d %s (count=%d)", rec.Code, rec.Body.String(), res.Count)
	}
	var got struct {
		Vector []float32 `json:"vector"`
	}
	if rec := do(t, h, "GET", "/v1/collections/bb/points/5", "", &got); rec.Code != http.StatusOK {
		t.Fatalf("get: %d %s", rec.Code, rec.Body.String())
	}
	want := vecs[4] // id 5 is the 5th point
	if len(got.Vector) != len(want) {
		t.Fatalf("vector len = %d, want %d", len(got.Vector), len(want))
	}
	for i := range want {
		if got.Vector[i] != want[i] {
			t.Fatalf("vector[%d] = %v, want %v", i, got.Vector[i], want[i])
		}
	}
}

// TestBinaryBulkContentTypeNegotiation checks that only application/octet-stream
// selects the binary framing, and that a JSON body on the same route is
// unaffected — including a JSON content type carrying a charset parameter.
func TestBinaryBulkContentTypeNegotiation(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "neg", 4)

	// JSON body, explicit JSON content type: unchanged path.
	r := httptest.NewRequest("POST", "/v1/collections/neg/points/bulk", strings.NewReader(`{"points":[{"id":1,"vector":[1,2,3,4]}]}`))
	r.Header.Set("Content-Type", "application/json; charset=utf-8")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusOK {
		t.Fatalf("json+charset: %d %s", rec.Code, rec.Body.String())
	}

	// JSON body with NO content type at all: still the JSON path.
	if rec := do(t, h, "POST", "/v1/collections/neg/points/bulk", `{"points":[{"id":2,"vector":[1,2,3,4]}]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("json+no content type: %d %s", rec.Code, rec.Body.String())
	}

	// Binary content type WITH a parameter: still selects binary.
	ids, vecs := testPoints(1, 4)
	rb := httptest.NewRequest("POST", "/v1/collections/neg/points/bulk", bytes.NewReader(binBody(0, ids, vecs, nil)))
	rb.Header.Set("Content-Type", "application/octet-stream; v=1")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, rb)
	if rec.Code != http.StatusOK {
		t.Fatalf("binary+param: %d %s", rec.Code, rec.Body.String())
	}

	// A binary body sent WITHOUT the binary content type must not be mistaken for
	// JSON-shaped input: it is rejected as invalid JSON, never silently accepted.
	rj := httptest.NewRequest("POST", "/v1/collections/neg/points/bulk", bytes.NewReader(binBody(0, ids, vecs, nil)))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, rj)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("binary body as JSON: %d %s, want 400", rec.Code, rec.Body.String())
	}
}

// TestBinaryBulkHostileBodies covers every way a crafted body can lie about its
// own size, plus the two flag combinations the staging route refuses.
func TestBinaryBulkHostileBodies(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "hostile", 4)

	ids, vecs := testPoints(2, 4)
	good := binBody(0, ids, vecs, nil)

	// hdr builds a header with arbitrary flags/count/dim and the given trailer.
	hdr := func(flags, count, dim uint32, trailer []byte) []byte {
		var b bytes.Buffer
		b.WriteString(binaryBulkMagic)
		_ = binary.Write(&b, binary.BigEndian, flags)
		_ = binary.Write(&b, binary.BigEndian, count)
		_ = binary.Write(&b, binary.BigEndian, dim)
		b.Write(trailer)
		return b.Bytes()
	}

	cases := []struct {
		name string
		path string
		body []byte
		want int
	}{
		{"empty body", "/v1/collections/hostile/points/bulk", nil, http.StatusBadRequest},
		{"header truncated", "/v1/collections/hostile/points/bulk", good[:9], http.StatusBadRequest},
		{"bad magic", "/v1/collections/hostile/points/bulk", append([]byte("XXXX"), good[4:]...), http.StatusBadRequest},
		{"unknown flag", "/v1/collections/hostile/points/bulk", hdr(1<<7, 0, 0, nil), http.StatusBadRequest},
		{"dim zero with points", "/v1/collections/hostile/points/bulk", hdr(0, 3, 0, nil), http.StatusBadRequest},
		// count near 2^32 with a tiny body: the product count*(8+dim*4) would wrap
		// if computed naively, so this MUST be rejected before any reservation.
		{"count overflow", "/v1/collections/hostile/points/bulk", hdr(0, 0xFFFFFFFF, 4, nil), http.StatusRequestEntityTooLarge},
		// A dim large enough to blow the byte cap is refused by the cap, in this
		// layer. A dim that is merely WRONG for the collection is not knowable here
		// (that would cost a routed config lookup per request) and is refused by the
		// shard instead — see TestBinaryBulkDimMustMatchCollection.
		{"dim overflow", "/v1/collections/hostile/points/bulk", hdr(0, 1, 0xFFFFFFFF, nil), http.StatusRequestEntityTooLarge},
		{"count exceeds body", "/v1/collections/hostile/points/bulk", hdr(0, 1000, 4, []byte{1, 2, 3}), http.StatusBadRequest},
		{"rows truncated", "/v1/collections/hostile/points/bulk", good[:len(good)-4], http.StatusBadRequest},
		// Extra bytes after the declared points mean the sender framed the body
		// differently than we read it; ingesting the prefix would stage a corpus
		// nobody asked for.
		{"trailing bytes", "/v1/collections/hostile/points/bulk", append(append([]byte{}, good...), 9, 9, 9), http.StatusBadRequest},
		{"trailing bytes on batch", "/v1/collections/hostile/points/batch", append(append([]byte{}, good...), 9), http.StatusBadRequest},
		{"upsert flag on staging route", "/v1/collections/hostile/points/bulk",
			binBody(binBulkFlagUpsert, ids, vecs, nil), http.StatusBadRequest},
		// STAGING route, payload section. The staging route used to refuse the
		// payload flag outright (the op had nowhere to put a payload); it now streams
		// the section through to vector_bulk_stage_payload, so every malformed shape
		// the batch route rejects has to be rejected here too — the transport's
		// bounds discipline is not something the new op inherits for free.
		{"staging payload section missing", "/v1/collections/hostile/points/bulk",
			binBody(binBulkFlagPayloads, ids, vecs, nil), http.StatusBadRequest},
		{"staging payload truncated", "/v1/collections/hostile/points/bulk",
			append(binBody(binBulkFlagPayloads, ids, vecs, nil), 0, 0, 0, 8, 'x'), http.StatusBadRequest},
		{"staging payload not JSON", "/v1/collections/hostile/points/bulk",
			binBody(binBulkFlagPayloads, ids, vecs, []string{"not json", "{}"}), http.StatusBadRequest},
		{"staging payload too large", "/v1/collections/hostile/points/bulk",
			append(binBody(binBulkFlagPayloads, ids, vecs, nil), 0, 0xFF, 0, 0, 0, 0, 0, 0),
			http.StatusRequestEntityTooLarge},
		{"staging trailing bytes after payloads", "/v1/collections/hostile/points/bulk",
			append(binBody(binBulkFlagPayloads, ids, vecs, []string{"{}", "{}"}), 9), http.StatusBadRequest},
		// Batch route: the payload section is declared but absent / malformed.
		{"payload section missing", "/v1/collections/hostile/points/batch",
			binBody(binBulkFlagPayloads, ids, vecs, nil), http.StatusBadRequest},
		{"payload truncated", "/v1/collections/hostile/points/batch",
			append(binBody(binBulkFlagPayloads, ids, vecs, nil), 0, 0, 0, 8, 'x'), http.StatusBadRequest},
		{"payload not JSON", "/v1/collections/hostile/points/batch",
			binBody(binBulkFlagPayloads, ids, vecs, []string{"not json", "{}"}), http.StatusBadRequest},
		// Both length prefixes are present (so the body clears the Content-Length
		// lower bound) but the first declares a 16 MiB payload for one point.
		{"payload too large", "/v1/collections/hostile/points/batch",
			append(binBody(binBulkFlagPayloads, ids, vecs, nil), 0, 0xFF, 0, 0, 0, 0, 0, 0),
			http.StatusRequestEntityTooLarge},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := doBin(t, h, tc.path, tc.body, nil)
			if rec.Code != tc.want {
				t.Fatalf("status = %d (%s), want %d", rec.Code, strings.TrimSpace(rec.Body.String()), tc.want)
			}
		})
	}

	// Nothing hostile landed: the collection is still empty and buildable.
	if rec := do(t, h, "POST", "/v1/collections/hostile/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build after hostile bodies: %d %s", rec.Code, rec.Body.String())
	}
}

// TestBinaryBulkEmptyBody checks a zero-point body is a well-formed no-op (the
// JSON path accepts an empty points array too).
func TestBinaryBulkEmptyBody(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "empty", 4)

	var res struct{ Staged int }
	if rec := doBin(t, h, "/v1/collections/empty/points/bulk", binBody(0, nil, nil, nil), &res); rec.Code != http.StatusOK {
		t.Fatalf("empty binary stage: %d %s", rec.Code, rec.Body.String())
	}
	if res.Staged != 0 {
		t.Fatalf("staged = %d, want 0", res.Staged)
	}
}

// TestBinaryBulkChunkedBody checks a body with no declared Content-Length (the
// Content-Length pre-check is skipped) is still bounded by the read-layer cap and
// still succeeds when well-formed.
func TestBinaryBulkChunkedBody(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "chunked", 4)

	ids, vecs := testPoints(3, 4)
	r := httptest.NewRequest("POST", "/v1/collections/chunked/points/bulk", bytes.NewReader(binBody(0, ids, vecs, nil)))
	r.Header.Set("Content-Type", "application/octet-stream")
	r.ContentLength = -1
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusOK {
		t.Fatalf("chunked binary stage: %d %s", rec.Code, rec.Body.String())
	}

	// Same, but the header over-declares its count: without Content-Length the
	// lie is only caught at the (short) row read, which must still be a 400.
	var b bytes.Buffer
	b.WriteString(binaryBulkMagic)
	_ = binary.Write(&b, binary.BigEndian, uint32(0))
	_ = binary.Write(&b, binary.BigEndian, uint32(1000))
	_ = binary.Write(&b, binary.BigEndian, uint32(4))
	r2 := httptest.NewRequest("POST", "/v1/collections/chunked/points/bulk", bytes.NewReader(b.Bytes()))
	r2.Header.Set("Content-Type", "application/octet-stream")
	r2.ContentLength = -1
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, r2)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("chunked over-declared count: %d %s, want 400", rec.Code, rec.Body.String())
	}
}
