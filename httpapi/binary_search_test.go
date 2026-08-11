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

// buildRVQ1 assembles a binary query body. Tests mutate the result rather than
// hand-rolling bytes, so a change to the framing breaks them at the assembler
// instead of in fourteen byte-literals.
func buildRVQ1(k int, query []float32, filterJSON string, rc, opa uint8, staleness uint64) []byte {
	var flags uint32
	if filterJSON != "" {
		flags |= binSearchFlagFilter
	}
	b := make([]byte, binarySearchHeaderLen)
	copy(b[0:4], binarySearchMagic)
	binary.BigEndian.PutUint32(b[4:8], flags)
	binary.BigEndian.PutUint32(b[8:12], uint32(k))
	binary.BigEndian.PutUint32(b[12:16], uint32(len(query)))
	b[16], b[17] = rc, opa
	binary.BigEndian.PutUint64(b[20:28], staleness)
	for _, f := range query {
		b = binary.BigEndian.AppendUint32(b, math.Float32bits(f))
	}
	if filterJSON != "" {
		b = binary.BigEndian.AppendUint32(b, uint32(len(filterJSON)))
		b = append(b, filterJSON...)
	}
	return b
}

func decodeRVQ1(t *testing.T, body []byte) (searchReq, *httptest.ResponseRecorder, bool) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/collections/c/points/search", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/octet-stream")
	w := httptest.NewRecorder()
	var req searchReq
	ok := decodeSearchBody(w, r, &req)
	return req, w, ok
}

// The point of the whole file: a binary body must produce exactly the searchReq
// its JSON twin would have, field for field.
func TestBinarySearchMatchesTheJSONBody(t *testing.T) {
	query := []float32{0.5, -1.25, 3, 0.0009765625}
	filter := `{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}`

	bin, rec, ok := decodeRVQ1(t, buildRVQ1(7, query, filter, 1, 1, 42))
	if !ok {
		t.Fatalf("binary body rejected: %d %s", rec.Code, rec.Body.String())
	}

	jsonBody, err := json.Marshal(map[string]any{
		"query": query, "k": 7, "filter": json.RawMessage(filter),
		"read_consistency": 1, "on_partition_unavailable": 1, "max_staleness": 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/collections/c/points/search", bytes.NewReader(jsonBody))
	r.Header.Set("Content-Type", "application/json")
	var want searchReq
	if !decodeSearchBody(httptest.NewRecorder(), r, &want) {
		t.Fatal("JSON body rejected")
	}

	if fmt.Sprint(bin.Query) != fmt.Sprint(want.Query) {
		t.Errorf("query: binary %v, JSON %v", bin.Query, want.Query)
	}
	if bin.K != want.K || bin.ReadConsistency != want.ReadConsistency ||
		bin.OnPartitionUnavailable != want.OnPartitionUnavailable || bin.MaxStaleness != want.MaxStaleness {
		t.Errorf("scalars differ:\n binary %+v\n JSON   %+v", bin, want)
	}
	if fmt.Sprint(bin.Filter) != fmt.Sprint(want.Filter) {
		t.Errorf("filter: binary %+v, JSON %+v", bin.Filter, want.Filter)
	}
}

// A JSON content type must still take the JSON path, unchanged.
func TestJSONBodyStillDecodesWhenBinaryExists(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/collections/c/points/search",
		strings.NewReader(`{"query":[1,2],"k":3}`))
	r.Header.Set("Content-Type", "application/json")
	var req searchReq
	if !decodeSearchBody(httptest.NewRecorder(), r, &req) {
		t.Fatal("JSON body rejected")
	}
	if req.K != 3 || len(req.Query) != 2 {
		t.Errorf("got %+v", req)
	}
}

// The reason maxBinarySearchDim is checked BEFORE dim is multiplied by four.
// On a 32-bit build 0x40000000*4 is exactly 0: without the cap this allocates
// nothing, reads nothing, and hands the engine an empty query as if the client
// had sent one.
func TestBinarySearchRejectsDimThatOverflowsWhenWidened(t *testing.T) {
	for _, dim := range []uint32{0x40000000, 0x80000000, 0xFFFFFFFF, maxBinarySearchDim + 1} {
		body := buildRVQ1(1, []float32{1}, "", 0, 0, 0)
		binary.BigEndian.PutUint32(body[12:16], dim)
		req, w, ok := decodeRVQ1(t, body)
		if ok {
			t.Errorf("dim=%#x accepted, query=%v", dim, req.Query)
			continue
		}
		if w.Code != http.StatusBadRequest {
			t.Errorf("dim=%#x: status %d, want 400", dim, w.Code)
		}
	}
}

// k above MaxInt32 must be refused outright rather than wrapping negative and
// arriving at validTopK as a plausible-looking "k <= 0".
func TestBinarySearchRejectsKAboveInt32(t *testing.T) {
	body := buildRVQ1(1, []float32{1}, "", 0, 0, 0)
	binary.BigEndian.PutUint32(body[8:12], math.MaxInt32+1)
	if _, w, ok := decodeRVQ1(t, body); ok || w.Code != http.StatusBadRequest {
		t.Errorf("ok=%v status=%d, want rejected with 400", ok, w.Code)
	}
}

// NaN and Inf are unrepresentable in the JSON body, so accepting them here would
// let the binary encoding say something its twin cannot — and a NaN coordinate
// poisons every distance it touches, producing a wrong ranking, not an error.
func TestBinarySearchRejectsNonFiniteFloats(t *testing.T) {
	for name, bits := range map[string]uint32{
		"NaN":  math.Float32bits(float32(math.NaN())),
		"+Inf": math.Float32bits(float32(math.Inf(1))),
		"-Inf": math.Float32bits(float32(math.Inf(-1))),
	} {
		body := buildRVQ1(1, []float32{1, 2}, "", 0, 0, 0)
		binary.BigEndian.PutUint32(body[binarySearchHeaderLen+4:], bits)
		if _, w, ok := decodeRVQ1(t, body); ok || w.Code != http.StatusBadRequest {
			t.Errorf("%s: ok=%v status=%d, want rejected with 400", name, ok, w.Code)
		}
	}
}

func TestBinarySearchRejectsMalformedFrames(t *testing.T) {
	good := buildRVQ1(5, []float32{1, 2, 3}, "", 0, 0, 0)

	cases := map[string][]byte{
		"bad magic":       append([]byte("XXXX"), good[4:]...),
		"short header":    good[:10],
		"truncated body":  good[:len(good)-6],
		"trailing bytes":  append(append([]byte{}, good...), 0x00),
		"empty body":      {},
		"filter len lies": buildRVQ1(5, []float32{1}, `{"op":"eq","field":"a","value":{"kind":"int","int":1}}`, 0, 0, 0)[:binarySearchHeaderLen+4+2],
	}
	unknownFlags := append([]byte{}, good...)
	binary.BigEndian.PutUint32(unknownFlags[4:8], 1<<31)
	cases["unknown flags"] = unknownFlags

	reserved := append([]byte{}, good...)
	binary.BigEndian.PutUint16(reserved[18:20], 1)
	cases["reserved not zero"] = reserved

	for name, body := range cases {
		_, w, ok := decodeRVQ1(t, body)
		if ok {
			t.Errorf("%s: accepted, want rejected", name)
			continue
		}
		if w.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400", name, w.Code)
		}
		if !strings.Contains(w.Body.String(), "binary query body") {
			t.Errorf("%s: error does not name this framing: %s", name, w.Body.String())
		}
	}
}

// An oversized filter must be refused from its declared length, before the
// server allocates for it.
func TestBinarySearchRejectsOversizeFilter(t *testing.T) {
	body := buildRVQ1(1, []float32{1}, `{"op":"eq","field":"a","value":{"kind":"int","int":1}}`, 0, 0, 0)
	binary.BigEndian.PutUint32(body[binarySearchHeaderLen+4:], maxBinarySearchFilter+1)
	if _, w, ok := decodeRVQ1(t, body); ok || w.Code != http.StatusBadRequest {
		t.Errorf("ok=%v status=%d, want rejected with 400", ok, w.Code)
	}
}

// A zero-dimension query is well-formed framing; it fails later, on the same
// path the JSON body fails on, rather than here.
func TestBinarySearchAcceptsEmptyQuery(t *testing.T) {
	req, _, ok := decodeRVQ1(t, buildRVQ1(3, nil, "", 0, 0, 0))
	if !ok {
		t.Fatal("empty query rejected at the framing layer")
	}
	if len(req.Query) != 0 || req.K != 3 {
		t.Errorf("got %+v", req)
	}
}
