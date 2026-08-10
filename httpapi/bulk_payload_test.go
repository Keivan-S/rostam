// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// THE FILTER CASE MUST BE ABLE TO USE THE BULK ROUTE.
//
// /points/bulk carried ids and vectors and nothing else, so any workload that
// needed a payload per point had to fall back to /points/batch — one indexed
// insert per point, measured ~6x slower to searchable on 1M x 768d. Both of this
// route's encodings now carry payloads through to the multi-core build, and the
// tests here hold them to the only standard that matters: the same corpus loaded
// through the bulk route and through the inline route must answer filtered
// searches identically.

// bulkPayloadJSON is one point's metadata in Rostam's tagged form, as both the
// JSON body and the RVB1 payload section spell it.
func bulkPayloadJSON(id uint64) string {
	return fmt.Sprintf(`{"id":{"kind":"int","int":%d},"bucket":{"kind":"string","str":"b%d"}}`,
		id, id%5)
}

// bulkFilterCases is the filter battery. Each must return the same id set from a
// bulk-loaded collection and an inline-loaded one — including the two degenerate
// ends, where an index that returns its candidate set instead of its matching set
// gives itself away.
func bulkFilterCases() []struct{ name, filter string } {
	return []struct{ name, filter string }{
		{"eq-int", `{"op":"eq","field":"id","value":{"kind":"int","int":7}}`},
		{"eq-string", `{"op":"eq","field":"bucket","value":{"kind":"string","str":"b2"}}`},
		{"range", `{"op":"gte","field":"id","value":{"kind":"int","int":30}}`},
		{"in", `{"op":"in","field":"bucket","value":{"kind":"strings","strs":["b1","b3"]}}`},
		{"match", `{"op":"match","field":"bucket","value":{"kind":"string","str":"b4"}}`},
		{"and", `{"op":"and","and":[
			{"op":"gte","field":"id","value":{"kind":"int","int":10}},
			{"op":"lt","field":"id","value":{"kind":"int","int":40}}]}`},
		{"matches-nothing", `{"op":"eq","field":"bucket","value":{"kind":"string","str":"nope"}}`},
		{"matches-everything", `{"op":"gte","field":"id","value":{"kind":"int","int":0}}`},
	}
}

// filteredIDs runs a filtered search and returns the ids, sorted, never nil.
func filteredIDs(t *testing.T, h http.Handler, coll string, vec []float32, k int, filter string) []uint64 {
	t.Helper()
	vb, _ := json.Marshal(vec)
	body := fmt.Sprintf(`{"query":%s,"k":%d,"filter":%s}`, vb, k, filter)
	var hits struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}
	rec := do(t, h, "POST", "/v1/collections/"+coll+"/points/search", body, &hits)
	if rec.Code != http.StatusOK {
		t.Fatalf("filtered search on %s: %d %s", coll, rec.Code, rec.Body.String())
	}
	out := []uint64{}
	for _, r := range hits.Results {
		out = append(out, r.ID)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func pointPayload(t *testing.T, h http.Handler, coll string, id uint64) map[string]any {
	t.Helper()
	var got struct {
		Payload map[string]any `json:"payload"`
	}
	rec := do(t, h, "GET", fmt.Sprintf("/v1/collections/%s/points/%d", coll, id), "", &got)
	if rec.Code != http.StatusOK {
		t.Fatalf("get %s/%d: %d %s", coll, id, rec.Code, rec.Body.String())
	}
	return got.Payload
}

// TestBulkPayloadRoutesMatchInlineRoute loads one corpus three ways — inline
// /points/batch, JSON /points/bulk, and binary /points/bulk — and requires all
// three to be indistinguishable under every filter in the battery.
func TestBulkPayloadRoutesMatchInlineRoute(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	const n, dim = 60, 16
	ids, vecs := testPoints(n, dim)
	for _, c := range []string{"inlinepay", "jsonpay", "binpay"} {
		createColl(t, h, c, dim)
	}

	// Inline: the reference path, one indexed insert per point.
	pts := make([]string, n)
	for i := range ids {
		vb, _ := json.Marshal(vecs[i])
		pts[i] = fmt.Sprintf(`{"id":%d,"vector":%s,"metadata":%s}`, ids[i], vb, bulkPayloadJSON(ids[i]))
	}
	if rec := do(t, h, "POST", "/v1/collections/inlinepay/points/batch",
		`{"points":[`+strings.Join(pts, ",")+`]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("inline batch: %d %s", rec.Code, rec.Body.String())
	}

	// JSON bulk staging, now carrying metadata.
	var jres struct{ Staged int }
	if rec := do(t, h, "POST", "/v1/collections/jsonpay/points/bulk",
		`{"points":[`+strings.Join(pts, ",")+`]}`, &jres); rec.Code != http.StatusOK {
		t.Fatalf("json bulk stage with payloads: %d %s", rec.Code, rec.Body.String())
	}

	// Binary bulk staging with the payload flag set.
	payloads := make([]string, n)
	for i, id := range ids {
		payloads[i] = bulkPayloadJSON(id)
	}
	var bres struct{ Staged int }
	if rec := doBin(t, h, "/v1/collections/binpay/points/bulk",
		binBody(binBulkFlagPayloads, ids, vecs, payloads), &bres); rec.Code != http.StatusOK {
		t.Fatalf("binary bulk stage with payloads: %d %s", rec.Code, rec.Body.String())
	}
	if jres.Staged != n || bres.Staged != n {
		t.Fatalf("staged counts: json=%d binary=%d want %d", jres.Staged, bres.Staged, n)
	}

	for _, c := range []string{"jsonpay", "binpay"} {
		if rec := do(t, h, "POST", "/v1/collections/"+c+"/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
			t.Fatalf("build %s: %d %s", c, rec.Code, rec.Body.String())
		}
	}

	// Every point's payload must come back identically on all three.
	for _, id := range ids {
		want := pointPayload(t, h, "inlinepay", id)
		if len(want) == 0 {
			t.Fatalf("id %d: the inline reference stored no payload — the comparison below "+
				"would be vacuous", id)
		}
		for _, c := range []string{"jsonpay", "binpay"} {
			if got := pointPayload(t, h, c, id); !reflect.DeepEqual(got, want) {
				t.Fatalf("id %d on %s: payload %v, inline reference has %v", id, c, got, want)
			}
		}
	}

	for _, tc := range bulkFilterCases() {

		t.Run(tc.name, func(t *testing.T) {
			want := filteredIDs(t, h, "inlinepay", vecs[3], n, tc.filter)
			switch tc.name {
			case "matches-nothing":
				if len(want) != 0 {
					t.Fatalf("the matches-nothing filter matched %v", want)
				}
			case "matches-everything":
				if len(want) != n {
					t.Fatalf("the matches-everything filter matched %d of %d", len(want), n)
				}
			default:
				if len(want) == 0 || len(want) == n {
					t.Fatalf("filter %s selects %d of %d points — it discriminates nothing and "+
						"the comparison below would be vacuous", tc.name, len(want), n)
				}
			}
			for _, c := range []string{"jsonpay", "binpay"} {
				if got := filteredIDs(t, h, c, vecs[3], n, tc.filter); !reflect.DeepEqual(got, want) {
					t.Fatalf("filter %s on %s: %v, inline reference: %v", tc.name, c, got, want)
				}
			}
		})
	}
}

// TestBulkStillRefusesWhatItCannotCarry: metadata now rides the staging wire,
// but content and sparse vectors still have no bulk representation. They must be
// REFUSED, not accepted and dropped — silently discarding them is how a caller
// ends up querying data that was never stored.
func TestBulkStillRefusesWhatItCannotCarry(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "refuse", 4)

	for _, tc := range []struct{ name, point string }{
		{"content", `{"id":1,"vector":[1,0,0,0],"content":"hello"}`},
		{"sparse", `{"id":1,"vector":[1,0,0,0],"sparse":{"indices":[1],"values":[1]}}`},
		{"content-with-metadata", `{"id":1,"vector":[1,0,0,0],"content":"hi","metadata":{"a":{"kind":"int","int":1}}}`},
		// key_ttl_ms is the case this route made REACHABLE by accepting metadata:
		// a per-key payload TTL is meaningless without a payload, so before this
		// change the point was 400'd for its metadata and the key TTL never got a
		// chance to be dropped. A caller sending both must not get a 200 and keys
		// that never expire.
		{"key-ttl-with-metadata", `{"id":1,"vector":[1,0,0,0],` +
			`"metadata":{"pii":{"kind":"string","str":"x"}},"key_ttl_ms":{"pii":86400000}}`},
		{"key-ttl-alone", `{"id":1,"vector":[1,0,0,0],"key_ttl_ms":{"pii":86400000}}`},
		{"ttl", `{"id":1,"vector":[1,0,0,0],"ttl_ms":60000}`},
		{"expected-version", `{"id":1,"vector":[1,0,0,0],"expected_version":0}`},
	} {

		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, "POST", "/v1/collections/refuse/points/bulk", `{"points":[`+tc.point+`]}`, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("staging a point with %s: %d %s (want 400)", tc.name, rec.Code, rec.Body.String())
			}
		})
	}

	// Upsert is still refused on this route, with or without payloads.
	rec := doBin(t, h, "/v1/collections/refuse/points/bulk",
		binBody(binBulkFlagPayloads|binBulkFlagUpsert, []uint64{1}, [][]float32{{1, 0, 0, 0}},
			[]string{`{"a":{"kind":"int","int":1}}`}), nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("binary bulk upsert: %d %s (want 400)", rec.Code, rec.Body.String())
	}
}

// TestBulkBinaryPayloadTruncationRejected holds the streamed payload section to
// the same bounds discipline as the rest of this transport: a declared length
// with no bytes behind it is a 400, not a reservation.
func TestBulkBinaryPayloadTruncationRejected(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "trunc", 4)

	ids := []uint64{1, 2}
	vecs := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	full := binBody(binBulkFlagPayloads, ids, vecs, []string{`{"a":{"kind":"int","int":1}}`, `{}`})

	for _, tc := range []struct {
		name string
		body []byte
	}{
		// The last payload's bytes never arrive.
		{"payload-cut", full[:len(full)-2]},
		// The payload section is missing entirely though the flag claims it.
		{"section-missing", full[:binaryBulkHeaderLen+2*(8+4*4)]},
		// Extra bytes after the last payload.
		{"trailing", append(append([]byte(nil), full...), 0xff)},
	} {

		t.Run(tc.name, func(t *testing.T) {
			rec := doBin(t, h, "/v1/collections/trunc/points/bulk", tc.body, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("%s: %d %s (want 400)", tc.name, rec.Code, rec.Body.String())
			}
		})
	}

	// Nothing may have been staged by any of the rejected bodies.
	if rec := do(t, h, "POST", "/v1/collections/trunc/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build after rejections: %d %s", rec.Code, rec.Body.String())
	}
	if got := filteredIDs(t, h, "trunc", []float32{1, 0, 0, 0}, 10, `{"op":"gte","field":"id","value":{"kind":"int","int":0}}`); len(got) != 0 {
		t.Fatalf("rejected bodies left %v staged and built", got)
	}
}
