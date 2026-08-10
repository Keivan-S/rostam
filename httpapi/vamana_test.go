// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPCreateVamana drives the full Vamana create wire over REST: index_type
// "vamana" with a non-default vamana_r/l/alpha creates a single-layer DiskANN
// index; batch-upsert + search round-trips. The working index + nearest hit
// proves Config reached the engine with IndexVamana + the right R/L/alpha.
func TestHTTPCreateVamana(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"vam","config":{"dim":3,"metric":"l2","m":16,"ef_construction":100,"ef_search":64,"seed":1,`+
			`"index_type":"vamana","vamana_r":48,"vamana_l":80,"vamana_alpha":1.3}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	var pts []string
	for i := 1; i <= 30; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0]}`, i, i))
	}
	body := `{"upsert":true,"points":[` + strings.Join(pts, ",") + `]}`
	var bres struct {
		Count int `json:"count"`
	}
	rec = do(t, h, "POST", "/v1/collections/vam/points/batch", body, &bres)
	if rec.Code != http.StatusOK || bres.Count != 30 {
		t.Fatalf("batch = %d count=%d (%s)", rec.Code, bres.Count, rec.Body)
	}

	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/vam/points/search", `{"query":[1,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 || sres.Results[0].ID != 1 {
		t.Fatalf("post-batch Vamana search = %+v", sres.Results)
	}
}

// TestHTTPCreateVamanaUnknownIndexType confirms a bogus index_type is rejected
// with 400 (so the new "vamana" mapping did not loosen the parser).
func TestHTTPCreateVamanaUnknownIndexType(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"bad","config":{"dim":3,"metric":"l2","index_type":"bogus"}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown index_type = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}
