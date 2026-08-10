// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// pqHNSWBulkSearchHTTP creates a PQ-HNSW collection over REST with the given JSON
// config body, bulk-stages + builds 40 vectors, then searches. The post-build
// nearest hit proves the ScaNN knob in the config reached the engine.
func pqHNSWBulkSearchHTTP(t *testing.T, name, configJSON string) {
	t.Helper()
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"`+name+`","config":`+configJSON+`}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	var pts []string
	for i := 1; i <= 40; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0,0,0,0,0,0]}`, i, i))
	}
	rec = do(t, h, "POST", "/v1/collections/"+name+"/points/bulk", `{"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("stage = %d (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/"+name+"/points/bulk/build", `{}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("build = %d (%s)", rec.Code, rec.Body)
	}

	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/"+name+"/points/search", `{"query":[1,0,0,0,0,0,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 || sres.Results[0].ID != 1 {
		t.Fatalf("%s post-build search = %+v, want id 1 first of 5", name, sres.Results)
	}
}

// TestHTTPCreateAnisotropicPQ drives a full anisotropic PQ-HNSW create over REST
// (quant "pq" + anisotropic_eta): the engine trains anisotropic codebooks and
// serves search. The working index proves anisotropic_eta reached the engine.
func TestHTTPCreateAnisotropicPQ(t *testing.T) {
	// L2 so id 1 (v0=1) is the deterministic nearest to query [1,0,...]; anisotropic
	// _eta rides through regardless of metric (it only shapes PQ training).
	pqHNSWBulkSearchHTTP(t, "aniso",
		`{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1,"quant":"pq","quant_pq_m":8,"anisotropic_eta":4}`)
}

// TestHTTPCreate4BitPQ drives a full 4-bit PQ-HNSW create over REST (quant "pq" +
// pq_nbits 4): the engine trains LUT16 codebooks and serves search. The working
// index proves pq_nbits reached the engine.
func TestHTTPCreate4BitPQ(t *testing.T) {
	pqHNSWBulkSearchHTTP(t, "pq4",
		`{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1,"quant":"pq","quant_pq_m":8,"pq_nbits":4}`)
}

// TestHTTPCreateSOARIVF drives a full SOAR-IVF create over REST (index_type "ivf"
// + soar): the engine builds the multi-assignment IVF (low ivf_train_threshold so
// it trains on the batch upsert) and serves search. The nearest hit proves
// soar/soar_lambda reached the engine.
func TestHTTPCreateSOARIVF(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"soar","config":{"dim":8,"metric":"l2","m":16,"ef_construction":100,"ef_search":64,"seed":1,`+
			`"index_type":"ivf","ivf_nlist":8,"ivf_nprobe":8,"ivf_train_threshold":32,"soar":true,"soar_lambda":2}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	var pts []string
	for i := 1; i <= 64; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,%d,0,0,0,0,0,0]}`, i, i, (i%7)+1))
	}
	body := `{"upsert":true,"points":[` + strings.Join(pts, ",") + `]}`
	var bres struct {
		Count int `json:"count"`
	}
	rec = do(t, h, "POST", "/v1/collections/soar/points/batch", body, &bres)
	if rec.Code != http.StatusOK || bres.Count != 64 {
		t.Fatalf("batch = %d count=%d (%s)", rec.Code, bres.Count, rec.Body)
	}

	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/soar/points/search", `{"query":[1,1,0,0,0,0,0,0],"k":5}`, &sres)
	if len(sres.Results) == 0 {
		t.Fatalf("SOAR-IVF search returned no results")
	}
	found := false
	for _, r := range sres.Results {
		if r.ID == 1 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("SOAR-IVF search missed the nearest id 1: %+v", sres.Results)
	}
}

// TestHTTPCreateSOARNonIVFFailLoud proves soar=true on a non-IVF index is rejected
// with 400 (the engine Validate ErrInvalidSOAR surfaced fail-loud at the edge),
// so the new soar mapping did not loosen the gate.
func TestHTTPCreateSOARNonIVFFailLoud(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"soarbad","config":{"dim":8,"metric":"l2","m":16,"soar":true}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("soar on non-IVF = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}
