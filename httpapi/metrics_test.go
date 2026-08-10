// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"strings"
	"testing"
)

// TestHTTPMetrics covers the Prometheus scrape surface: after inserting points
// into two collections, both /metrics and /v1/metrics return 200 with the
// Prometheus content type and an exposition body carrying each collection's
// labeled series.
func TestHTTPMetrics(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	for _, name := range []string{"docs", "images"} {
		rec := do(t, h, "POST", "/v1/collections",
			`{"name":"`+name+`","config":{"dim":3,"metric":"l2"}}`, nil)
		if rec.Code != http.StatusCreated {
			t.Fatalf("create %s = %d (%s)", name, rec.Code, rec.Body)
		}
		rec = do(t, h, "POST", "/v1/collections/"+name+"/points",
			`{"id":1,"vector":[1,0,0]}`, nil)
		if rec.Code != http.StatusOK && rec.Code != http.StatusCreated {
			t.Fatalf("insert into %s = %d (%s)", name, rec.Code, rec.Body)
		}
	}

	for _, path := range []string{"/metrics", "/v1/metrics"} {
		rec := do(t, h, "GET", path, "", nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s = %d (%s)", path, rec.Code, rec.Body)
		}
		if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain; version=0.0.4") {
			t.Fatalf("GET %s content-type = %q", path, ct)
		}
		body := rec.Body.String()
		for _, want := range []string{
			`rostam_vector_size{collection="default/docs"}`,
			`rostam_vector_size{collection="default/images"}`,
			"rostam_vector_insert_ops_total",
		} {
			if !strings.Contains(body, want) {
				t.Fatalf("GET %s body missing %q\n---\n%s", path, want, body)
			}
		}
	}
}
