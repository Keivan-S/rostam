// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"sort"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPTopology proves GET /v1/topology dispatches __topology__ and renders
// the decoded routing snapshot: shard count, members (node_id + server_addr),
// per-shard leader addrs, and per-shard owner placement. The recording dispatcher
// returns a canned gob-encoded Topology so the JSON shape is asserted at the edge.
func TestHTTPTopology(t *testing.T) {
	topo := ops.Topology{
		NumShards: 2,
		Members: []ops.TopologyMember{
			{NodeID: "n1", ServerAddr: "10.0.0.1:6543"},
			{NodeID: "n2", ServerAddr: "10.0.0.2:6543"},
		},
		Leaders:   []string{"10.0.0.1:6543", "10.0.0.2:6543"},
		Placement: [][]string{{"n1", "n2"}, {"n2", "n1"}},
	}
	enc, err := ops.EncodeTopology(topo)
	if err != nil {
		t.Fatalf("encode topology: %v", err)
	}
	disp := &recordingDispatcher{result: enc}
	h := Handler(disp, Options{})

	var out topologyResponse
	rec := do(t, h, "GET", "/v1/topology", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("topology = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "__topology__" {
		t.Fatalf("calls = %+v, want one __topology__", disp.calls)
	}
	if out.NumShards != 2 {
		t.Fatalf("num_shards = %d, want 2", out.NumShards)
	}
	if len(out.Members) != 2 || out.Members[0].NodeID != "n1" || out.Members[0].ServerAddr != "10.0.0.1:6543" {
		t.Fatalf("members = %+v", out.Members)
	}
	if len(out.Leaders) != 2 || out.Leaders[0] != "10.0.0.1:6543" {
		t.Fatalf("leaders = %+v", out.Leaders)
	}
	if len(out.Placement) != 2 || len(out.Placement[0]) != 2 || out.Placement[0][0] != "n1" {
		t.Fatalf("placement = %+v", out.Placement)
	}
}

// TestHTTPCollections covers GET /v1/collections: after creating two collections
// they both appear in the {"collections":[{"name":..}]} list, sourced from the
// same enumeration __metrics__ uses.
func TestHTTPCollections(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	for _, name := range []string{"docs", "images"} {
		rec := do(t, h, "POST", "/v1/collections",
			`{"name":"`+name+`","config":{"dim":3,"metric":"l2"}}`, nil)
		if rec.Code != http.StatusCreated {
			t.Fatalf("create %s = %d (%s)", name, rec.Code, rec.Body)
		}
	}

	var out struct {
		Collections []collectionRef `json:"collections"`
	}
	rec := do(t, h, "GET", "/v1/collections", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("collections = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	got := make([]string, 0, len(out.Collections))
	for _, c := range out.Collections {
		got = append(got, c.Name)
	}
	sort.Strings(got)
	// The enumeration source is CollectionStore.CollectionNames() — the SAME one
	// __metrics__ labels — so names are tenant-qualified ("default/<name>"),
	// matching the Prometheus collection="default/docs" labels.
	if len(got) != 2 || got[0] != "default/docs" || got[1] != "default/images" {
		t.Fatalf("collections = %v, want [default/docs default/images]", got)
	}
}

// TestHTTPCollectionsEmpty proves the list is a non-nil empty array on a store
// with no collections (JSON [] not null).
func TestHTTPCollectionsEmpty(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "GET", "/v1/collections", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("collections = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if body := strings.TrimSpace(rec.Body.String()); !strings.Contains(body, `"collections":[]`) {
		t.Fatalf("empty collections body = %q, want []", body)
	}
}

// TestHTTPCollectionConfig covers GET /v1/collections/{name}: it wraps
// vector_get_config and renders the config with readable metric/quant/index_type
// strings, round-tripping the values a create set.
func TestHTTPCollectionConfig(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":8,"metric":"dot","m":12,"ef_construction":80,"ef_search":40,"quant":"sq8"}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	var out struct {
		Name   string           `json:"name"`
		Config collectionConfig `json:"config"`
	}
	rec = do(t, h, "GET", "/v1/collections/docs", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("get config = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if out.Name != "docs" {
		t.Fatalf("name = %q, want docs", out.Name)
	}
	if out.Config.Dim != 8 {
		t.Fatalf("dim = %d, want 8", out.Config.Dim)
	}
	if out.Config.Metric != "dot" {
		t.Fatalf("metric = %q, want dot", out.Config.Metric)
	}
	if out.Config.Quant != "sq8" {
		t.Fatalf("quant = %q, want sq8", out.Config.Quant)
	}
	if out.Config.M != 12 || out.Config.EfConstruction != 80 || out.Config.EfSearch != 40 {
		t.Fatalf("m/efc/efs = %d/%d/%d, want 12/80/40", out.Config.M, out.Config.EfConstruction, out.Config.EfSearch)
	}
	// A default (unspecified) index type renders as the readable "hnsw".
	if out.Config.IndexType != "hnsw" {
		t.Fatalf("index_type = %q, want hnsw", out.Config.IndexType)
	}
}

// TestHTTPCollectionConfigUnknown proves an unknown collection surfaces the op's
// 404 through the standard dispatch-error mapping.
func TestHTTPCollectionConfigUnknown(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "GET", "/v1/collections/nope", "", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown collection config = %d, want 404 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPDashboardStatic covers the embedded SPA mount: the bare /dashboard
// 301-redirects to /dashboard/, the index is served at the root, and an unknown
// deep link falls back to the same index.html (client-side routing) rather than
// 404ing. The checked-in placeholder is enough to assert the fallback wiring.
func TestHTTPDashboardStatic(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	// /dashboard -> 301 /dashboard/.
	rec := do(t, h, "GET", "/dashboard", "", nil)
	if rec.Code != http.StatusMovedPermanently {
		t.Fatalf("/dashboard = %d, want 301 (%s)", rec.Code, rec.Body)
	}
	if loc := rec.Header().Get("Location"); loc != "/dashboard/" {
		t.Fatalf("/dashboard Location = %q, want /dashboard/", loc)
	}

	// Root serves the SPA shell. The marker "Rostam" is build-agnostic: it is in
	// both the compiled index.html (<title>Rostam Dashboard</title>) and the
	// built-in unbuilt placeholder, so this passes whether or not `make ui` ran.
	rec = do(t, h, "GET", "/dashboard/", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("/dashboard/ = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "Rostam") {
		t.Fatalf("/dashboard/ body missing shell marker:\n%s", rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("/dashboard/ content-type = %q, want text/html", ct)
	}

	// An unknown deep link (client-side route) falls back to the SPA shell.
	rec = do(t, h, "GET", "/dashboard/collections/docs", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("deep link = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "Rostam") {
		t.Fatalf("deep-link fallback missing SPA shell:\n%s", rec.Body)
	}
}

// TestFromConfigEnumStrings proves the reverse enum helpers map every metric,
// quant and index-type enum onto the readable string the create API accepts, so a
// config round-trips (fromConfig -> toConfig) without loss.
func TestFromConfigEnumStrings(t *testing.T) {
	metrics := map[vector.Metric]string{vector.Cosine: "cosine", vector.L2: "l2", vector.DotProduct: "dot"}
	for m, want := range metrics {
		if got := metricString(m); got != want {
			t.Errorf("metricString(%v) = %q, want %q", m, got, want)
		}
	}
	quants := map[vector.QuantMode]string{
		vector.QuantNone: "none", vector.QuantSQ8: "sq8", vector.QuantBQ1: "bq1",
		vector.QuantPQ: "pq", vector.QuantSQ: "sq", vector.QuantPRQ: "prq",
	}
	for q, want := range quants {
		if got := quantString(q); got != want {
			t.Errorf("quantString(%v) = %q, want %q", q, got, want)
		}
	}
	indexes := map[vector.IndexType]string{vector.IndexHNSW: "hnsw", vector.IndexIVF: "ivf", vector.IndexVamana: "vamana"}
	for it, want := range indexes {
		if got := indexTypeString(it); got != want {
			t.Errorf("indexTypeString(%v) = %q, want %q", it, got, want)
		}
	}
}
