// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/httpapi"
	"github.com/rostamlabs/rostam/vector"
)

// TestRemoteSearchDocsResponseBytesMatchSingleNode is the REMOTE half of the
// search_docs raw-rendering change.
//
// The HTTP layer renders documents by splicing each hit's metadata straight from
// the op result wire (ops.DecodeVectorDocsDegradedRaw) instead of decoding it just
// to re-encode it. That is only safe if the wire is unchanged — including on the
// path where the bytes genuinely cross a process boundary, which is the whole
// reason the []byte op contract exists.
//
// So: run a REAL 3-node cluster, put the same 300 documents in a P=6 collection
// (partitions spread across all three nodes, so a fan-out response is assembled
// from bytes that arrived over the network) and in a P=1 collection (served
// entirely locally). Issue the same query against both, through a real HTTP
// server, and require the RESPONSE BODIES TO BE BYTE-IDENTICAL.
//
// Byte-identical is a much stronger claim than "same documents": it says the
// remote partitions' metadata survived marshal → network → unmarshal → merge →
// re-marshal → splice with not one character of escaping, ordering or number
// formatting changed against the all-local path.
func TestRemoteSearchDocsResponseBytesMatchSingleNode(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const (
		fanned = "rawbytes_docs_p6"
		single = "rawbytes_docs_p1"
		n      = 300
	)

	create := func(name string, p int) {
		createCollectionTolerant(t, ctx, stores[0], name, rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: p})
	}
	create(fanned, 6)
	create(single, 1)

	// Metadata chosen so the RENDERING is load-bearing: characters encoding/json
	// escapes by default, non-ASCII, a value that is itself JSON text, and keys that
	// must sort the same way on both paths.
	meta := func(id uint64) vector.Metadata {
		return vector.Metadata{
			"title":  vector.NewString(fmt.Sprintf("<b>doc %d</b> & more", id)),
			"cmp":    vector.NewString("a < b && b > c"),
			"emoji":  vector.NewString("🎉 日本語 ünïcödé"),
			"nested": vector.NewString(`{"a":[1,2],"b":null}`),
			"tags":   vector.NewStrings([]string{"x", "<y>", "&z"}),
			"ratio":  vector.NewFloat(float64(id) / 3),
			"big":    vector.NewFloat(1e21),
			"even":   vector.NewBool(id%2 == 0),
			"at":     vector.NewGeo(-33.8688, 151.2093),
		}
	}
	for id := uint64(1); id <= n; id++ {
		v := []float32{float32(id), 0, 0, 0}
		content := fmt.Sprintf("chunk %d <b>&</b> 🎉", id)
		md, idc := meta(id), id
		retryUntil(t, "upsert fanned", func() error {
			return stores[0].VectorUpsert(ctx, fanned, idc, v, content, rostam.VectorInsertOpts{Metadata: md})
		})
		retryUntil(t, "upsert single", func() error {
			return stores[0].VectorUpsert(ctx, single, idc, v, content, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Coordinate from node 1 — a node that did NOT create either collection — so the
	// fan-out really does gather partitions hosted by its peers.
	e1 := stores[1].(*rostam.Embedded)
	// Only the PARTITIONED collection needs a catalog wait: an unpartitioned one has
	// no partition entry to converge on (PartitionsGen reports ok=false for it), and
	// routing it does not depend on the catalog.
	waitEmbeddedCatalog(t, e1, fanned, 6, 5*time.Second)

	ts := httptest.NewServer(httpapi.Handler(rostam.NewFanoutDispatcher(e1, e1.Node()), httpapi.Options{}))
	t.Cleanup(ts.Close)

	post := func(path, body string) []byte {
		t.Helper()
		resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		defer func() { _ = resp.Body.Close() }()
		out, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("POST %s: read body: %v", path, err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("POST %s = %d (%s)", path, resp.StatusCode, out)
		}
		return out
	}

	// 0.5 < 1 keeps L2 distances strictly increasing in id, so both responses have
	// one unambiguous ordering and a byte comparison is meaningful.
	for _, k := range []int{1, 10, 100} {
		req := fmt.Sprintf(`{"query":[0.5,0,0,0],"k":%d}`, k)
		gotFanned := post("/v1/collections/"+fanned+"/points/search/docs", req)
		gotSingle := post("/v1/collections/"+single+"/points/search/docs", req)

		if !bytes.Equal(gotFanned, gotSingle) {
			t.Fatalf("k=%d: cross-node fan-out response != single-partition response\n fanned: %s\n single: %s",
				k, gotFanned, gotSingle)
		}
		// Guard against a vacuous pass: the bodies must actually carry documents,
		// and carry them with encoding/json's HTML escaping intact — that escaped
		// form is precisely what the splice has to reproduce.
		if !bytes.Contains(gotSingle, []byte("\\u003cb\\u003edoc ")) {
			t.Fatalf("k=%d: response carries no escaped metadata (vacuous): %s", k, gotSingle)
		}
	}
}
