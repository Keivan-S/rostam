// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// TestClusterServerVectors brings up a replicated (Raft) server over HTTP and
// drives vectors through it: multiple collections (partitioned across shards by
// name) created, upserted, and searched. Proves the network server runs in
// replicated mode and vectors route + serve correctly.
func TestClusterServerVectors(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewServer(rostam.ServerConfig{
		Cluster: &rostam.EmbeddedConfig{
			NodeID:    "n1",
			DataDir:   t.TempDir(),
			NumShards: 8, // collections partition across these shards by name
			Bootstrap: true,
			Ops:       reg,
		},
		HTTPAddr: "127.0.0.1:0",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	base := "http://" + srv.HTTPAddr()

	post := func(path, body string) (int, []byte) {
		resp, err := http.Post(base+path, "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			return 0, nil
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, b
	}

	// Writes return 503 ("not leader") until the target shard's Raft group has
	// elected a leader. Each collection routes to its own shard (an independent
	// Raft group with independent election), so every write retries on 503.
	postWrite := func(path, body string, want int) {
		t.Helper()
		deadline := time.Now().Add(20 * time.Second)
		var code int
		var b []byte
		for time.Now().Before(deadline) {
			code, b = post(path, body)
			if code != http.StatusServiceUnavailable {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		if code != want {
			t.Fatalf("%s = %d, want %d (%s)", path, code, want, b)
		}
	}

	// Two collections — each partitions to its own shard by name.
	postWrite("/v1/collections", `{"name":"docs","config":{"dim":3,"metric":"l2"}}`, http.StatusCreated)
	postWrite("/v1/collections", `{"name":"other","config":{"dim":3,"metric":"l2"}}`, http.StatusCreated)

	// Upsert into both collections.
	for _, col := range []string{"docs", "other"} {
		for i := 1; i <= 4; i++ {
			body := `{"id":` + itoa(i) + `,"vector":[` + itoa(i) + `,0,0],"content":"chunk","upsert":true}`
			postWrite("/v1/collections/"+col+"/points", body, http.StatusOK)
		}
	}

	// Search each collection (read served from the replicated shard store).
	for _, col := range []string{"docs", "other"} {
		code, b := post("/v1/collections/"+col+"/points/search", `{"query":[1,0,0],"k":3}`)
		if code != http.StatusOK {
			t.Fatalf("search %s = %d (%s)", col, code, b)
		}
		var sr struct {
			Results []struct {
				ID uint64 `json:"id"`
			} `json:"results"`
		}
		if err := json.Unmarshal(b, &sr); err != nil {
			t.Fatal(err)
		}
		if len(sr.Results) != 3 || sr.Results[0].ID != 1 {
			t.Errorf("search %s = %+v, want id 1 first of 3", col, sr.Results)
		}
	}
}
