// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strings"
	"testing"
)

// TestWritePrometheus checks the Prometheus exposition output: counters/gauges
// reflect activity, the collection label is present and escaped, and the
// latency histograms expose cumulative buckets that sum to the count.
func TestWritePrometheus(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 20; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 5; i++ {
		if _, err := h.Search([]float32{1, 0, 0, 0}, 3); err != nil {
			t.Fatal(err)
		}
	}

	var sb strings.Builder
	if err := h.Stats().WritePrometheus(&sb, `my"coll`); err != nil {
		t.Fatal(err)
	}
	out := sb.String()

	checks := []string{
		`# TYPE rostam_vector_size gauge`,
		`rostam_vector_size{collection="my\"coll"} 20`,
		`# TYPE rostam_vector_search_ops_total counter`,
		`rostam_vector_search_ops_total{collection="my\"coll"} 5`,
		`# TYPE rostam_vector_insert_ops_total counter`,
		`# TYPE rostam_vector_search_latency_seconds histogram`,
		`rostam_vector_search_latency_seconds_bucket{collection="my\"coll",le="+Inf"} 5`,
		`rostam_vector_search_latency_seconds_count{collection="my\"coll"} 5`,
	}
	for _, c := range checks {
		if !strings.Contains(out, c) {
			t.Errorf("Prometheus output missing line:\n  %s\n--- full output ---\n%s", c, out)
		}
	}
}
