// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// WritePrometheus renders the stats snapshot in the Prometheus text exposition
// format, tagging every series with a collection="<collection>" label. Metric
// names are prefixed rostam_vector_. Counters use the _total suffix; the two
// latency distributions are emitted as Prometheus histograms (cumulative
// _bucket series in seconds, plus _sum and _count).
func (s Stats) WritePrometheus(w io.Writer, collection string) error {
	label := `collection="` + escapeLabelValue(collection) + `"`
	bw := &errWriter{w: w}

	gauge := func(name, help string, v int) {
		bw.printf("# HELP rostam_vector_%s %s\n", name, help)
		bw.printf("# TYPE rostam_vector_%s gauge\n", name)
		bw.printf("rostam_vector_%s{%s} %d\n", name, label, v)
	}
	counter := func(name, help string, v uint64) {
		bw.printf("# HELP rostam_vector_%s %s\n", name, help)
		bw.printf("# TYPE rostam_vector_%s counter\n", name)
		bw.printf("rostam_vector_%s{%s} %d\n", name, label, v)
	}

	gauge("size", "Live (non-tombstoned) vectors.", s.Size)
	gauge("tombstoned", "Tombstoned (logically deleted) vectors.", s.Tombstoned)
	gauge("sparse_vectors", "Live vectors carrying a sparse vector.", s.SparseVectors)
	counter("search_ops_total", "Cumulative search operations.", s.SearchOps)
	counter("insert_ops_total", "Cumulative insert operations.", s.InsertOps)
	counter("expired_total", "Vectors filtered or swept due to TTL.", s.Expired)
	counter("quota_rejects_total", "Inserts rejected by a quota or rate limit.", s.QuotaRejects)
	counter("filter_rejects_total", "Candidates rejected by an active search filter.", s.FilterRejects)
	counter("filter_gates_total", "Filtered searches that armed the payload-index bitset admission gate.", s.FilterGates)
	counter("filter_complement_gates_total", "Filtered searches whose admission gate was built from the filter's rejection side.", s.ComplementGates)
	counter("filter_column_gates_total", "Filtered searches whose admission was answered by the numeric column sidecar.", s.ColumnGates)
	counter("filter_column_drops_total", "Times an insert reclaimed the numeric column sidecar to stay inside MaxBytes.", s.ColumnDrops)

	writeLatencyHistogram(bw, "search_latency_seconds", "Per-search wall time.", label, s.SearchLatency)
	writeLatencyHistogram(bw, "insert_latency_seconds", "Per-insert wall time.", label, s.InsertLatency)

	return bw.err
}

// writeLatencyHistogram emits a LatencyHistogram as a Prometheus histogram. The
// stored buckets are per-bucket counts in microseconds; Prometheus requires
// cumulative counts and seconds for the le bounds.
func writeLatencyHistogram(bw *errWriter, name, help, label string, h LatencyHistogram) {
	bw.printf("# HELP rostam_vector_%s %s\n", name, help)
	bw.printf("# TYPE rostam_vector_%s histogram\n", name)
	var cum uint64
	for i, b := range h.Buckets {
		cum += b
		le := "+Inf"
		if latencyBucketBounds[i] != ^uint64(0) {
			le = strconv.FormatFloat(float64(latencyBucketBounds[i])/1e6, 'g', -1, 64)
		}
		bw.printf("rostam_vector_%s_bucket{%s,le=%q} %d\n", name, label, le, cum)
	}
	bw.printf("rostam_vector_%s_sum{%s} %g\n", name, label, float64(h.Sum)/1e6)
	bw.printf("rostam_vector_%s_count{%s} %d\n", name, label, h.Count)
}

// escapeLabelValue escapes a Prometheus label value (backslash, double-quote,
// newline) per the exposition format.
func escapeLabelValue(s string) string {
	if !strings.ContainsAny(s, "\\\"\n") {
		return s
	}
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return r.Replace(s)
}

// errWriter defers io errors so the metric-emitting code stays linear.
type errWriter struct {
	w   io.Writer
	err error
}

func (e *errWriter) printf(format string, args ...any) {
	if e.err != nil {
		return
	}
	_, e.err = fmt.Fprintf(e.w, format, args...)
}
