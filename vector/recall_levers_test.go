// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

// RECALL/QPS PARETO HARNESS for the build-time graph-quality knobs.
//
// Config.ExtendCandidates and Config.Level0FullDegree both act at BUILD time:
// they change WHICH edges a node keeps, not how many a query traverses. The
// question they raise is therefore not "does recall go up" (turning any knob
// that densifies a graph raises recall at a fixed ef) but "does the recall/QPS
// curve MOVE" — a knob that buys recall by making every query proportionally
// more expensive has bought nothing.
//
// So this measures recall AND throughput per variant and compares each variant
// against the baseline curve interpolated to the SAME recall.
//
// Why it lives in a test rather than a cmd/: it needs a real corpus and several
// minutes, so it is env-gated and skipped by default. Recall is
// hardware-independent (a property of the graph and the query, not of CPU
// speed), so the recall columns are trustworthy anywhere; the QPS columns are
// only meaningful as RATIOS between variants measured back-to-back in one
// process, which is exactly how the Pareto comparison uses them.
//
// Run it with:
//
//	ROSTAM_RECALL_LEVERS=1 go test ./vector -run TestRecallLevers -v -timeout 30m
//
// Corpus: SIFT-1M in the standard fvecs layout, at the same path as
// TestSIFT1M's location; the test skips when it is absent.
//
// Findings when this was written (SIFT, 200k x 128d, k=10, m=16, efc=200),
// across two independent runs — the replication is the evidence, since the
// effect sizes are single-digit percentages on a shared machine:
//
//	                   run 1   run 2   every point same sign?
//	Level0FullDegree   +8.5%  +12.2%   yes, both runs
//	ExtendCandidates   +0.6%   -4.2%   no, sign flips
//	both               -0.2%   +4.5%   no
//
// So Level0FullDegree is a real Pareto gain (~1.6x build time, and level 0
// already allocates 2*M edge slots so it costs no extra memory), while
// ExtendCandidates is noise-to-harmful for ~2.5x build time — despite being
// the knob its own doc comment recommends for recall-critical collections.
// Stacking both is worse than Level0FullDegree alone.
//
// Those are SIFT numbers (128d, L2). Confirm on the target corpus before
// changing a default; graph-quality effects are geometry-dependent.

type leverVariant struct {
	name  string
	apply func(*Config)
}

// leverPoint is one (variant, ef) measurement.
type leverPoint struct {
	ef     int
	recall float64
	qps    float64
}

func TestRecallLevers(t *testing.T) {
	if os.Getenv("ROSTAM_RECALL_LEVERS") == "" {
		t.Skip("set ROSTAM_RECALL_LEVERS=1 to run the recall/QPS lever sweep")
	}
	if testing.Short() {
		t.Skip("skipping in -short: needs a real corpus and several minutes")
	}
	// Same dataset and layout as TestSIFT1M — see its docstring for how to
	// populate it.
	dir := filepath.Join(os.TempDir(), "rostam-sift1m", "sift")
	if _, err := os.Stat(filepath.Join(dir, "sift_base.fvecs")); err != nil {
		t.Skipf("SIFT corpus not found at %s — see TestSIFT1M docstring: %v", dir, err)
	}

	const (
		nBase = 200_000
		nQ    = 500
		k     = 10 // below the default 100 on purpose: ef is clamped to k
	)
	efs := []int{10, 20, 50, 100, 200}

	base, err := readFvecs(filepath.Join(dir, "sift_base.fvecs"))
	if err != nil {
		t.Fatalf("read base: %v", err)
	}
	queries, err := readFvecs(filepath.Join(dir, "sift_query.fvecs"))
	if err != nil {
		t.Fatalf("read queries: %v", err)
	}
	if len(base) > nBase {
		base = base[:nBase]
	}
	if len(queries) > nQ {
		queries = queries[:nQ]
	}
	if len(base) == 0 || len(queries) == 0 {
		t.Fatalf("empty corpus: %d base, %d queries", len(base), len(queries))
	}
	dim := len(base[0])
	t.Logf("SIFT: %d base x %dd, %d queries, k=%d", len(base), dim, len(queries), k)

	// Ground truth is computed here rather than read from sift_groundtruth.ivecs:
	// the shipped file is for the FULL 1M base, so it is simply wrong for any
	// subset, and silently so — it would report a plausible-looking low recall.
	truth := exactTopK(base, queries, k)

	ids := make([]uint64, len(base))
	for i := range ids {
		ids[i] = uint64(i)
	}

	variants := []leverVariant{
		{"baseline", func(c *Config) {}},
		{"extendCandidates", func(c *Config) { c.ExtendCandidates = true }},
		{"level0FullDegree", func(c *Config) { c.Level0FullDegree = true }},
		{"both", func(c *Config) { c.ExtendCandidates = true; c.Level0FullDegree = true }},
	}

	measured := make(map[string][]leverPoint, len(variants))
	for _, v := range variants {
		var pts []leverPoint
		var buildSec float64
		for i, ef := range efs {
			cfg := DefaultConfig()
			cfg.Dim = dim
			cfg.Metric = L2
			cfg.M = 16
			cfg.EfConstruction = 200
			cfg.EfSearch = ef
			cfg.Seed = 42
			v.apply(&cfg)

			// ef_search is collection-level, so each ef needs its own build.
			c, err := NewCollection(fmt.Sprintf("levers-%s-%d", v.name, ef), cfg)
			if err != nil {
				t.Fatalf("%s ef=%d: new collection: %v", v.name, ef, err)
			}
			tb := time.Now()
			if err := c.BuildConcurrent(ids, base, runtime.NumCPU()/2); err != nil {
				t.Fatalf("%s ef=%d: build: %v", v.name, ef, err)
			}
			if i == 0 {
				buildSec = time.Since(tb).Seconds()
			}
			r, qps := recallAndQPS(t, c, queries, truth, k)
			pts = append(pts, leverPoint{ef, r, qps})
			c.Stop()
		}
		measured[v.name] = pts
		t.Logf("%-17s build=%6.1fs  %s", v.name, buildSec, fmtPoints(pts))
	}

	// --- invariants worth failing on ---
	for name, pts := range measured {
		for i := 1; i < len(pts); i++ {
			// Recall must not fall as ef grows: a wider candidate list can only
			// find more. A drop means the graph or the search is broken, not
			// that a knob is ineffective.
			if pts[i].recall < pts[i-1].recall-0.005 {
				t.Errorf("%s: recall fell with larger ef: ef=%d %.4f -> ef=%d %.4f",
					name, pts[i-1].ef, pts[i-1].recall, pts[i].ef, pts[i].recall)
			}
		}
		if top := pts[len(pts)-1].recall; top < 0.95 {
			t.Errorf("%s: recall only %.4f at ef=%d; expected >=0.95 on SIFT-%d",
				name, top, pts[len(pts)-1].ef, len(base))
		}
	}

	// --- the Pareto comparison, reported not asserted ---
	//
	// Deliberately not a pass/fail gate: the effect sizes are single-digit
	// percentages, this runs on whatever machine invoked it, and turning that
	// into an assertion would produce a flaky test that says nothing. The
	// numbers are the output.
	baseline := measured["baseline"]
	for _, v := range variants {
		if v.name == "baseline" {
			continue
		}
		var sum float64
		var n int
		var detail string
		for _, p := range measured[v.name] {
			b, ok := interpQPS(baseline, p.recall)
			if !ok {
				continue
			}
			d := (p.qps/b - 1) * 100
			sum += d
			n++
			detail += fmt.Sprintf(" r=%.4f %+.1f%%", p.recall, d)
		}
		if n == 0 {
			t.Logf("%-17s no overlap with the baseline recall range", v.name)
			continue
		}
		t.Logf("%-17s QPS at matched recall: mean %+.1f%% (%d points)%s",
			v.name, sum/float64(n), n, detail)
	}
}

func fmtPoints(pts []leverPoint) string {
	s := ""
	for _, p := range pts {
		s += fmt.Sprintf(" ef=%d:%.4f/%.0fqps", p.ef, p.recall, p.qps)
	}
	return s
}

// interpQPS reads the baseline curve at a given recall. Returns false outside
// the measured range: extrapolating a curve that steepens toward recall 1 would
// invent the very number the comparison depends on.
func interpQPS(curve []leverPoint, recall float64) (float64, bool) {
	type rq struct{ r, q float64 }
	pts := make([]rq, 0, len(curve))
	for _, p := range curve {
		pts = append(pts, rq{p.recall, p.qps})
	}
	if len(pts) < 2 {
		return 0, false
	}
	if recall < pts[0].r || recall > pts[len(pts)-1].r {
		return 0, false
	}
	for i := 0; i+1 < len(pts); i++ {
		r0, r1 := pts[i].r, pts[i+1].r
		if r0 <= recall && recall <= r1 && r1 > r0 {
			q0, q1 := pts[i].q, pts[i+1].q
			return q0 + (recall-r0)*((q1-q0)/(r1-r0)), true
		}
	}
	return 0, false
}

func recallAndQPS(t *testing.T, c *Collection, queries [][]float32, truth [][]uint64, k int) (float64, float64) {
	t.Helper()
	warm := len(queries)
	if warm > 50 {
		warm = 50
	}
	for _, q := range queries[:warm] {
		_, _ = c.Search(q, k)
	}
	hits, denom := 0, 0
	start := time.Now()
	for qi, q := range queries {
		res, err := c.Search(q, k)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		want := make(map[uint64]bool, k)
		for _, id := range truth[qi] {
			want[id] = true
		}
		for _, r := range res {
			if want[r.ID] {
				hits++
			}
		}
		denom += len(want)
	}
	el := time.Since(start).Seconds()
	if denom == 0 || el <= 0 {
		return 0, 0
	}
	return float64(hits) / float64(denom), float64(len(queries)) / el
}

// exactTopK computes brute-force top-k per query with a bounded max-heap (a
// full sort per query would dominate the run) across all cores.
func exactTopK(base, queries [][]float32, k int) [][]uint64 {
	out := make([][]uint64, len(queries))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for qi := range queries {
		wg.Add(1)
		go func(qi int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			q := queries[qi]
			h := &boundedTopK{k: k}
			for i, v := range base {
				var s float32
				for d := range q {
					diff := q[d] - v[d]
					s += diff * diff
				}
				h.push(uint64(i), s)
			}
			out[qi] = h.ids()
		}(qi)
	}
	wg.Wait()
	return out
}

type topKItem struct {
	id uint64
	d  float32
}

// boundedTopK is a max-heap capped at k: the worst element sits at the root, so
// a candidate is rejected in O(1) and only sifted when it wins.
type boundedTopK struct {
	k int
	h []topKItem
}

func (t *boundedTopK) push(id uint64, d float32) {
	if len(t.h) < t.k {
		t.h = append(t.h, topKItem{id, d})
		t.up(len(t.h) - 1)
		return
	}
	if d >= t.h[0].d {
		return
	}
	t.h[0] = topKItem{id, d}
	t.down(0)
}

func (t *boundedTopK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if t.h[p].d >= t.h[i].d {
			break
		}
		t.h[p], t.h[i] = t.h[i], t.h[p]
		i = p
	}
}

func (t *boundedTopK) down(i int) {
	n := len(t.h)
	for {
		l, r, big := 2*i+1, 2*i+2, i
		if l < n && t.h[l].d > t.h[big].d {
			big = l
		}
		if r < n && t.h[r].d > t.h[big].d {
			big = r
		}
		if big == i {
			return
		}
		t.h[i], t.h[big] = t.h[big], t.h[i]
		i = big
	}
}

func (t *boundedTopK) ids() []uint64 {
	out := make([]uint64, len(t.h))
	for i, it := range t.h {
		out[i] = it.id
	}
	return out
}
