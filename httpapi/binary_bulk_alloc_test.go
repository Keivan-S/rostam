// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"bytes"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
)

// This file is the regression suite for a REAL defect: the binary bulk routes
// used to size their allocations from the count/dim integers in the request
// header, before reading a single body byte.
//
// The original repro: a 16-byte chunked body declaring count=22369620, dim=1
// allocated 2.58 GiB on /points/batch and 0.25 GiB on /points/bulk — with an
// authenticator that REJECTS the request, because authorization ran after the
// allocations. About a dozen concurrent stalled 16-byte POSTs would OOM a 32 GB
// node at ~200 B/s of attacker traffic, unauthenticated.
//
// The guard that was supposed to stop it compared the declared size against
// Content-Length, which is worthless twice over: chunked bodies report -1 so the
// check was skipped entirely, and it only caught UNDER-declaration, so
// "Content-Length: 1TiB" with 16 bytes sent sailed through. http.MaxBytesReader
// does not help either — it caps what is READ, never what is RESERVED.
//
// The fix is that no declared number sizes an allocation: sections grow strictly
// as bytes arrive. These tests assert BOTH halves — the request is rejected AND
// the process did not allocate a corpus on the way to rejecting it.

// hostileHeader builds a 16-byte RVB1 header with arbitrary declared numbers and
// no body behind it.
func hostileHeader(flags, count, dim uint32) []byte {
	var b bytes.Buffer
	b.WriteString(binaryBulkMagic)
	_ = binary.Write(&b, binary.BigEndian, flags)
	_ = binary.Write(&b, binary.BigEndian, count)
	_ = binary.Write(&b, binary.BigEndian, dim)
	return b.Bytes()
}

// allocatedBy returns the bytes allocated while serving one request. TotalAlloc
// is cumulative and never decreases, so it captures transient reservations that
// a GC would otherwise hide — exactly what an amplification attack produces.
func allocatedBy(h http.Handler, r *http.Request) (uint64, *httptest.ResponseRecorder) {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc, rec
}

// allocBudget is the ceiling a hostile 16-byte request must stay under.
//
// Measured behaviour is 3.8-71 KiB (the 64 KiB bufio buffer dominates), so 1 MiB
// is ~14x headroom over reality. It was 32 MiB, which is ~450x looser than
// measured and would have sat there silently while someone reintroduced a
// 20 MiB-per-16-byte amplifier. A budget that cannot fail is not a budget.
const allocBudget = 1 << 20

// stallRatioBudget bounds TestBinaryBatchStallAfterRowsIsNotAmplified's
// allocation as a multiple of bytes actually delivered.
//
// Measured on a stalled 100k-row, dim=1 body (1,200,016 bytes on the wire):
// normal builds land at ~3.6x, stable across runs. 4x leaves room for that
// plus the bufio buffer, while the defect's shape (rows expanded before the
// body completes) lands near 12x and fails.
const stallRatioBudget = 4

// stallRatioBudgetRace is the -race counterpart of stallRatioBudget.
//
// The race detector instruments every allocation, which roughly doubles
// runtime.MemStats.TotalAlloc for identical, legitimate work: the same body
// that measures ~3.6x under a normal build measures a stable ~7.05x under
// -race (no amplification occurred — go test -race ./httpapi confirms the
// request still holds the same object graph, just more expensively
// accounted). That inflation is not proportional to what a reintroduced
// amplifier would do (a 12x-normal defect measures ~23x under -race), so a
// separately stated, still-tight race budget catches a regression without
// the normal-mode budget above having to be loosened to accommodate -race.
const stallRatioBudgetRace = 8

func newHostileReq(path string, body []byte, chunked bool, contentLength int64) *http.Request {
	r := httptest.NewRequest("POST", path, bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/octet-stream")
	if chunked {
		// A chunked body declares no length: ContentLength == -1 is precisely the
		// case the old Content-Length guard skipped.
		r.ContentLength = -1
		r.TransferEncoding = []string{"chunked"}
	} else {
		r.ContentLength = contentLength
	}
	return r
}

// TestBinaryBulkHostileHeaderDoesNotAllocate is the reviewer's exact repro,
// across both routes, both framings of the lie (chunked vs over-declared
// Content-Length), and both authenticated and anonymous callers.
func TestBinaryBulkHostileHeaderDoesNotAllocate(t *testing.T) {
	// count/dim from the original repro: 22369620 points of dim 1 is 268 MiB of
	// declared rows behind a 16-byte body.
	const hostileCount, hostileDim = 22369620, 1

	for _, anon := range []bool{false, true} {
		name := "authorized"
		var opts Options
		if anon {
			// Deny everything: an unauthenticated attacker must not be able to
			// drive ANY allocation before being turned away.
			name = "anonymous"
			opts.Authenticator = func(authz.AuthRequest) bool { return false }
		}
		t.Run(name, func(t *testing.T) {
			h, cleanup := newTestAPIOpts(t, opts)
			defer cleanup()
			if !anon {
				createColl(t, h, "victim", 4)
			}

			for _, route := range []string{"bulk", "batch"} {
				for _, lie := range []string{"chunked", "overdeclared-content-length"} {
					t.Run(route+"/"+lie, func(t *testing.T) {
						body := hostileHeader(0, hostileCount, hostileDim)
						r := newHostileReq("/v1/collections/victim/points/"+route, body,
							lie == "chunked", 1<<40)

						alloc, rec := allocatedBy(h, r)

						if rec.Code == http.StatusOK {
							t.Fatalf("hostile body accepted: %d %s", rec.Code, rec.Body.String())
						}
						if anon && rec.Code != http.StatusUnauthorized {
							t.Fatalf("anonymous caller got %d (%s), want 401 — auth must precede body work",
								rec.Code, rec.Body.String())
						}
						if alloc > allocBudget {
							t.Fatalf("allocated %d bytes serving a 16-byte body declaring %d points "+
								"(budget %d) — a declared count is sizing a reservation again",
								alloc, hostileCount, allocBudget)
						}
						t.Logf("%s/%s: status=%d allocated=%dB", route, lie, rec.Code, alloc)
					})
				}
			}
		})
	}
}

// TestBinaryBulkPointCeiling covers the amplifier the byte cap cannot see: at
// dim=1 a point costs 12 wire bytes but ~112 bytes of decoded pointReq, so a
// body that IS fully delivered still needs a ceiling on points per request.
func TestBinaryBulkPointCeiling(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "ceil", 1)

	// One over the ceiling is refused on the header alone.
	body := hostileHeader(0, maxBinaryBulkPoints+1, 1)
	alloc, rec := allocatedBy(h, newHostileReq("/v1/collections/ceil/points/batch", body, true, -1))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d (%s), want 413", rec.Code, rec.Body.String())
	}
	if alloc > allocBudget {
		t.Fatalf("allocated %d bytes rejecting an over-ceiling count", alloc)
	}
}

// TestBinaryBulkDimMustMatchCollection covers the other amplifier: a huge
// declared dim sizes every read on this path. The dimension is authoritative only
// at the shard that owns the collection config, so the rejection comes from there
// (Collection.StageBulk → vector.ErrDimMismatch → 400) rather than from a
// per-request config lookup in this layer, which would cost a routed round-trip
// per request in cluster mode.
//
// The transport's job is to make sure the wait is bounded while that happens, and
// these cases assert both halves: refused, and refused cheaply.
func TestBinaryBulkDimMustMatchCollection(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "dimcheck", 4)

	// dim=67108856, count=1 declares exactly 256 MiB of rows and passes every byte
	// bound. Nothing behind the header ever arrives, so it is refused having
	// reserved nothing for the bytes that never came.
	body := hostileHeader(0, 1, 67108856)
	alloc, rec := allocatedBy(h, newHostileReq("/v1/collections/dimcheck/points/bulk", body, true, -1))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
	}
	if alloc > allocBudget {
		t.Fatalf("allocated %d bytes rejecting a 256 MiB declaration behind a 16-byte body", alloc)
	}

	// A FULLY DELIVERED body with the wrong dim: the rows all arrive, so nothing
	// in the transport can refuse it — the shard must, and the error has to reach
	// the client as a 400 rather than a 500.
	for _, route := range []string{"bulk", "batch"} {
		ids, vecs := testPoints(2, 8) // collection is dim 4
		rec := doBin(t, h, "/v1/collections/dimcheck/points/"+route, binBody(0, ids, vecs, nil), nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: dim 8 against a dim-4 collection gave %d (%s), want 400",
				route, rec.Code, rec.Body.String())
		}
		if !strings.Contains(rec.Body.String(), "Dim") && !strings.Contains(rec.Body.String(), "dim") {
			t.Fatalf("%s: error does not mention the dimension: %s", route, rec.Body.String())
		}
	}

	// And nothing from the rejected batches landed: the collection is still empty,
	// so a build succeeds and a search finds nothing.
	if rec := do(t, h, "POST", "/v1/collections/dimcheck/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build after rejected dim-mismatched stages: %d %s", rec.Code, rec.Body.String())
	}
}

// TestJSONBulkDimIsCheckedToo records the reach of moving the check to the shard:
// the JSON staging body had the SAME unchecked-dim gap, and putting the check
// where the authority is closed it for free — no transport had to opt in.
func TestJSONBulkDimIsCheckedToo(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "jsondim", 4)

	// dim 3 against a dim-4 collection, over the plain JSON body.
	rec := do(t, h, "POST", "/v1/collections/jsondim/points/bulk",
		`{"points":[{"id":1,"vector":[1,2,3]}]}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("JSON stage with wrong dim gave %d (%s), want 400", rec.Code, rec.Body.String())
	}
	// It must fail at STAGE time, not silently accept and blow up at build.
	if rec := do(t, h, "POST", "/v1/collections/jsondim/points/bulk/build", `{}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("build after a rejected JSON stage: %d %s", rec.Code, rec.Body.String())
	}
}

// TestBinaryBatchStallAfterRowsIsNotAmplified covers the slow-drip shape: a
// client delivers every row and then stops before the payload section it
// declared. The server must not have expanded those rows into per-point
// structures yet — that expansion is ~11.7x on this route, and with no
// ReadTimeout configured the client can hold it indefinitely (~62 connections
// killed a 32 GiB node on 186 MiB of traffic).
//
// The body here is truncated exactly at the payload boundary, which is what a
// stalled connection looks like once the read gives up: the request must be
// refused having held roughly what was sent, not a multiple of it.
func TestBinaryBatchStallAfterRowsIsNotAmplified(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	// dim=1 is the worst case and the one worth pinning: a row is 12 wire bytes
	// but decodes into ~112 bytes of pointReq plus 4 of float backing, a 9.7x
	// expansion. 100k points is ~1.2 MiB on the wire and ~11.6 MiB expanded.
	const n, dim = 100000, 1
	createColl(t, h, "stall", dim)
	ids, vecs := testPoints(n, dim)

	// Every row present, payload flag set, payload section entirely absent.
	body := binBody(binBulkFlagPayloads|binBulkFlagUpsert, ids, vecs, nil)

	alloc, rec := allocatedBy(h, newHostileReq("/v1/collections/stall/points/batch", body, true, -1))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
	}
	// TotalAlloc is CUMULATIVE, so arrival-driven buffer doubling shows up as ~2x
	// the body on its own even when nothing is expanded. See stallRatioBudget and
	// stallRatioBudgetRace for how the two ratios were measured.
	ratio := stallRatioBudget
	if raceEnabled {
		ratio = stallRatioBudgetRace
	}
	if budget := ratio * len(body); alloc > uint64(budget) {
		t.Fatalf("a client that sent %d bytes and stalled made the server allocate %d "+
			"(budget %d, %dx ratio): rows are being expanded before the body is complete",
			len(body), alloc, budget, ratio)
	}
	t.Logf("sent=%dB allocated=%dB (%.1fx)", len(body), alloc, float64(alloc)/float64(len(body)))
}

// TestBinaryBulkPartialDeliveryIsRatioBounded covers the growth rule itself.
//
// The reservation ceiling is max(binBulkReserveBytes, received × ratio), so a
// client that delivers a LITTLE cannot unlock a lot. This is the case between
// the two already covered — zero delivery (bounded by the flat floor) and full
// delivery (bounded by what was actually sent) — and it is the one the ratio
// exists for: send a few hundred KiB against a 200 MiB declaration and stop.
func TestBinaryBulkPartialDeliveryIsRatioBounded(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	const dim = 768
	createColl(t, h, "partial", dim)

	// Declare ~200 MiB of rows, deliver 256 KiB of them, then stop.
	const declared = 68000 // × 3080 B/row ≈ 200 MiB
	const delivered = 256 << 10
	body := append(hostileHeader(0, declared, dim), make([]byte, delivered)...)

	alloc, rec := allocatedBy(h, newHostileReq("/v1/collections/partial/points/bulk", body, true, -1))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
	}
	// 256 KiB delivered × 64 = 16 MiB ceiling; the declaration was 200 MiB. Allow
	// the ceiling plus buffer slack, and assert we are nowhere near the declared
	// size — that is the property under test.
	const budget = 24 << 20
	if alloc > budget {
		t.Fatalf("delivering %d bytes against a %d-byte declaration allocated %d "+
			"(budget %d): the reservation is tracking the declaration, not delivery",
			delivered, declared*(8+dim*4), alloc, budget)
	}
	t.Logf("declared=%dB delivered=%dB allocated=%dB", declared*(8+dim*4), delivered, alloc)
}

// TestBinaryBulkOversizeName pins an ordering hazard: the op-args encoders PANIC
// on a name too long for the wire's single length byte (rather than emit a header
// that would retarget the write at another collection), so the name must be
// checked BEFORE any args are built for it.
//
// Through the mux, nameLenGuard (247) rejects such a name first — so the
// route-level case below proves the request is refused, not that this transport's
// own guard ran. The direct call is what pins the transport's guard, which exists
// so this path does not depend on middleware in another package having run.
func TestBinaryBulkOversizeName(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	long := strings.Repeat("a", 300)
	for _, route := range []string{"bulk", "batch"} {
		rec := doBin(t, h, "/v1/collections/"+long+"/points/"+route,
			hostileHeader(0, 1, 4), nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: status = %d (%s), want 400", route, rec.Code, rec.Body.String())
		}
	}

	// Straight at the handler, with no mux and therefore no nameLenGuard: a name
	// one byte past what the wire can encode must be a 400, not a panic.
	over := strings.Repeat("a", ops.MaxCollectionNameWire+1)
	a := &api{disp: nil} // never reached: the name check precedes every dispatch
	for _, call := range []struct {
		name string
		fn   func(http.ResponseWriter, *http.Request)
	}{
		{"bulk", a.putPointsBulkBinary},
		{"batch", func(w http.ResponseWriter, r *http.Request) {
			a.decodeBinPointsBatch(w, r, r.PathValue("name"), &pointsBatchReq{})
		}},
	} {
		t.Run("direct/"+call.name, func(t *testing.T) {
			r := newHostileReq("/v1/collections/x/points/"+call.name, hostileHeader(0, 1, 4), true, -1)
			r.SetPathValue("name", over)
			rec := httptest.NewRecorder()
			call.fn(rec, r) // panics on regression
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestBinaryBulkTruncatedBodyIsBounded checks the honest-but-short case: a
// header promising a large body that then stops early must be rejected having
// held only what actually arrived.
func TestBinaryBulkTruncatedBodyIsBounded(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	createColl(t, h, "short", 768)

	// Declare 200k points of dim 768 (~600 MiB) and send one row.
	body := append(hostileHeader(0, 200000, 768), make([]byte, 8+768*4)...)
	alloc, rec := allocatedBy(h, newHostileReq("/v1/collections/short/points/bulk", body, true, -1))
	if rec.Code != http.StatusRequestEntityTooLarge && rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400 or 413", rec.Code, rec.Body.String())
	}
	if alloc > allocBudget {
		t.Fatalf("allocated %d bytes for a body that delivered %d", alloc, len(body))
	}
}
