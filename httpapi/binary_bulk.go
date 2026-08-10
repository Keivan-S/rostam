// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime"
	"net/http"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Binary bulk-ingest framing ("RVB1") for /points/bulk and /points/batch.
//
// WHY: the JSON body is the ingest bottleneck, not the index. Building HNSW over
// 1M × 768d takes ~164s, but shipping those vectors as JSON text runs at ~1.2k
// vec/s — the number decodes 768 base-10 float literals per point and allocates a
// []float32 per point on top. The vectors are already f32 on both ends; the text
// round-trip is pure overhead.
//
// This framing is a dense re-encoding of the SAME request body, selected by
// Content-Type. It adds NO new semantics: each route decodes a binary body into
// exactly the request the JSON body would have produced and then runs the
// identical code. A request with a JSON content type takes the byte-identical
// pre-existing path.
//
//	magic  [4]byte  "RVB1"
//	flags  u32      bit0 payloads present, bit1 upsert
//	count  u32      number of points
//	dim    u32      vector dimension (every vector in the body has it)
//	rows   count × [ id u64 ][ dim × f32 ]
//	pays   count × [ len u32 ][ len bytes of JSON object ]   (only when bit0)
//
// EVERYTHING IS BIG-ENDIAN, deliberately. Rostam's op wire and native TCP wire
// are big-endian, and a staged row here is byte-identical to a vector_bulk_stage
// row — so /points/bulk streams the row region straight off the socket in behind
// the op header (ops.BulkStageArgsHeader) with zero per-float work. A
// little-endian framing would have cost a full byte-swap plus a [][]float32
// materialization of the entire corpus for no gain on either end (clients
// byte-swap an array in one call).
//
// The declared dim is NOT checked against the collection here; the shard that
// owns the config rejects a mismatch (see binBulkHeader.validate).
//
// A per-point payload is a JSON object in Rostam's tagged metadata form
// ({"id":{"kind":"int","int":7}}) — the filter-case path, where a scalar has to
// travel with each vector. len=0 means "no payload for this point".
//
// BOTH routes carry payloads. /points/batch indexes each point inline;
// /points/bulk stages them for the multi-core build via
// vector_bulk_stage_payload, whose payload section is byte-for-byte this one, so
// the region streams straight through behind the op header exactly as the rows
// do. (/points/bulk used to REJECT a payload-bearing body, because the staging op
// had nowhere to put one — which is precisely what forced every filtered workload
// onto the ~6x slower inline route.)
const (
	binaryBulkMagic     = "RVB1"
	binaryBulkHeaderLen = 16 // magic(4) + flags(4) + count(4) + dim(4)

	binBulkFlagPayloads uint32 = 1 << 0
	binBulkFlagUpsert   uint32 = 1 << 1
	binBulkKnownFlags          = binBulkFlagPayloads | binBulkFlagUpsert
)

// Body bounds.
//
// maxBinaryBulkBody is the same memory ceiling as maxBulkJSONBody, so the binary
// route can never buffer more than the JSON route it replaces.
//
// maxBinaryBulkPoints caps the DECLARED point count independently of the byte
// cap, because points are not bounded by bytes alone: a point costs ~112 bytes
// of pointReq on the batch route but only 12 wire bytes at dim=1, a ~9x
// amplifier that the byte cap does not see. At dim=768 the byte cap binds first
// (~85k points), so this ceiling only ever bites pathological low-dim requests.
//
// binBulkReserveBytes is the MOST that may be reserved before a single body byte
// has arrived. Everything past it is grown only as data actually lands, so a
// client that declares a huge count and then stalls holds this much and no more,
// however large the integer in its header.
//
// It is deliberately SMALL — the same order as the per-request bufio buffer, so
// it adds nothing an attacker did not already get for free by opening the
// request. It was 8 MiB, sized to swallow a whole reference chunk without
// growing, which made a 16-byte header worth 8 MiB of unpaid reservation: the
// exact amplifier this file exists to prevent, reintroduced as an optimization.
//
// binBulkGrowthRatio is how far past what has ALREADY ARRIVED the reservation
// may run. Growth is justified by delivery and by nothing else, so the ceiling
// is max(binBulkReserveBytes, received × ratio): a client that has sent 16 bytes
// can never hold more than the floor, while one genuinely streaming a 3 MiB
// chunk reaches full size in three passes.
//
// Plain doubling was tried instead and cost 41% of wire throughput on a 1M×768d
// load (151k → 89k vec/s) — a dozen reallocations and copies per request, and
// the GC pressure of a dozen multi-MiB buffers where one would do. Tying the
// step to arrival keeps the bound and gets the throughput back.
//
// maxBinaryPayloadLen bounds ONE point's payload blob.
const (
	maxBinaryBulkBody   = 256 << 20 // 256 MiB
	maxBinaryBulkPoints = 1 << 18   // 262,144 points per request
	binBulkReserveBytes = 64 << 10  // 64 KiB reserved before any byte arrives
	binBulkGrowthRatio  = 64        // then at most 64x what has arrived
	maxBinaryPayloadLen = 1 << 20   // 1 MiB per point
)

// isBinaryBulk reports whether the request selects the binary framing. Selection
// is by media type alone (parameters like charset are ignored); anything else —
// including an absent Content-Type — takes the JSON path unchanged.
func isBinaryBulk(r *http.Request) bool {
	ct := r.Header.Get("Content-Type")
	if ct == "" {
		return false
	}
	mt, _, err := mime.ParseMediaType(ct)
	if err != nil {
		return false
	}
	return mt == "application/octet-stream"
}

// binBulkHeader is a validated binary bulk header. rowBytes is the exact size of
// the row region, already proven to fit within maxBinaryBulkBody.
type binBulkHeader struct {
	flags    uint32
	count    int
	dim      int
	rowBytes int
}

func (h binBulkHeader) payloads() bool { return h.flags&binBulkFlagPayloads != 0 }
func (h binBulkHeader) upsert() bool   { return h.flags&binBulkFlagUpsert != 0 }

// readBinBulkPrefix caps the body and reads the 16-byte FIXED header — magic,
// flags, and the two declared numbers — performing only the checks that need
// nothing but the header itself.
//
// It runs before authorization, which is safe precisely because 16 bytes is a
// constant: reading it costs less than the request line already did, and nothing
// here is sized by a number the client wrote. Every LIMIT on the declared
// numbers lives in binBulkHeader.validate, which runs after the caller has
// authorized, so an anonymous client is turned away rather than taught the
// server's limits. Splitting it this way is also what lets the batch route pick
// insert-vs-upsert scope from the header's flag WITHOUT first consuming a
// client-sized body.
//
// Content-Length is deliberately never consulted anywhere on this path: it is
// absent (-1) on a chunked body and freely over-declarable on any body, so a
// check against it would be skipped exactly when it mattered and satisfied by a
// liar the rest of the time. Delivery is the only evidence that counts.
func readBinBulkPrefix(w http.ResponseWriter, r *http.Request) (binBulkHeader, *bufio.Reader, bool) {
	// Cap at the read layer first: everything below reads through this, so a
	// lying Content-Length (or none at all, e.g. chunked) still cannot stream
	// more than the cap into the heap.
	r.Body = http.MaxBytesReader(w, r.Body, maxBinaryBulkBody)
	// Small buffer on purpose: bufio.Read hands a read whose destination is at
	// least the buffer size straight to the underlying reader, so a modest buffer
	// serves the 4-byte length reads while the multi-MiB row read still lands in
	// the caller's buffer with no intermediate copy.
	br := bufio.NewReaderSize(r.Body, 64<<10)

	var hdr [binaryBulkHeaderLen]byte
	if !readFullBin(w, br, hdr[:], "header") {
		return binBulkHeader{}, nil, false
	}
	if string(hdr[0:4]) != binaryBulkMagic {
		writeError(w, http.StatusBadRequest, "invalid binary bulk body: bad magic (expected "+binaryBulkMagic+")")
		return binBulkHeader{}, nil, false
	}
	flags := binary.BigEndian.Uint32(hdr[4:8])
	if flags&^binBulkKnownFlags != 0 {
		// Fail loud on an unknown flag rather than ignoring it: a future framing
		// bit means the body is shaped differently than we are about to read it.
		writeError(w, http.StatusBadRequest, "invalid binary bulk body: unknown flags")
		return binBulkHeader{}, nil, false
	}
	// The declared numbers are carried out UNVALIDATED on purpose: every limit
	// check on them lives in validate(), which runs after the caller has
	// authorized, so an anonymous client is told 401 rather than being taught the
	// server's limits. Only the two checks above — which decide whether this is
	// even an RVB1 frame, and thus whether the upsert flag can be trusted to pick
	// the authorization scope — happen here.
	return binBulkHeader{
		flags: flags,
		count: int(binary.BigEndian.Uint32(hdr[8:12])),
		dim:   int(binary.BigEndian.Uint32(hdr[12:16])),
	}, br, true
}

// validate applies every limit this transport can enforce on its own, once the
// caller has authorized, and computes rowBytes. It calls ops.CountFitsIn rather
// than restating its arithmetic, so the bound on this transport cannot drift
// from the bound on the op wire — a drifted copy of a bound fails as an OOM, not
// as a test.
//
// It deliberately does NOT check dim against the collection's configured
// dimension. This layer has no local knowledge of it, and fetching it would cost
// a routed round-trip PER REQUEST in cluster mode — ~1000 extra routed calls on
// a 1M load at the reference chunk size. The shard that owns the config rejects
// a mismatch instead (Collection.StageBulk → ErrDimMismatch → 400), which is
// both free and broader: it covers the JSON body and the native TCP wire too,
// neither of which this function ever sees.
//
// That leaves a well-formed body whose dim is wrong readable up to the byte cap
// before the shard refuses it. rowBytes here is an EXPECTATION, never a
// reservation: readSection grows strictly with what arrives.
//
// "The sender pays every byte" is how an earlier version of this comment
// dismissed that residual, and it understated it. What the sender pays for is
// WIRE bytes; what the server then holds is decoded structures, and on
// /points/batch those are ~11.7x larger (3 MiB of rows became 35 MiB of pointReq
// and float32 backing). A client could deliver every row, let the server absorb
// that expansion, and then stall before the payload section — holding the
// expanded form indefinitely, since no ReadTimeout or IdleTimeout bounds a
// request here. ~62 such connections killed a 32 GiB node on 186 MiB of traffic.
//
// decodeBinPointsBatch now consumes the ENTIRE body before deriving anything
// from it, so a stalled client holds only what it sent and the ratio is ~1x.
// The expansion still happens after the body is complete, bounded by
// maxBinaryBulkPoints — that part is the same cost any legitimate request pays.
func (h binBulkHeader) validate(w http.ResponseWriter) (binBulkHeader, bool) {
	// Reject a negative widening first. int is 64-bit on every platform this ships
	// on, so a u32 always widens cleanly — but on a 32-bit build a count near 2^32
	// lands NEGATIVE and sails past every `>` bound below.
	if h.count < 0 || h.dim < 0 {
		writeError(w, http.StatusRequestEntityTooLarge, "binary bulk body too large")
		return binBulkHeader{}, false
	}
	// The point ceiling is checked before anything is sized: bytes alone do not
	// bound what a point costs in memory once decoded (~112 B of pointReq per 12
	// wire bytes at dim=1), so the byte cap cannot stand in for it.
	if h.count > maxBinaryBulkPoints {
		writeError(w, http.StatusRequestEntityTooLarge,
			"binary bulk body declares too many points per request; split it into smaller requests")
		return binBulkHeader{}, false
	}
	if h.count == 0 {
		return h, true
	}
	if h.dim == 0 {
		writeError(w, http.StatusBadRequest, "invalid binary bulk body: dim must be > 0")
		return binBulkHeader{}, false
	}
	budget := maxBinaryBulkBody - binaryBulkHeaderLen
	// dim ≤ (budget-8)/4 keeps perRow itself from overflowing before CountFitsIn
	// can divide by it.
	if h.dim > (budget-8)/4 {
		writeError(w, http.StatusRequestEntityTooLarge, "binary bulk body too large")
		return binBulkHeader{}, false
	}
	perRow := ops.BulkStageRowLen(h.dim)
	if !ops.CountFitsIn(h.count, budget, perRow) {
		writeError(w, http.StatusRequestEntityTooLarge, "binary bulk body too large")
		return binBulkHeader{}, false
	}
	h.rowBytes = h.count * perRow
	return h, true
}

// readSection reads exactly n bytes of the body, appended to dst, reserving only
// what the client has actually delivered.
//
// This is the load-bearing anti-amplification primitive. n comes from a header
// the client wrote, so it is a claim, not a size: `make([]byte, n)` would let a
// 16-byte request reserve gigabytes and stall, and MaxBytesReader would not stop
// it because MaxBytesReader caps what is READ, never what is RESERVED.
// bytes.Buffer.ReadFrom grows strictly as bytes land, so a stalling client can
// only make us hold what it has paid for; the up-front Grow is clamped to
// binBulkReserveBytes so even the first reservation is bounded.
func readSection(w http.ResponseWriter, br *bufio.Reader, dst *bytes.Buffer, n int, what string) bool {
	base := dst.Len()
	for {
		got := dst.Len() - base
		if got >= n {
			break
		}
		// How much may be held right now: a flat floor before anything has arrived,
		// then a multiple of what the client has ALREADY delivered. Delivery is the
		// only thing that justifies a reservation, so the ceiling rises with it and
		// with nothing else. A 16-byte body never gets past the floor no matter what
		// its header claims; a real transfer reaches full size in three passes
		// (64 KiB → 4 MiB → done) instead of a dozen doublings.
		allow := binBulkReserveBytes
		if justified := got * binBulkGrowthRatio; justified > allow {
			allow = justified
		}
		if allow > n {
			allow = n
		}
		read, err := io.CopyN(dst, br, int64(allow-got))
		if err != nil {
			var mbe *http.MaxBytesError
			if errors.As(err, &mbe) {
				writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
				return false
			}
			// io.EOF here means the body ended before the header's declared length.
			// Nothing was ever reserved for the bytes that never came.
			writeError(w, http.StatusBadRequest, "invalid binary bulk body: truncated "+what)
			return false
		}
		if read == 0 {
			// No progress and no error: refuse rather than spin.
			writeError(w, http.StatusBadRequest, "invalid binary bulk body: truncated "+what)
			return false
		}
	}
	if dst.Len()-base != n {
		writeError(w, http.StatusBadRequest, "invalid binary bulk body: truncated "+what)
		return false
	}
	return true
}

// readPayloadSection reads the count length-prefixed payload blobs that follow
// the row region, appending them to dst VERBATIM — length prefix included — so
// that what lands in dst is byte-identical to the op wire's payload section and
// can be dispatched without re-encoding.
//
// wantSpans asks for one [offset, len] span per point locating that point's JSON
// bytes INSIDE dst. Only the inline batch route needs them (it unmarshals each
// blob itself); the staging route dispatches the bytes untouched and wants
// nothing back, and building the slice for it anyway would allocate ~4 MB of
// immediate garbage per request at the maxBinaryBulkPoints cap — on the one path
// whose entire purpose is not to touch payloads.
//
// Nothing here is sized by a declared number: each blob goes through readSection,
// which grows strictly with what the client has delivered, and each declared
// length is bounded by maxBinaryPayloadLen BEFORE it is read. The spans slice
// grows one entry at a time as blobs actually arrive, so it too is paid for.
func readPayloadSection(w http.ResponseWriter, br *bufio.Reader, dst *bytes.Buffer, count int, wantSpans bool) ([][2]int, bool) {
	var spans [][2]int
	var lenBuf [4]byte
	for i := 0; i < count; i++ {
		if !readFullBin(w, br, lenBuf[:], "payload length") {
			return nil, false
		}
		// int(uint32) only widens negatively on a 32-bit build, where a length near
		// 2^32 would land negative and sail past the `>` bound below.
		n := int(binary.BigEndian.Uint32(lenBuf[:]))
		if n < 0 || n > maxBinaryPayloadLen {
			writeError(w, http.StatusRequestEntityTooLarge, "binary bulk payload too large")
			return nil, false
		}
		dst.Write(lenBuf[:])
		start := dst.Len()
		if n > 0 && !readSection(w, br, dst, n, "payload") {
			return nil, false
		}
		if wantSpans {
			spans = append(spans, [2]int{start, n})
		}
	}
	return spans, true
}

// expectEOF rejects a body with bytes left over after its declared contents.
// Trailing data means the sender framed the request differently than we just
// read it, so accepting the prefix would ingest points nobody asked for; the
// JSON path is equally strict (DisallowUnknownFields).
func expectEOF(w http.ResponseWriter, br *bufio.Reader) bool {
	if _, err := br.ReadByte(); err != io.EOF {
		writeError(w, http.StatusBadRequest, "invalid binary bulk body: trailing bytes after the last point")
		return false
	}
	return true
}

// readFullBin fills a SMALL fixed-size buf (a header, a length prefix) from br.
// Never use it for a length the client declared — see readSection.
func readFullBin(w http.ResponseWriter, br *bufio.Reader, buf []byte, what string) bool {
	if _, err := io.ReadFull(br, buf); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return false
		}
		writeError(w, http.StatusBadRequest, "invalid binary bulk body: truncated "+what)
		return false
	}
	return true
}

// validWireName rejects a collection name the op wire cannot encode. The wire
// stores the name's length in ONE byte, so a >=256-byte name would wrap modulo
// 256 and retarget the write at a different collection.
//
// It must be called BEFORE building any op args for that name: the encoders
// panic on an over-long name rather than emit a corrupt header, so checking
// afterwards would be checking after the crash. nameLenGuard enforces the same
// bound as middleware, but a transport that builds op args cannot depend on a
// guard in another package having run.
func validWireName(w http.ResponseWriter, name string) bool {
	if len(name) > ops.MaxCollectionNameWire {
		writeError(w, http.StatusBadRequest, "collection name too long")
		return false
	}
	return true
}

// putPointsBulkBinary stages a binary body through the SAME op as the JSON
// staging route. The framing's row region is byte-identical to the op's row
// region — and, when the body carries payloads, so is its payload section — so
// both are streamed in directly behind the op header: no decode, no [][]float32,
// no per-point JSON round-trip, no intermediate materialization of the corpus.
//
// The payload bit selects vector_bulk_stage_payload; without it the dispatch is
// byte-identical to what it always was.
func (a *api) putPointsBulkBinary(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !validWireName(w, name) {
		return
	}
	// The payload bit lives in the FIXED 16-byte header, so the op this request
	// dispatches — and therefore the scope to authorize against — is known without
	// consuming a single client-SIZED byte. Reading that constant prefix ahead of
	// authorization is the same trade /points/batch makes for its upsert bit: 16
	// bytes cost less than the request line already did, while every limit on the
	// declared numbers stays behind validate(), which runs after the check, so an
	// anonymous client is turned away rather than taught the server's limits.
	h, br, ok := readBinBulkPrefix(w, r)
	if !ok {
		return
	}
	op := "vector_bulk_stage"
	if h.payloads() {
		op = "vector_bulk_stage_payload"
	}
	if !a.authorize(w, r, op, bulkStageAuthArgs(name)) {
		return
	}
	if h.upsert() {
		writeError(w, http.StatusBadRequest,
			"upsert is not supported on the bulk staging route; use /points/batch")
		return
	}
	if h, ok = h.validate(w); !ok {
		return
	}
	// The op header is a few dozen bytes and is sized by nothing the client
	// declared; the rows — and then the payload blobs — land behind it as they
	// arrive. BulkStageArgsHeader is shared by both ops for exactly this reason:
	// the payload-bearing wire is the vectors-only wire plus a suffix.
	var buf bytes.Buffer
	buf.Write(ops.BulkStageArgsHeader(name, h.dim, h.count))
	if !readSection(w, br, &buf, h.rowBytes, "rows") {
		return
	}
	if h.payloads() {
		if _, ok := readPayloadSection(w, br, &buf, h.count, false); !ok {
			return
		}
	}
	if !expectEOF(w, br) {
		return
	}
	// Already authorized above, so dispatch directly rather than re-running auth.
	if _, err := a.disp.Call(op, buf.Bytes()); err != nil {
		writeDispatchError(w, op, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"staged": h.count})
}

// decodeBinPointsBatch decodes a binary body into the SAME pointsBatchReq the
// JSON body decodes into, so /points/batch runs one shared apply path for both
// encodings (see putPointsBatch). Vectors come out of one flat backing array
// rather than a slice per point.
//
// It authorizes against the op the header's upsert flag selects, BEFORE reading
// any client-sized section, and nothing here is sized by a declared number until
// the bytes behind it have actually arrived.
func (a *api) decodeBinPointsBatch(w http.ResponseWriter, r *http.Request, name string, req *pointsBatchReq) bool {
	if !validWireName(w, name) {
		return false
	}
	h, br, ok := readBinBulkPrefix(w, r)
	if !ok {
		return false
	}
	req.Upsert = h.upsert()

	// The upsert bit lives in the FIXED header, so the correct scope is known
	// without consuming anything the client sized. Build the representative args
	// with the same encoders putPointsBatch uses so the wire layout the authorizer
	// inspects is identical.
	opName := "vector_insert"
	authArgs := ops.EncodeVectorInsertArgsExt(name, 0, nil, 0, nil, vector.SparseVector{})
	if req.Upsert {
		opName = "vector_upsert"
		authArgs = ops.EncodeVectorUpsertArgs(name, 0, nil, "", 0, nil, vector.SparseVector{})
	}
	if !a.authorize(w, r, opName, authArgs) {
		return false
	}
	if h, ok = h.validate(w); !ok {
		return false
	}

	// THE WHOLE BODY IS CONSUMED FIRST, and only then is anything derived from it.
	//
	// The obvious structure — decode the rows into pointReqs, then read the payload
	// section — is a slow-drip amplifier: a client sends every row, watches the
	// server turn 3 MiB of wire into 35 MiB of pointReq and float32 backing (11.7x),
	// and then simply stops before the payload section. The server holds all of it
	// for as long as the connection lives, and no ReadTimeout or IdleTimeout is
	// configured to cut it off, so ~62 such connections exhaust a 32 GiB node on
	// 186 MiB of traffic. Consuming the body first means a stalled client holds
	// only the bytes it actually sent.
	var rowBuf bytes.Buffer
	if !readSection(w, br, &rowBuf, h.rowBytes, "rows") {
		return false
	}
	// Payload blobs land in one buffer with a span per point, both of which grow
	// only as bytes arrive — no structure here is sized by a declared number.
	var payBuf bytes.Buffer
	var paySpans [][2]int
	if h.payloads() {
		var ok bool
		if paySpans, ok = readPayloadSection(w, br, &payBuf, h.count, true); !ok {
			return false
		}
	}
	if !expectEOF(w, br) {
		return false
	}

	// The body is fully delivered and the connection can no longer stall us. Only
	// now do the per-point structures get built: h.count is no longer a claim —
	// h.rowBytes bytes were really sent — and maxBinaryBulkPoints bounds the ~9x
	// pointReq-per-wire-byte amplification that remains at low dim.
	rows := rowBuf.Bytes()
	req.Points = make([]pointReq, h.count)
	// One backing array for every vector: 1M points cost one allocation here
	// instead of 1M.
	flat := make([]float32, h.count*h.dim)
	off := 0
	for i := range req.Points {
		req.Points[i].ID = binary.BigEndian.Uint64(rows[off:])
		off += 8
		v := flat[i*h.dim : (i+1)*h.dim : (i+1)*h.dim]
		for d := range v {
			v[d] = math.Float32frombits(binary.BigEndian.Uint32(rows[off:]))
			off += 4
		}
		req.Points[i].Vector = v
	}
	pays := payBuf.Bytes()
	for i, span := range paySpans {
		if span[1] == 0 {
			continue // no payload for this point
		}
		var meta vector.Metadata
		if err := json.Unmarshal(pays[span[0]:span[0]+span[1]], &meta); err != nil {
			writeError(w, http.StatusBadRequest, "invalid binary bulk payload JSON: "+err.Error())
			return false
		}
		req.Points[i].Metadata = meta
	}
	return true
}
