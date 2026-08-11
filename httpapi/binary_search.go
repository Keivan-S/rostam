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
	"net/http"
)

// Binary query framing ("RVQ1") for the search routes.
//
// WHY: the same reason /points/bulk has one, on the other side of the workload.
// A query vector travels as base-10 float text, and encoding it dominates the
// request. Against a local server at dim=768, k=10, on a kept-alive connection:
// 0.845 ms per search, of which building the JSON body is 0.258 ms — 31% of the
// request spent turning float32s the client already holds into decimal, for the
// server to parse straight back. The same vector encodes in 0.011 ms as bytes.
// The server-side decode of those literals disappears along with it.
//
// This is a dense re-encoding of the SAME request, selected by Content-Type. It
// adds NO new semantics: it decodes into exactly the searchReq the JSON body
// would have produced, and then runs the identical validation and dispatch. A
// request with a JSON content type takes the byte-identical pre-existing path.
//
//	magic     [4]byte  "RVQ1"
//	flags     u32      bit0 filter present
//	k         u32
//	dim       u32      length of the query vector
//	rc        u8       read_consistency
//	opa       u8       on_partition_unavailable
//	_         u16      reserved, must be zero
//	staleness u64      max_staleness (the bound for rc=3)
//	query     dim × f32
//	filter    [len u32][len bytes of filter JSON]   (only when bit0)
//
// BIG-ENDIAN throughout, matching RVB1 and the op wire, so a client that
// already byte-swaps an array to send a bulk load reuses that code here.
//
// The declared dim is NOT checked against the collection: this layer has no
// local knowledge of it, and the shard that owns the config rejects a mismatch
// for every transport at once. Same reasoning as the bulk header.
const (
	binarySearchMagic     = "RVQ1"
	binarySearchHeaderLen = 28 // magic(4) flags(4) k(4) dim(4) rc(1) opa(1) pad(2) staleness(8)

	binSearchFlagFilter uint32 = 1 << 0
	binSearchKnownFlags        = binSearchFlagFilter
)

// Body bounds.
//
// A search request carries ONE vector, so every bound here is far tighter than
// the JSON route it mirrors (maxJSONBody, 32 MiB). Tighter is the point: the
// binary framing exists to make a small request cheap, so a body anywhere near
// the JSON ceiling is not a query this path should be serving.
//
// maxBinarySearchDim bounds the DECLARED dimension, and is checked BEFORE it is
// multiplied by four. That order is the whole guard: on a 32-bit build a dim of
// 0x40000000 multiplies to exactly zero, which would allocate nothing, read
// nothing, and hand the engine a silently empty query rather than an error.
// Capped first, dim×4 cannot overflow on any supported word size, and the
// largest buffer a client can make the server reserve before sending a byte is
// 256 KiB.
const (
	maxBinarySearchBody   = 8 << 20 // 8 MiB, whole body
	maxBinarySearchDim    = 1 << 16 // 65,536 floats → 256 KiB
	maxBinarySearchFilter = 1 << 20 // 1 MiB of filter JSON
)

// isBinarySearch reports whether the request selects the binary query framing.
// Selection is the same media-type test the bulk routes use: one binary content
// type across the whole API, so a client sets the same header everywhere.
func isBinarySearch(r *http.Request) bool { return isBinaryBulk(r) }

// readFullQuery mirrors readFullBin rather than sharing it, only so the error
// names this framing instead of the bulk one — a client sent an RVQ1 body and
// should not be told its bulk body is truncated.
func readFullQuery(w http.ResponseWriter, br *bufio.Reader, buf []byte, what string) bool {
	if _, err := io.ReadFull(br, buf); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return false
		}
		writeError(w, http.StatusBadRequest, "invalid binary query body: truncated "+what)
		return false
	}
	return true
}

// decodeSearchBody decodes a search request in whichever encoding it arrived in.
// Every search route goes through here so the two encodings cannot drift apart:
// there is one place that produces a searchReq, and one set of validations
// downstream of it.
func decodeSearchBody(w http.ResponseWriter, r *http.Request, req *searchReq) bool {
	if isBinarySearch(r) {
		return decodeBinarySearch(w, r, req)
	}
	return decodeBody(w, r, req)
}

// decodeBinarySearch parses an RVQ1 body into req.
//
// It rejects a body with trailing bytes. A frame whose declared contents end
// before the body does means the sender and this decoder disagree about the
// layout, and continuing on the half they agree about is how a request comes to
// mean something neither intended.
func decodeBinarySearch(w http.ResponseWriter, r *http.Request, req *searchReq) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxBinarySearchBody)
	br := bufio.NewReaderSize(r.Body, 8<<10)

	var hdr [binarySearchHeaderLen]byte
	if !readFullQuery(w, br, hdr[:], "header") {
		return false
	}
	if string(hdr[0:4]) != binarySearchMagic {
		writeError(w, http.StatusBadRequest, "invalid binary query body: bad magic (expected "+binarySearchMagic+")")
		return false
	}
	flags := binary.BigEndian.Uint32(hdr[4:8])
	if flags&^binSearchKnownFlags != 0 {
		// Fail loud rather than ignore: a future framing bit means the bytes
		// after this header are shaped differently than we are about to read them.
		writeError(w, http.StatusBadRequest, "invalid binary query body: unknown flags")
		return false
	}
	if binary.BigEndian.Uint16(hdr[18:20]) != 0 {
		writeError(w, http.StatusBadRequest, "invalid binary query body: reserved bytes must be zero")
		return false
	}

	k := binary.BigEndian.Uint32(hdr[8:12])
	dim := binary.BigEndian.Uint32(hdr[12:16])
	if dim > maxBinarySearchDim {
		writeError(w, http.StatusBadRequest, "invalid binary query body: dim exceeds 65536")
		return false
	}
	// k is bounded here only against the int conversion: a k above MaxInt32 would
	// land negative on a 32-bit build and reach validTopK as "<= 0", which is the
	// right refusal for the wrong reason. The real ceiling stays in validTopK, so
	// both encodings are held to one limit.
	if k > math.MaxInt32 {
		writeError(w, http.StatusBadRequest, "k must be between 1 and 65536")
		return false
	}

	query := make([]float32, dim)
	if dim > 0 {
		raw := make([]byte, int(dim)*4) // dim is capped above, so this cannot overflow
		if !readFullQuery(w, br, raw, "query vector") {
			return false
		}
		for i := range query {
			f := math.Float32frombits(binary.BigEndian.Uint32(raw[i*4:]))
			// JSON cannot express NaN or ±Inf, so accepting them here would make
			// the binary encoding mean something its JSON twin cannot say — and a
			// NaN coordinate silently poisons every distance it touches, returning
			// a wrong ranking rather than an error.
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				writeError(w, http.StatusBadRequest, "invalid binary query body: query contains NaN or Inf")
				return false
			}
			query[i] = f
		}
	}

	if flags&binSearchFlagFilter != 0 {
		var lenBuf [4]byte
		if !readFullQuery(w, br, lenBuf[:], "filter length") {
			return false
		}
		n := binary.BigEndian.Uint32(lenBuf[:])
		if n > maxBinarySearchFilter {
			writeError(w, http.StatusBadRequest, "invalid binary query body: filter too large")
			return false
		}
		buf := make([]byte, n)
		if !readFullQuery(w, br, buf, "filter") {
			return false
		}
		dec := json.NewDecoder(bytes.NewReader(buf))
		dec.DisallowUnknownFields() // the JSON route's rule, applied to the same bytes
		if err := dec.Decode(&req.Filter); err != nil {
			writeError(w, http.StatusBadRequest, "invalid binary query body: bad filter JSON: "+err.Error())
			return false
		}
	}

	if _, err := br.ReadByte(); err == nil {
		writeError(w, http.StatusBadRequest, "invalid binary query body: trailing bytes after frame")
		return false
	} else if !errors.Is(err, io.EOF) {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return false
		}
		writeError(w, http.StatusBadRequest, "invalid binary query body: "+err.Error())
		return false
	}

	req.Query = query
	req.K = int(k)
	req.ReadConsistency = hdr[16]
	req.OnPartitionUnavailable = hdr[17]
	req.MaxStaleness = binary.BigEndian.Uint64(hdr[20:28])
	return true
}
