// SPDX-License-Identifier: Apache-2.0
//go:build cuda

// Package cuda is the cgo binding for the CUDA exact-KNN kernel (knn.cu/knn.h).
// It lives in its OWN package because the parent `vector` package contains
// hand-written Go assembly (distance_amd64.s, fastscan_amd64.s,
// distance_avx512_amd64.s), and the Go toolchain forbids a single package from
// mixing cgo with Go-syntax assembly. Isolating the cgo here keeps the parent
// package assembly-only and lets gpu_cuda.go (also //go:build cuda) call into a
// Go-typed handle.
//
// Build: the libknn.a archive must exist before building this package —
//
//	make -C vector/cuda
//	CGO_ENABLED=1 go build -tags cuda ./vector/...
//
// The whole package is gated behind -tags cuda (this file's build tag), so the
// DEFAULT (pure-Go, CGO_ENABLED=0, !cuda) build never compiles it and never
// needs nvcc / libcudart.
//
// CUDA library path: the LDFLAGS below add the conventional CUDA Toolkit
// location (/usr/local/cuda/lib64) so the build works on a stock CUDA install
// regardless of distro. If libcudart lives elsewhere (a distro package, a
// versioned toolkit, a non-standard prefix), point the build at it WITHOUT
// editing this file via the standard cgo environment variable, e.g.:
//
//	CGO_LDFLAGS="-L$CUDA_HOME/lib64" CGO_ENABLED=1 go build -tags cuda ./vector/...
//
// CGO_LDFLAGS is APPENDED to the directives here by the Go toolchain, so the
// extra -L is searched first. The previous hard-coded Debian/Ubuntu multiarch
// path (-L/usr/lib/x86_64-linux-gnu) broke the build on every other distro.
package cuda

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -L/usr/local/cuda/lib64 -lknn -lcudart -lstdc++
#include <stdlib.h>
#include "knn.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Metric codes mirror vector.Metric (Cosine=0, L2=1, DotProduct=2) and the
// kernel's METRIC_* defines.
const (
	MetricCosine     = 0
	MetricL2         = 1
	MetricDotProduct = 2
)

// MaxK is the hard upper bound on the per-query top-k the kernel can produce. It
// is surfaced from the kernel's GPU_MAX_K macro (knn.h) so the Go host can guard
// against it: the host never dispatches Search with k > MaxK and falls back to a
// CPU-exact brute force above it (or when a selective filter / tombstone purge
// would require over-fetching more than MaxK). Sourced from C so there is a
// single source of truth for the limit.
const MaxK = C.GPU_MAX_K

// Handle is a device context: a resident corpus buffer + reusable scratch on
// one GPU. Not safe for concurrent use; the caller (gpuIndex) serializes via
// its own lock.
type Handle struct {
	h   *C.gpuKNNHandle
	dim int
}

// New creates a device context for vectors of dimension dim. Returns an error
// when the device cannot be initialized (no GPU / out of memory).
func New(dim int) (*Handle, error) {
	if dim <= 0 {
		return nil, errors.New("cuda: dim must be > 0")
	}
	h := C.gpuKNNCreate(C.int(dim))
	if h == nil {
		return nil, errors.New("cuda: gpuKNNCreate failed (no GPU or out of memory)")
	}
	return &Handle{h: h, dim: dim}, nil
}

// Upload (re)uploads the resident corpus: n row-major float32 vectors. corpus
// must have length n*dim (n == 0 empties the resident corpus). The prior buffer
// is reused when large enough, otherwise reallocated.
func (h *Handle) Upload(corpus []float32, n int) error {
	if h.h == nil {
		return errors.New("cuda: Upload on closed handle")
	}
	if n < 0 || (n > 0 && len(corpus) < n*h.dim) {
		return errors.New("cuda: Upload corpus too short for n*dim")
	}
	var ptr *C.float
	if n > 0 {
		ptr = (*C.float)(unsafe.Pointer(&corpus[0]))
	}
	rc := C.gpuKNNUpload(h.h, ptr, C.int(n))
	if rc != 0 {
		return errors.New("cuda: gpuKNNUpload failed")
	}
	return nil
}

// Search runs an exact KNN over the resident corpus for nq row-major queries
// (queries has length nq*dim), returning per-query top-k row indices and their
// raw kernel scores (dot for Cosine/Dot, squared distance for L2). outIdx and
// outScore are filled in place; each must be at least nq*k long. k is clamped to
// the resident row count internally; missing rows are -1 / sentinel. The caller
// reorients scores to its distFunc convention.
func (h *Handle) Search(queries []float32, nq, k, metric int, outIdx []int32, outScore []float32) error {
	if h.h == nil {
		return errors.New("cuda: Search on closed handle")
	}
	if nq <= 0 || k <= 0 {
		return nil
	}
	if len(queries) < nq*h.dim {
		return errors.New("cuda: Search queries too short for nq*dim")
	}
	if len(outIdx) < nq*k || len(outScore) < nq*k {
		return errors.New("cuda: Search output buffers too short for nq*k")
	}
	rc := C.gpuKNNSearch(h.h,
		(*C.float)(unsafe.Pointer(&queries[0])),
		C.int(nq), C.int(k), C.int(metric),
		(*C.int)(unsafe.Pointer(&outIdx[0])),
		(*C.float)(unsafe.Pointer(&outScore[0])))
	if rc != 0 {
		return errors.New("cuda: gpuKNNSearch failed")
	}
	return nil
}

// Free releases the device context. Idempotent.
func (h *Handle) Free() {
	if h.h != nil {
		C.gpuKNNFree(h.h)
		h.h = nil
	}
}
