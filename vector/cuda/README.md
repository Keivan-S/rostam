# CUDA exact-KNN kernel (`-tags cuda` GPU index)

This directory holds the hand-written CUDA brute-force / exact-KNN kernel and its
cgo binding for the OPTIONAL GPU index (`IndexGPU`, selected only in a build with
`-tags cuda`). The DEFAULT pure-Go, `CGO_ENABLED=0`, single-static-binary build
never compiles any of this — `IndexGPU` there fails loud with `ErrGPUNotCompiled`.

## Files

- `knn.h` / `knn.cu` — the CUDA kernel: a resident corpus float32 matrix on the
  device + an exact per-query top-k (one thread per query, insertion top-k over
  all rows). `extern "C"` wrappers: `gpuKNNCreate` / `gpuKNNUpload` /
  `gpuKNNSearch` / `gpuKNNFree`. No external dependency beyond the base CUDA
  runtime (no cuBLAS/cuVS).
- `knn.go` (`//go:build cuda`) — the cgo binding (`package cuda`), exposing a
  Go-typed `Handle`. It lives in its OWN package because the parent `vector`
  package has hand-written Go assembly (`distance_amd64.s`, ...) and the Go
  toolchain forbids mixing cgo with Go-syntax assembly in one package.
- `Makefile` — runs `nvcc` → `libknn.a`.
- `.gitignore` — the `*.o` / `*.a` build artifacts are local-only (not committed).

## Build + run (host-side, requires an NVIDIA GPU + CUDA toolkit)

```sh
# 1. Compile the kernel into libknn.a (needs nvcc).
make -C vector/cuda

# 2. Build / test the GPU index (needs CGO + libcudart).
CGO_ENABLED=1 go build -tags cuda ./vector/...
CGO_ENABLED=1 go test  -tags cuda ./vector/ -run GPU -count=1
```

The cgo directives in `knn.go` link `libknn.a` plus the system `libcudart`
(`/usr/lib/x86_64-linux-gnu/libcudart.so.12` on Debian/Ubuntu) and `libstdc++`.
No `go.mod` dependency is added — the GPU binding is a build/link-time C dep, not
a Go module.

## Correctness model

The GPU index is EXACT brute force: it scores every live vector and selects an
exact top-k, so a GPU result matches a CPU exact brute force (ids + distances
within float tolerance) for every metric (L2 / Cosine / DotProduct). The kernel
returns the raw dot (Cosine/Dot) or squared distance (L2); the Go side reorients
to the metric's `distFunc` convention (smaller = nearer) and applies the same
tombstone / TTL / filter admission gate the CPU path uses. See
`vector/gpu_cuda.go` and `vector/gpu_cuda_test.go`.

### Kernel limit + CPU-exact fallback

A single kernel dispatch emits at most `GPU_MAX_K` (256, surfaced to Go as
`cuda.MaxK`) candidates per query — this sizes the per-thread top-k scratch in
local memory. The host NEVER asks the kernel for more (a `k > GPU_MAX_K` is
rejected, not silently clamped). The **GPU fast path** covers `k <= 256` on a
non-selective query. When the admitted top-k cannot be drawn from the GPU
top-256 — a highly selective filter, more than 256 tombstones, or `k > 256` —
the host transparently falls back to a **CPU-exact brute force** over the live
arena (`cpuExactSearchLocked`). The returned top-k is therefore always exact and
never silently truncated to 256; the GPU buys throughput for the common case.

### GPU-exact surface

Only the four core dense-KNN entry points (`Search`, `SearchFiltered`,
`SearchInto`, `SearchFilteredWith`) are GPU-exact. `HybridSearch`, `SearchMMR`,
`SearchGroups`, `DiscoverVecs` and `RecommendVecs` are inherited from the inner
HNSW and serve APPROXIMATE graph results (not re-routed through the GPU kernel).
