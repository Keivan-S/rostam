// knn.h — C ABI for the CUDA exact (brute-force) KNN kernel used by the
// -tags cuda GPU index (gpuIndex). Pure C declarations so cgo can bind them
// without seeing any CUDA/C++ types. The corpus is a row-major float32 matrix
// (n rows x dim cols) kept resident on the device; queries are scored against
// every row and an exact per-query top-k is selected on the GPU.
//
// Metric codes (must match vector.Metric in distance.go):
//   0 = Cosine     -> kernel computes dot(q, row); host converts to 1 - dot
//                     (both corpus rows and the query are pre-normalized, so the
//                     dot is the cosine similarity).
//   1 = L2         -> kernel computes sum((q-row)^2), the squared Euclidean
//                     distance (smaller = nearer), matching l2Squared.
//   2 = DotProduct -> kernel computes dot(q, row); host negates to -dot.
//
// For Cosine/DotProduct the kernel returns the raw dot product as the "score"
// and selects the LARGEST dots (nearest); for L2 it returns the squared
// distance and selects the SMALLEST. The Go side reorients the returned values
// to the metric's distFunc convention (smaller = nearer) before building
// Result. This keeps the GPU's selection numerically identical to a CPU exact
// brute force under the same metric.

#ifndef ROSTAM_CUDA_KNN_H
#define ROSTAM_CUDA_KNN_H

// GPU_MAX_K is the hard upper bound on the per-query top-k the kernel can
// produce: it sizes the per-thread top-k scratch array held in local memory
// (see MAX_K in knn.cu, which is defined from this macro — single source of
// truth). The Go host surfaces it as cuda.MaxK and NEVER dispatches a kernel
// call with k greater than this; for any requested k beyond it (or whenever the
// over-fetch needed to satisfy a selective filter / tombstone purge exceeds it)
// the host falls back to a CPU-exact brute force so the top-k contract is never
// silently truncated.
#define GPU_MAX_K 256

#ifdef __cplusplus
extern "C" {
#endif

// gpuKNNHandle is an opaque per-index device context (resident corpus buffer +
// reusable query/result scratch). NULL on a failed gpuKNNCreate.
typedef struct gpuKNNHandle gpuKNNHandle;

// gpuKNNCreate allocates a device context for vectors of the given dimension.
// Returns NULL on allocation failure (no GPU, OOM). dim must be > 0.
gpuKNNHandle* gpuKNNCreate(int dim);

// gpuKNNUpload (re)uploads the resident corpus: `n` row-major float32 vectors
// (corpus has length n*dim). It (re)allocates the device corpus buffer to hold
// n rows and copies the host data up. A prior buffer is freed first. n == 0 is
// valid (empties the resident corpus). Returns 0 on success, non-zero on a CUDA
// error (the handle is left with an empty corpus on failure).
int gpuKNNUpload(gpuKNNHandle* h, const float* corpus, int n);

// gpuKNNSearch runs an exact KNN over the resident corpus for a batch of
// `nq` queries (queries has length nq*dim, row-major) returning, for each
// query, the top-k row indices (into the uploaded corpus order) and their
// scores. outIdx and outScore are caller-allocated, each nq*k long, row-major
// per query. k is clamped to the resident row count n internally; rows beyond
// the available count are filled with index -1 and a sentinel score. metric is
// one of the codes above. k MUST be <= GPU_MAX_K (the host enforces this and
// uses a CPU-exact fallback above it); a k > GPU_MAX_K is REJECTED (returns 4)
// rather than silently clamped. Returns 0 on success, non-zero on a CUDA error.
//
// "score" is the raw metric value the kernel computed (dot for Cosine/Dot,
// squared distance for L2); selection order is nearest-first under the metric
// (largest dot / smallest L2). The Go side reorients to distFunc convention.
int gpuKNNSearch(gpuKNNHandle* h, const float* queries, int nq, int k,
                 int metric, int* outIdx, float* outScore);

// gpuKNNFree releases the device context (resident corpus + scratch). Safe on
// NULL.
void gpuKNNFree(gpuKNNHandle* h);

#ifdef __cplusplus
}
#endif

#endif // ROSTAM_CUDA_KNN_H
