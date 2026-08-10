// knn.cu — CUDA exact (brute-force) KNN kernel for the -tags cuda GPU index.
//
// Design (v1, exact == correctness anchor):
//   * The corpus is a row-major float32 matrix (n x dim) kept RESIDENT on the
//     device (gpuKNNUpload). It is re-uploaded only when the host marks the
//     index dirty (insert/delete), so the device->host copy is amortized across
//     many queries — the "real acceleration" property (not upload-per-query).
//   * One CUDA thread per query. Each thread streams the whole corpus once,
//     computing the metric value to every row, and maintains an exact top-k via
//     insertion into a small per-thread array held in local memory. Because we
//     scan ALL rows and keep an exact partial sort, the result is the EXACT
//     top-k — identical (within float tolerance) to a CPU brute force.
//   * Batched queries give throughput (each thread is independent); k is small
//     (KNN), so the O(k) insertion per kept candidate is cheap relative to the
//     O(dim) distance.
//
// This deliberately avoids any external dependency (cuBLAS/cuVS): only the base
// CUDA runtime. A GEMM-based distance matrix + bitonic top-k is a future
// throughput follow-up; correctness/parity is the goal here.

#include "knn.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>

// Metric codes — must match knn.h / vector.Metric.
#define METRIC_COSINE 0
#define METRIC_L2     1
#define METRIC_DOT    2

// MAX_K bounds the per-thread top-k scratch array (kept in local memory). It is
// defined from GPU_MAX_K in knn.h (the single source of truth, also surfaced to
// the Go host as cuda.MaxK). KNN k is small in practice; the host guarantees it
// never dispatches with k > MAX_K (it falls back to a CPU-exact brute force
// instead), so gpuKNNSearch REJECTS k > MAX_K rather than silently clamping.
#define MAX_K GPU_MAX_K

struct gpuKNNHandle {
    int dim;
    int n;          // resident row count
    float* dCorpus; // device corpus (n*dim floats), NULL when n==0
    int cap;        // allocated capacity in ROWS for dCorpus
};

extern "C" gpuKNNHandle* gpuKNNCreate(int dim) {
    if (dim <= 0) return NULL;
    gpuKNNHandle* h = (gpuKNNHandle*)calloc(1, sizeof(gpuKNNHandle));
    if (!h) return NULL;
    h->dim = dim;
    h->n = 0;
    h->dCorpus = NULL;
    h->cap = 0;
    return h;
}

extern "C" int gpuKNNUpload(gpuKNNHandle* h, const float* corpus, int n) {
    if (!h) return 1;
    if (n < 0) return 1;
    // Grow the device buffer if needed (reuse it across re-uploads of <= cap).
    if (n > h->cap) {
        if (h->dCorpus) {
            cudaFree(h->dCorpus);
            h->dCorpus = NULL;
            h->cap = 0;
        }
        size_t bytes = (size_t)n * (size_t)h->dim * sizeof(float);
        cudaError_t err = cudaMalloc((void**)&h->dCorpus, bytes);
        if (err != cudaSuccess) {
            h->dCorpus = NULL;
            h->cap = 0;
            h->n = 0;
            return 2;
        }
        h->cap = n;
    }
    if (n > 0) {
        size_t bytes = (size_t)n * (size_t)h->dim * sizeof(float);
        cudaError_t err = cudaMemcpy(h->dCorpus, corpus, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            h->n = 0;
            return 3;
        }
    }
    h->n = n;
    return 0;
}

// knnKernel: one thread per query. Scans the full corpus, keeps an exact top-k.
// For Cosine/Dot, "best" = LARGEST dot (nearest); for L2, "best" = SMALLEST
// squared distance. The kept arrays are ordered nearest-first so position 0 is
// the running best and the worst kept is at the tail (the eviction candidate).
__global__ void knnKernel(const float* __restrict__ corpus, int n, int dim,
                          const float* __restrict__ queries, int nq, int k,
                          int metric, int* __restrict__ outIdx,
                          float* __restrict__ outScore) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq) return;

    const float* q = queries + (size_t)qi * dim;

    // Per-thread top-k scratch (local memory). Ordered nearest-first.
    int   topIdx[MAX_K];
    float topScore[MAX_K];
    int   count = 0;

    // larger==better for Cosine/Dot (dot); smaller==better for L2.
    const bool largerBetter = (metric != METRIC_L2);

    for (int r = 0; r < n; r++) {
        const float* row = corpus + (size_t)r * dim;
        float s;
        if (metric == METRIC_L2) {
            float acc = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = q[d] - row[d];
                acc += diff * diff;
            }
            s = acc; // squared L2
        } else {
            float acc = 0.0f;
            for (int d = 0; d < dim; d++) {
                acc += q[d] * row[d];
            }
            s = acc; // dot (Cosine over normalized rows, or DotProduct)
        }

        // Insert (r, s) into the nearest-first top-k if it qualifies.
        if (count < k) {
            // Insertion sort into the growing array.
            int p = count;
            while (p > 0) {
                bool better = largerBetter ? (s > topScore[p - 1])
                                           : (s < topScore[p - 1]);
                if (!better) break;
                topScore[p] = topScore[p - 1];
                topIdx[p]   = topIdx[p - 1];
                p--;
            }
            topScore[p] = s;
            topIdx[p]   = r;
            count++;
        } else {
            // Full: compare against the worst kept (tail).
            bool better = largerBetter ? (s > topScore[k - 1])
                                       : (s < topScore[k - 1]);
            if (!better) continue;
            int p = k - 1;
            while (p > 0) {
                bool b2 = largerBetter ? (s > topScore[p - 1])
                                       : (s < topScore[p - 1]);
                if (!b2) break;
                topScore[p] = topScore[p - 1];
                topIdx[p]   = topIdx[p - 1];
                p--;
            }
            topScore[p] = s;
            topIdx[p]   = r;
        }
    }

    // Emit. Pad with -1 / sentinel when fewer than k rows are available.
    int* oi = outIdx + (size_t)qi * k;
    float* os = outScore + (size_t)qi * k;
    for (int i = 0; i < k; i++) {
        if (i < count) {
            oi[i] = topIdx[i];
            os[i] = topScore[i];
        } else {
            oi[i] = -1;
            os[i] = largerBetter ? -FLT_MAX : FLT_MAX;
        }
    }
}

extern "C" int gpuKNNSearch(gpuKNNHandle* h, const float* queries, int nq,
                            int k, int metric, int* outIdx, float* outScore) {
    if (!h) return 1;
    if (nq <= 0 || k <= 0) return 0;
    // The host must never ask for more than MAX_K (it uses a CPU-exact fallback
    // above that). Reject loud rather than silently clamp to 256 — a silent
    // clamp is exactly the over-fetch bug this guards against.
    if (k > MAX_K) return 4;
    // Self-protecting ABI: a non-empty handle MUST have an uploaded device
    // corpus before the kernel reads h->dCorpus. The host maintains this
    // invariant, but guard it here too (matching the fail-loud k > MAX_K check)
    // so a mis-driven handle returns an error instead of dereferencing NULL on
    // the device.
    if (h->n > 0 && !h->dCorpus) return 5;
    if (k > h->n) k = h->n;       // cannot return more than the resident count
    if (k <= 0) {                  // empty corpus: fill sentinels, no kernel
        return 0;
    }

    size_t qBytes = (size_t)nq * (size_t)h->dim * sizeof(float);
    size_t oiBytes = (size_t)nq * (size_t)k * sizeof(int);
    size_t osBytes = (size_t)nq * (size_t)k * sizeof(float);

    float* dQ = NULL;
    int* dOi = NULL;
    float* dOs = NULL;
    cudaError_t err;

    err = cudaMalloc((void**)&dQ, qBytes);
    if (err != cudaSuccess) { return 10; }
    err = cudaMalloc((void**)&dOi, oiBytes);
    if (err != cudaSuccess) { cudaFree(dQ); return 11; }
    err = cudaMalloc((void**)&dOs, osBytes);
    if (err != cudaSuccess) { cudaFree(dQ); cudaFree(dOi); return 12; }

    err = cudaMemcpy(dQ, queries, qBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto fail; }

    {
        int threads = 128;
        int blocks = (nq + threads - 1) / threads;
        knnKernel<<<blocks, threads>>>(h->dCorpus, h->n, h->dim, dQ, nq, k,
                                       metric, dOi, dOs);
        err = cudaGetLastError();
        if (err != cudaSuccess) { goto fail; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { goto fail; }
    }

    err = cudaMemcpy(outIdx, dOi, oiBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { goto fail; }
    err = cudaMemcpy(outScore, dOs, osBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { goto fail; }

    cudaFree(dQ);
    cudaFree(dOi);
    cudaFree(dOs);
    return 0;

fail:
    cudaFree(dQ);
    cudaFree(dOi);
    cudaFree(dOs);
    return 20;
}

extern "C" void gpuKNNFree(gpuKNNHandle* h) {
    if (!h) return;
    if (h->dCorpus) cudaFree(h->dCorpus);
    free(h);
}
