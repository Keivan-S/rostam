// SPDX-License-Identifier: Apache-2.0
//go:build !cuda

package vector

// newGPUIndex / openGPUIndex are the DEFAULT (pure-Go, CGO_ENABLED=0) stubs for
// the GPU index dispatch arms in newIndex/openIndex. The GPU code path is not
// compiled into this binary, so both fail LOUD with ErrGPUNotCompiled — never a
// silent HNSW fallback. The real implementation lives in gpu_cuda.go (built only
// with -tags cuda). Validate already rejects IndexGPU here at config time; these
// are the defense-in-depth construction-path guards.

func newGPUIndex(cfg Config) (VectorIndex, error) {
	return nil, ErrGPUNotCompiled
}

func openGPUIndex(cfg Config, metaPath string) (VectorIndex, error) {
	return nil, ErrGPUNotCompiled
}
