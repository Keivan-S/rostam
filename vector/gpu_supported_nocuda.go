// SPDX-License-Identifier: Apache-2.0
//go:build !cuda

package vector

// gpuSupported reports whether the GPU index code path was compiled into this
// binary. In the DEFAULT (pure-Go, CGO_ENABLED=0) build it is false: selecting
// IndexGPU fails LOUD with ErrGPUNotCompiled at config time. See
// gpu_supported_cuda.go for the -tags cuda build.
const gpuSupported = false
