// SPDX-License-Identifier: Apache-2.0
//go:build cuda

package vector

// gpuSupported reports whether the GPU index code path was compiled into this
// binary. It is true ONLY in a build with -tags cuda (which also requires CGO +
// the CUDA toolkit + an NVIDIA GPU). See gpu_supported_nocuda.go for the default.
const gpuSupported = true
