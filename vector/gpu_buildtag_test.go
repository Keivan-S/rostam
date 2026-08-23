// SPDX-License-Identifier: Apache-2.0
//go:build !cuda

package vector

import (
	"errors"
	"testing"
)

// validGPUTestConfig returns a Config that passes every Validate check EXCEPT the
// IndexType gate, so a test can isolate the IndexGPU behavior (M/EfConstruction
// must be set or Validate fails on those first).
func validGPUTestConfig(it IndexType) Config {
	return Config{Dim: 8, M: 16, EfConstruction: 200, EfSearch: 64, IndexType: it}
}

// These tests pin the DEFAULT (pure-Go, CGO_ENABLED=0, !cuda) build behavior of
// the IndexGPU build-tag seam: selecting IndexGPU fails LOUD with
// ErrGPUNotCompiled at config time AND at construction time — never a silent
// HNSW fallback. They are tagged !cuda because gpuSupported is only false there;
// in a -tags cuda build the seam behaves differently (and is validated host-side
// in Slice 2).

// gpuSupported must be false in the default build (this is the whole point of the
// build-tag split: the GPU code path is not compiled in).
func TestGPUNotSupportedInDefaultBuild(t *testing.T) {
	if gpuSupported {
		t.Fatal("gpuSupported = true in the default (!cuda) build; want false")
	}
}

// Validate must reject IndexGPU with ErrGPUNotCompiled in the default build, and
// must keep accepting the real index types (HNSW/IVF/Vamana) and rejecting any
// IndexType beyond IndexGPU with ErrInvalidIndexType.
func TestValidateGPUGate(t *testing.T) {
	t.Run("IndexGPU fails loud", func(t *testing.T) {
		err := ValidateConfig(validGPUTestConfig(IndexGPU))
		if !errors.Is(err, ErrGPUNotCompiled) {
			t.Fatalf("Validate(IndexGPU) = %v; want ErrGPUNotCompiled", err)
		}
	})

	t.Run("real index types still accepted", func(t *testing.T) {
		for _, it := range []IndexType{IndexHNSW, IndexIVF, IndexVamana} {
			if err := ValidateConfig(validGPUTestConfig(it)); err != nil {
				t.Fatalf("Validate(IndexType=%d) = %v; want nil", it, err)
			}
		}
	})

	t.Run("beyond IndexGPU rejected", func(t *testing.T) {
		err := ValidateConfig(validGPUTestConfig(IndexGPU + 1))
		if !errors.Is(err, ErrInvalidIndexType) {
			t.Fatalf("Validate(IndexGPU+1) = %v; want ErrInvalidIndexType", err)
		}
	})
}

// newIndex must fail loud with ErrGPUNotCompiled for IndexGPU — NOT silently
// return an HNSW index.
func TestNewIndexGPUFailsLoud(t *testing.T) {
	idx, err := newIndex(validGPUTestConfig(IndexGPU))
	if !errors.Is(err, ErrGPUNotCompiled) {
		t.Fatalf("newIndex(IndexGPU) err = %v; want ErrGPUNotCompiled", err)
	}
	if idx != nil {
		t.Fatalf("newIndex(IndexGPU) returned a non-nil index (%T); want nil (no silent HNSW fallback)", idx)
	}
}

// openGPUIndex must also fail loud (defense-in-depth for the reopen path).
func TestOpenIndexGPUFailsLoud(t *testing.T) {
	idx, err := openGPUIndex(validGPUTestConfig(IndexGPU), "")
	if !errors.Is(err, ErrGPUNotCompiled) {
		t.Fatalf("openGPUIndex(IndexGPU) err = %v; want ErrGPUNotCompiled", err)
	}
	if idx != nil {
		t.Fatalf("openGPUIndex(IndexGPU) returned a non-nil index (%T); want nil", idx)
	}
}

// CreateCollection with IndexGPU must fail loud and create NO collection — the
// store must not end up with a (silently HNSW-backed) collection under that name.
func TestCreateCollectionGPUFailsLoud(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	err = cs.CreateCollection("gpu", validGPUTestConfig(IndexGPU))
	if !errors.Is(err, ErrGPUNotCompiled) {
		t.Fatalf("CreateCollection(IndexGPU) = %v; want ErrGPUNotCompiled", err)
	}
	if _, ok := cs.Get("gpu"); ok {
		t.Fatal("CreateCollection(IndexGPU) failed but the collection was still created (silent fallback)")
	}
}
