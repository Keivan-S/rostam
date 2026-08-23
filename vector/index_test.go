// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

func TestConfigDefaults(t *testing.T) {
	c := DefaultConfig()
	if c.M != 16 || c.EfConstruction != 200 || c.EfSearch != 64 {
		t.Fatalf("default config: got %+v", c)
	}
}

func TestConfigPartitionsValidation(t *testing.T) {
	base := Config{Dim: 8, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64}
	// Default (0) and 1 are single-partition and valid.
	for _, p := range []int{0, 1, 8, 256} {
		c := base
		c.Partitions = p
		if err := ValidateConfig(c); err != nil {
			t.Errorf("Partitions=%d: unexpected error %v", p, err)
		}
	}
	// Negative is invalid.
	c := base
	c.Partitions = -1
	err := ValidateConfig(c)
	if err == nil {
		t.Error("Partitions=-1 should be invalid")
	} else if !errors.Is(err, ErrInvalidPartitions) {
		t.Errorf("Partitions=-1: want ErrInvalidPartitions, got %v", err)
	}
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want error
	}{
		{"dim zero", Config{Dim: 0, M: 16, EfConstruction: 200, EfSearch: 64}, ErrInvalidDim},
		{"dim negative", Config{Dim: -1, M: 16, EfConstruction: 200, EfSearch: 64}, ErrInvalidDim},
		{"unknown metric", Config{Dim: 128, Metric: Metric(99), M: 16, EfConstruction: 200, EfSearch: 64}, ErrInvalidMetric},
		{"m zero", Config{Dim: 128, M: 0, EfConstruction: 200, EfSearch: 64}, ErrInvalidM},
		{"m too big", Config{Dim: 128, M: 129, EfConstruction: 200, EfSearch: 64}, ErrInvalidM},
		{"ef construction zero", Config{Dim: 128, M: 16, EfConstruction: 0, EfSearch: 64}, ErrInvalidEf},
		{"ef search zero", Config{Dim: 128, M: 16, EfConstruction: 200, EfSearch: 0}, ErrInvalidEf},
		{"valid", Config{Dim: 128, M: 16, EfConstruction: 200, EfSearch: 64}, nil},
		{"ivf valid", Config{Dim: 128, M: 16, EfConstruction: 200, EfSearch: 64, IndexType: IndexIVF, IVFNlist: 64, IVFNprobe: 8}, nil},
		// IVF + Persistent is now supported via the instant-restart mmap sidecar
		// (un-rejected). Both quantized and IVF-Flat (QuantNone) persistent IVF validate.
		{"ivf + persistent quantized valid", Config{Dim: 128, M: 16, EfConstruction: 200, EfSearch: 64, IndexType: IndexIVF, Persistent: true, Quant: QuantSQ8}, nil},
		{"ivf-flat + persistent valid", Config{Dim: 128, M: 16, EfConstruction: 200, EfSearch: 64, IndexType: IndexIVF, IVFNlist: 64, IVFNprobe: 8, Persistent: true}, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ValidateConfig(tc.cfg)
			if !errors.Is(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}
