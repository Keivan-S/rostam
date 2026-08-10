// SPDX-License-Identifier: Apache-2.0
//go:build !linux

package vector

import (
	"errors"
	"os"
)

// errMmapUnsupported is returned on platforms without mmap support; callers
// should fall back to QuantInRAM.
var errMmapUnsupported = errors.New("vector: mmap-backed vectors not supported on this platform (use QuantInRAM)")

func openVecMmap(_ string, _ int64) (*os.File, []byte, error) {
	return nil, nil, errMmapUnsupported
}

func growVecMmap(_ *os.File, _ []byte, _ int64) ([]byte, error) {
	return nil, errMmapUnsupported
}

func syncVecMmap(_ []byte) error { return errMmapUnsupported }

func closeVecMmap(_ *os.File, _ []byte) error { return nil }
