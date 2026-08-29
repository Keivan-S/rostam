// SPDX-License-Identifier: Apache-2.0
//go:build !linux && !windows

package cache

import (
	"errors"
	"os"
)

// mmapSupported reports that no shard on this platform can be backed by a file
// mapping, so Config.Validate rejects a non-empty DataDir.
const mmapSupported = false

func mmapFile(_ string, _ int64, _ bool) (*os.File, []byte, error) {
	return nil, nil, errors.New("cache: mmap not supported on this platform; set Config.DataDir=\"\"")
}

func mmapFileAlloc(_ string, _, _ int64, _ bool) (*os.File, []byte, error) {
	return nil, nil, errors.New("cache: mmap not supported on this platform; set Config.DataDir=\"\"")
}

func msync(_ *os.File, _ []byte) error { return nil }

func munmapAndClose(_ *os.File, _ []byte) error { return nil }
