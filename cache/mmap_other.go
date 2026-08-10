// SPDX-License-Identifier: Apache-2.0
//go:build !linux

package cache

import (
	"errors"
	"os"
)

func mmapFile(_ string, _ int64, _ bool) (*os.File, []byte, error) {
	return nil, nil, errors.New("cache: mmap not supported on this platform; set Config.DataDir=\"\"")
}

func mmapFileAlloc(_ string, _, _ int64, _ bool) (*os.File, []byte, error) {
	return nil, nil, errors.New("cache: mmap not supported on this platform; set Config.DataDir=\"\"")
}

func msync(_ []byte) error { return nil }

func munmapAndClose(_ *os.File, _ []byte) error { return nil }
