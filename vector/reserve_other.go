// SPDX-License-Identifier: Apache-2.0
//go:build !linux || !(amd64 || arm64)

package vector

import "os"

// slabReservation is the no-op stand-in wherever reserve-then-commit-in-place is
// not available: every non-Linux target, plus 32-bit Linux, where the scheme's
// free-address-space premise does not hold (see reserve_linux.go's build tag).
// Every slab keeps the copy/remap growth path it had before — reserve.go's point
// is that a reservation is an optimization, never a requirement. Where mmap
// storage is unavailable (mmap_other.go) this only affects heap-backed slabs;
// on Windows and 32-bit Linux, which do have mmap storage, it affects both, and
// both stay correct.
type slabReservation struct{}

// slabReservationsSupported is false here, so tests that only mean something with
// reservations skip instead of failing.
const slabReservationsSupported = false

func (r *slabReservation) mapped() bool { return false }

func newSlabReservation(_ *os.File, _, _ int64) (*slabReservation, error) {
	return nil, errSlabReserveUnsupported
}

func (r *slabReservation) commitTo(_ int64) error { return errSlabReserveUnsupported }

func (r *slabReservation) region() []byte { return nil }

func (r *slabReservation) sync() error { return nil }

func (r *slabReservation) release() error { return nil }
