// SPDX-License-Identifier: Apache-2.0

//go:build race

package httpapi

// raceEnabled is true in -race builds. The race detector instruments every
// allocation, which inflates runtime.MemStats.TotalAlloc for identical
// legitimate work — see binary_bulk_alloc_test.go for the measured effect on
// the allocation-amplification guards.
const raceEnabled = true
