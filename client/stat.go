// SPDX-License-Identifier: Apache-2.0

package client

// Stat is a per-server pool snapshot suitable for logging or metrics.
type Stat struct {
	// AcquireCount is the number of successful Acquires.
	AcquireCount int64
	// Total active connections (in use + idle).
	TotalConns int32
	IdleConns  int32
	// Counters for connections destroyed by lifecycle policies.
	NewConnsCount        int64
	LifetimeDestroyCount int64
	IdleDestroyCount     int64
}
