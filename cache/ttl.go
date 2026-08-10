// SPDX-License-Identifier: Apache-2.0

package cache

import "time"

// expiryAt returns the absolute expiry timestamp in milliseconds since
// the Unix epoch for the given TTL. A TTL of zero returns zero, which is
// the "no expiry" sentinel.
func expiryAt(ttl time.Duration) uint64 {
	if ttl <= 0 {
		return 0
	}
	return uint64(time.Now().Add(ttl).UnixMilli())
}

// expiryAtFrom is expiryAt against an EXPLICIT base clock (nowMs) rather than
// wall time. It is the deterministic-apply counterpart of expiryAt: on the
// replicated apply path every replica computes exp = leaderStampMs + ttl from
// the SAME leader-stamped nowMs baked into the log entry, so all replicas store
// byte-identical absolute expiries regardless of their own wall clocks (#4 Phase
// B / B1). A TTL of zero returns the 0 "no expiry" sentinel, matching expiryAt.
func expiryAtFrom(ttl time.Duration, nowMs uint64) uint64 {
	if ttl <= 0 {
		return 0
	}
	return nowMs + uint64(ttl.Milliseconds()) //nolint:gosec // ttl > 0 checked above; ms is non-negative
}

// isExpired returns true if expiryMs is non-zero and <= now.
func isExpired(expiryMs, nowMs uint64) bool {
	return expiryMs != 0 && expiryMs <= nowMs
}

// nowMs returns the current Unix timestamp in milliseconds.
func nowMs() uint64 {
	return uint64(time.Now().UnixMilli())
}
