// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"time"
)

// resolveTTLSweepMs maps the -ttl-sweep-interval duration onto the sentinel that
// rostam.CacheConfig.TTLSweepIntervalMs uses:
//
//   - d < 0   → error (a negative interval is a misconfiguration; use 0 to disable)
//   - d == 0  → -1, meaning "disable active reaping" (lazy-on-read expiry still applies)
//   - d > 0   → the interval in whole milliseconds, floored to 1ms so a sub-ms
//     duration never collapses to 0 (which the public config reads as "library
//     default", not what the operator asked for)
func resolveTTLSweepMs(d time.Duration) (int, error) {
	switch {
	case d < 0:
		return 0, errors.New("-ttl-sweep-interval must not be negative (use 0 to disable active reaping)")
	case d == 0:
		return -1, nil
	default:
		ms := int(d.Milliseconds())
		if ms < 1 {
			ms = 1
		}
		return ms, nil
	}
}
