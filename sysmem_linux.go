// SPDX-License-Identifier: Apache-2.0

//go:build linux

package rostam

import "golang.org/x/sys/unix"

// systemMemoryBytes returns total host RAM, or 0 if it cannot be determined.
//
// Sysinfo reports the host's memory, NOT this process's cgroup limit, so a
// container with a memory limit below the host's RAM still derives its budget
// from the host. Callers that run under a cgroup cap should state an explicit
// budget rather than rely on the derived default.
func systemMemoryBytes() int64 {
	var si unix.Sysinfo_t
	if err := unix.Sysinfo(&si); err != nil {
		return 0
	}
	// Totalram is in units of si.Unit bytes (Unit is 1 on every mainstream
	// kernel, but honour it rather than assume).
	unit := int64(si.Unit)
	if unit == 0 {
		unit = 1
	}
	return int64(si.Totalram) * unit //nolint:gosec,unconvert // Totalram is a kernel-reported size, not attacker-controlled
}
