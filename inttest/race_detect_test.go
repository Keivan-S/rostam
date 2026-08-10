// SPDX-License-Identifier: Apache-2.0

//go:build race

package inttest

// raceEnabled is true in -race builds. The race detector slows execution ~10x,
// so CPU-contended-CI deadline scaling (cpuScaled) widens further under it.
const raceEnabled = true
