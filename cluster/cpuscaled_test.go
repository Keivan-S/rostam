// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"runtime"
	"time"
)

// cpuScaled widens a setup/convergence deadline for CPU-constrained CI. GitHub's
// 2-vCPU runners are far slower and noisier than a developer's throttled cores,
// and -race adds ~10x on top, so deadlines scale with the core budget and again
// under -race. Upper bounds only: a healthy run returns well before them, so
// generous scaling costs wall-time only on a genuine failure.
func cpuScaled(d time.Duration) time.Duration {
	f := 1
	switch n := runtime.GOMAXPROCS(0); {
	case n <= 2:
		f = 4
	case n <= 4:
		f = 2
	}
	if raceEnabled {
		f *= 2
	}
	return d * time.Duration(f)
}
