// SPDX-License-Identifier: Apache-2.0

// Package inttest holds the slow cross-process/cluster integration tests, split out
// of the root rostam package so they compile into their own test binary with an
// independent -timeout (the root binary was ~620s, over Go's 10-minute default).
package inttest
