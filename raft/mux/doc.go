// SPDX-License-Identifier: Apache-2.0

// Package mux provides a multiplexed [hashicorp/raft.StreamLayer]
// over a single TCP listener. Every outbound connection is prefixed
// with a 4-byte big-endian group ID; accepted connections are routed
// to the per-group accept queue matching that ID.
//
// One mux StreamLayer serves N Raft groups (e.g., 256 data shards plus
// one meta-Raft) on a single port per node. The per-group
// raft.StreamLayer returned by [StreamLayer.For] is what hashicorp/raft
// dials and listens on; the mux is transparent to the Raft library.
package mux
