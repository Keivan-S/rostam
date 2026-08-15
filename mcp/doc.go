// SPDX-License-Identifier: Apache-2.0

// Package mcp implements a Model Context Protocol (MCP) server over stdio,
// exposing agent-memory and vector-database tools backed by any rostam.Store
// (embedded or remote). The protocol layer is a hand-rolled subset of MCP —
// newline-delimited JSON-RPC 2.0 with the tools capability — kept
// dependency-free by design, like the rest of the repository's protocol code.
package mcp
