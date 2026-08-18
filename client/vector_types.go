// SPDX-License-Identifier: Apache-2.0

package client

import (
	"errors"
	"strings"
)

// Consistency selects the read-consistency level for reads/searches. It is
// passed through as the server's read_consistency byte; 0 is the default
// (strong) path.
type Consistency uint8

const (
	ConsistencyDefault Consistency = 0 // strong / leader read (server default)
	// Levels 1..3 map to the server's bounded/relaxed read tiers; pass the raw
	// byte value when you need them.
)

var (
	ErrVersionConflict    = errors.New("client: version conflict (CAS expected_version mismatch)")
	ErrCollectionExists   = errors.New("client: collection already exists")
	ErrCollectionNotFound = errors.New("client: collection not found")
)

// isVersionConflict reports whether err's text looks like the server's
// CAS-conflict error (expected_version mismatch).
func isVersionConflict(err error) bool {
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "version") && (strings.Contains(s, "conflict") || strings.Contains(s, "mismatch"))
}

// isCollectionNotFound reports whether err's text looks like the server's
// missing-collection error. Two spellings reach the client depending on which
// code path acquires the collection: vector.CollectionStore's
// "vector: no collection %q" (e.g. Get, via GetPointVersionInto) and
// ops/builtin.go's "ops: unknown collection %q" (e.g. Search, via
// handleVectorSearch's own Acquire). Both substrings are on
// server/handlers.go's clientFacingErr allowlist, so they cross the wire
// verbatim instead of being redacted to "internal error".
func isCollectionNotFound(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "no collection") || strings.Contains(s, "unknown collection")
}

// mapCollErr normalizes a missing-collection server error to the exported
// ErrCollectionNotFound sentinel, leaving every other error (including a
// missing-point ErrNotFound) unchanged. Call it on the error returned from
// col.c.Call in every method that talks to the server, so
// errors.Is(err, ErrCollectionNotFound) is consistent across the client.
func mapCollErr(err error) error {
	if isCollectionNotFound(err) {
		return ErrCollectionNotFound
	}
	return err
}
