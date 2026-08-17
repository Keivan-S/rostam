// SPDX-License-Identifier: Apache-2.0
package client

import "errors"

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
