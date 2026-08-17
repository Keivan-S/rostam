// SPDX-License-Identifier: Apache-2.0
package client

import (
	"errors"
	"testing"
)

func TestCollectionHandleBindsName(t *testing.T) {
	c := &Client{} // no I/O in this test
	col := c.Collection("posts")
	if col == nil || col.name != "posts" {
		t.Fatalf("Collection() = %+v, want handle bound to \"posts\"", col)
	}
	if col.c != c {
		t.Fatal("Collection() did not retain its client")
	}
}

func TestVectorErrorsAreDistinct(t *testing.T) {
	for _, err := range []error{ErrNotFound, ErrVersionConflict, ErrCollectionExists, ErrCollectionNotFound} {
		if err == nil {
			t.Fatal("sentinel error is nil")
		}
	}
	if errors.Is(ErrNotFound, ErrVersionConflict) {
		t.Fatal("sentinels must not alias each other")
	}
}
