// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"testing"
	"time"
)

func TestExpiryAtZero(t *testing.T) {
	if got := expiryAt(0); got != 0 {
		t.Errorf("expiryAt(0) = %d, want 0 (no expiry sentinel)", got)
	}
}

func TestExpiryAtTTL(t *testing.T) {
	now := uint64(time.Now().UnixMilli())
	got := expiryAt(5 * time.Second)
	if got < now+4900 || got > now+5100 {
		t.Errorf("expiryAt(5s) = %d, want ~now+5000 (now=%d)", got, now)
	}
}

func TestIsExpired(t *testing.T) {
	now := uint64(time.Now().UnixMilli())
	if isExpired(0, now) {
		t.Error("0 expiry must never be expired")
	}
	if isExpired(now+1000, now) {
		t.Error("future expiry must not be expired")
	}
	if !isExpired(now-1, now) {
		t.Error("past expiry must be expired")
	}
}
