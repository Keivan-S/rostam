// SPDX-License-Identifier: Apache-2.0

package client

import (
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

func TestTopologyCacheGetReturnsNilWhenEmpty(t *testing.T) {
	var c topologyCache
	if got := c.get(); got != nil {
		t.Errorf("empty cache get = %+v, want nil", got)
	}
}

func TestTopologyCacheSetAndGet(t *testing.T) {
	var c topologyCache
	in := ops.Topology{
		NumShards: 2,
		Members:   []ops.TopologyMember{{NodeID: "n1", ServerAddr: "a:1"}},
		Leaders:   []string{"a:1", ""},
	}
	c.set(in)
	got := c.get()
	if got == nil {
		t.Fatal("cache empty after set")
	}
	if got.NumShards != 2 || got.Leaders[0] != "a:1" {
		t.Errorf("got %+v", *got)
	}
}

func TestTopologyCacheConcurrentReadsWrites(_ *testing.T) {
	var c topologyCache
	const writers = 4
	const readers = 16
	const iters = 1000
	var wg sync.WaitGroup
	for w := range writers {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range iters {
				c.set(ops.Topology{NumShards: id*1000 + i})
			}
		}(w)
	}
	for range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iters {
				_ = c.get()
			}
		}()
	}
	wg.Wait()
}
