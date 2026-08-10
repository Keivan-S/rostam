// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

func benchStore(b *testing.B, noSync bool) *Store {
	b.Helper()
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID: "node-bench", DataDir: b.TempDir(),
		Cache: cc, Ops: reg,
		Bootstrap:       true,
		RaftHeartbeatMs: 50, RaftElectionMs: 100,
		NoSync: noSync,
	}
	s, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = s.Close() })
	// Wait for leader.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if s.IsLeader() {
			return s
		}
		time.Sleep(10 * time.Millisecond)
	}
	b.Fatal("node never became leader")
	return nil
}

func BenchmarkStorePut_NoSync(b *testing.B) {
	s := benchStore(b, true)
	val := make([]byte, 256)
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := fmt.Appendf(nil, "k%07d", i)
		_ = s.Put(k, val, 0)
		i++
	}
}

func BenchmarkStorePut_Sync(b *testing.B) {
	s := benchStore(b, false)
	val := make([]byte, 256)
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := fmt.Appendf(nil, "k%07d", i)
		_ = s.Put(k, val, 0)
		i++
	}
}

func BenchmarkStoreGet(b *testing.B) {
	s := benchStore(b, true)
	val := make([]byte, 256)
	for i := range 10_000 {
		k := fmt.Appendf(nil, "k%07d", i)
		_ = s.Put(k, val, 0)
	}
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := fmt.Appendf(nil, "k%07d", i%10_000)
		_, _ = s.Get(k)
		i++
	}
}

func BenchmarkStoreCall_ReadOnly(b *testing.B) {
	s := benchStore(b, true)
	val := make([]byte, 256)
	for i := range 10_000 {
		k := fmt.Appendf(nil, "k%07d", i)
		_ = s.Put(k, val, 0)
	}
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := fmt.Appendf(nil, "k%07d", i%10_000)
		_, _ = s.Call("get", ops.EncodeKeyArgs(k))
		i++
	}
}
