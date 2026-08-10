// SPDX-License-Identifier: Apache-2.0

package rostam_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// Example_embedded shows the minimum needed to bring up a single-node
// in-process Rostam and round-trip a value.
func Example_embedded() {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		log.Fatal(err)
	}

	dir, err := os.MkdirTemp("", "rostam-example-*")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	store, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "demo",
		DataDir:   dir,
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	// Wait for leader; production code should bound this and handle the
	// timeout case explicitly.
	deadline := time.Now().Add(5 * time.Second)
	for !store.IsLeader([]byte("k")) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}

	ctx := context.Background()
	_ = store.Put(ctx, []byte("k"), []byte("v"), 0)
	got, _ := store.Get(ctx, []byte("k"))
	fmt.Printf("got=%q\n", got)
	// Output: got="v"
}

// Example_registerOp shows how to register a custom atomic
// read-modify-write op and call it through the Store interface. The
// same pattern works in both Embedded and Client mode.
func Example_registerOp() {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)

	// A trivial RMW handler: count visits per key. The args blob IS the
	// key for routing purposes.
	_ = reg.RegisterRoutable("visit_inc", ops.OpReadWrite,
		func(tx *ops.TxContext, args []byte) ([]byte, error) {
			raw, err := tx.Get(args)
			var count uint64
			if err == nil && len(raw) == 8 {
				for i, b := range raw {
					count |= uint64(b) << (8 * i)
				}
			} else if err != nil && !errors.Is(err, cache.ErrNotFound) {
				return nil, err
			}
			count++
			buf := make([]byte, 8)
			for i := 0; i < 8; i++ {
				buf[i] = byte((count >> (8 * i)) & 0xff) //nolint:gosec // masked to one byte
			}
			return buf, tx.Put(args, buf, 0)
		},
		func(args []byte) ([]byte, bool) { return args, true },
	)

	fmt.Println("visit_inc registered")
	// Output: visit_inc registered
}
