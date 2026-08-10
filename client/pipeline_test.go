// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestPipelinedCallParity: with pipelining on, a simple put/get round-trips
// correctly (the response goes to the right caller).
func TestPipelinedCallParity(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := New(Config{Servers: []string{addr}, PipelineDepth: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	if _, err := c.Call(ctx, "put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("pipelined put: %v", err)
	}
	res, err := c.Call(ctx, "get", ops.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("pipelined get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("get = %q, want v", res)
	}
}

// TestPipelinedResponsesMatchCallersUnderLoad is the correctness core: many
// concurrent callers share a FEW pipelined connections (so lots of requests are
// in flight per conn at once). Each writes a distinct key→value then reads it
// back and checks it got ITS OWN value. If FIFO response-matching were wrong,
// callers would receive each other's responses and the values would mismatch.
func TestPipelinedResponsesMatchCallersUnderLoad(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	// Deep pipeline, only 2 conns → heavy multiplexing of many in-flight ops.
	c, err := New(Config{Servers: []string{addr}, PipelineDepth: 64, PipelineConns: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	const workers, perWorker = 24, 40
	var wg sync.WaitGroup
	errCh := make(chan error, workers*perWorker)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			ctx := context.Background()
			for i := 0; i < perWorker; i++ {
				key := []byte(fmt.Sprintf("w%d-k%d", w, i))
				val := []byte(fmt.Sprintf("w%d-v%d", w, i))
				if _, err := c.Call(ctx, "put", ops.EncodePutArgs(key, val, 0)); err != nil {
					errCh <- fmt.Errorf("put %s: %w", key, err)
					return
				}
				got, err := c.Call(ctx, "get", ops.EncodeKeyArgs(key))
				if err != nil {
					errCh <- fmt.Errorf("get %s: %w", key, err)
					return
				}
				if !bytes.Equal(got, val) {
					errCh <- fmt.Errorf("key %s: got %q, want %q — pipelined response matched the WRONG caller", key, got, val)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errCh)
	for e := range errCh {
		t.Fatal(e)
	}
}

// TestPipelinedNotFoundAndRemoteError: the pipelined path maps non-OK statuses
// (NotFound, RemoteError) the same as the pooled path.
func TestPipelinedNotFoundAndRemoteError(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := New(Config{Servers: []string{addr}, PipelineDepth: 16})
	defer func() { _ = c.Close() }()

	if _, err := c.Call(context.Background(), "get", ops.EncodeKeyArgs([]byte("absent"))); err != ErrNotFound {
		t.Fatalf("missing key err = %v, want ErrNotFound", err)
	}
	if _, err := c.Call(context.Background(), "no_such_op", nil); err == nil {
		t.Fatal("unknown op: want error")
	}
}
