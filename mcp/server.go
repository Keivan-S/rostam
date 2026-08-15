// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/internal/buildinfo"
	"github.com/rostamlabs/rostam/semcache"
)

const protocolVersion = "2025-06-18"

// Config configures a Server.
type Config struct {
	Store       rostam.Store      // required: embedded or remote engine
	Embedder    semcache.Embedder // nil => BM25-only mode (stub vectors, text recall)
	Destructive bool              // register delete/delete_by_filter for arbitrary collections
	Logger      *slog.Logger      // stderr logger; nil => slog.Default()
}

type toolDef struct {
	Name        string
	Description string
	InputSchema map[string]any
	Handler     func(ctx context.Context, args json.RawMessage) (any, error)
}

// Server is one stdio MCP session over a rostam.Store.
type Server struct {
	store       rostam.Store
	emb         semcache.Embedder
	hybrid      bool // real embedder configured: recall uses dense+BM25 fusion
	destructive bool
	log         *slog.Logger
	tools       []toolDef
	initialized bool

	// Lazy mcp_memory bootstrap. A mutex plus a succeeded-flag rather than a
	// sync.Once: Once would latch a transient first failure for the lifetime
	// of the process, and only success may be latched here. See ensureMemory.
	memMu    sync.Mutex
	memReady bool

	// dispatchMu is held for the whole of one tool call, and by Shutdown. It
	// is what lets a caller close the Store safely while Serve is still parked
	// on its input: see Shutdown.
	dispatchMu sync.Mutex
	stopped    bool
}

func NewServer(ctx context.Context, cfg Config) (*Server, error) {
	if cfg.Store == nil {
		return nil, errors.New("mcp: Config.Store is required")
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	s := &Server{store: cfg.Store, destructive: cfg.Destructive, log: log}
	if cfg.Embedder != nil {
		s.emb, s.hybrid = cfg.Embedder, true
	} else {
		s.emb = semcache.NewStubEmbedder("stub", 64)
	}
	if err := s.checkEmbedderIdentity(ctx); err != nil {
		return nil, err
	}
	s.registerMemoryTools()
	s.registerDBTools()
	if s.destructive {
		s.registerDestructiveTools()
	}
	return s, nil
}

func (s *Server) register(t toolDef) { s.tools = append(s.tools, t) }

func textResult(v any) map[string]any {
	b, err := json.Marshal(v)
	if err != nil { // handler values are our own structs; this is a programming error
		return errResult(fmt.Errorf("mcp: marshal result: %w", err))
	}
	return map[string]any{"content": []map[string]any{{"type": "text", "text": string(b)}}}
}

func errResult(err error) map[string]any {
	return map[string]any{
		"content": []map[string]any{{"type": "text", "text": err.Error()}},
		"isError": true,
	}
}

// Serve runs the session until EOF on r. The caller owns the Store's
// lifecycle; Serve never closes it.
func (s *Server) Serve(r io.Reader, w io.Writer) error {
	ctx := context.Background()
	c := newConn(r, w)
	for {
		req, err := c.next()
		switch {
		case errors.Is(err, io.EOF):
			return nil
		case errors.Is(err, errLine):
			if werr := c.replyError(json.RawMessage("null"), codeParse, err.Error()); werr != nil {
				return werr
			}
			continue
		case err != nil:
			return err
		}
		// Envelope validation runs BEFORE the notification check: an object
		// with a broken envelope cannot be trusted to be a notification just
		// because its id did not parse, so it is answered (with a null id
		// where none was usable) rather than silently swallowed.
		if verr := req.validate(); verr != nil {
			if werr := c.replyError(errorID(req.ID), codeInvalidRequest, "invalid request: "+verr.Error()); werr != nil {
				return werr
			}
			continue
		}
		if req.ID == nil { // notification: never answered
			continue
		}
		s.dispatchMu.Lock()
		if s.stopped {
			s.dispatchMu.Unlock()
			return nil // Shutdown: the Store may already be closed
		}
		err = s.dispatch(ctx, c, req)
		s.dispatchMu.Unlock()
		if err != nil {
			return err // write failure: the client is gone
		}
	}
}

// Shutdown stops the session from handling any further request and waits for
// the one in flight, if any, to return. It is safe to call from another
// goroutine, and safe to call more than once.
//
// It exists so a caller can close the Store on a signal. Serve reads from a
// blocking stdin, and there is no portable way to interrupt that read from
// outside — closing the file does not reliably unblock it (the fd is not
// registered with the runtime poller) and would race the in-flight read
// against fd reuse. So Shutdown does not try to make Serve return. It
// guarantees the weaker but sufficient thing: after it returns, no handler is
// running and none will start, which is exactly the condition Store.Close
// needs. Serve's goroutine may stay parked on a read that never completes; in
// the signal path the process is exiting anyway.
func (s *Server) Shutdown() {
	s.dispatchMu.Lock()
	defer s.dispatchMu.Unlock()
	s.stopped = true
}

func (s *Server) dispatch(ctx context.Context, c *conn, req *request) error {
	switch req.Method {
	case "initialize":
		s.initialized = true
		return c.reply(req.ID, map[string]any{
			"protocolVersion": protocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			// Derived, not written down: a literal here goes stale at the
			// next release and the client shows a version that never existed.
			"serverInfo": map[string]any{"name": "rostam", "version": buildinfo.Version()},
		})
	case "ping":
		return c.reply(req.ID, map[string]any{})
	}
	if !s.initialized {
		return c.replyError(req.ID, codeInvalidRequest, "server not initialized")
	}
	switch req.Method {
	case "tools/list":
		list := make([]map[string]any, len(s.tools))
		for i, t := range s.tools {
			list[i] = map[string]any{"name": t.Name, "description": t.Description, "inputSchema": t.InputSchema}
		}
		return c.reply(req.ID, map[string]any{"tools": list})
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return c.replyError(req.ID, codeInvalidParams, "bad tools/call params: "+err.Error())
		}
		for _, t := range s.tools {
			if t.Name == p.Name {
				v, err := t.Handler(ctx, p.Arguments)
				if err != nil {
					s.log.Warn("tool failed", "tool", t.Name, "err", err)
					return c.reply(req.ID, errResult(err))
				}
				return c.reply(req.ID, textResult(v))
			}
		}
		return c.replyError(req.ID, codeInvalidParams, "unknown tool: "+p.Name)
	default:
		return c.replyError(req.ID, codeMethodNotFound, "unknown method: "+req.Method)
	}
}
