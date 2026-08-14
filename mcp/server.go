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

	memOnce sync.Once // lazy mcp_memory bootstrap
	memErr  error
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
	return s, nil
}

// checkEmbedderIdentity is completed in the memory task; until the memory
// collection exists there is nothing to validate.
func (s *Server) checkEmbedderIdentity(ctx context.Context) error { return nil }

// Placeholder registrars completed by later tasks.
func (s *Server) registerMemoryTools() {}
func (s *Server) registerDBTools()     {}

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
		if req.ID == nil { // notification: never answered
			continue
		}
		if err := s.dispatch(ctx, c, req); err != nil {
			return err // write failure: the client is gone
		}
	}
}

func (s *Server) dispatch(ctx context.Context, c *conn, req *request) error {
	switch req.Method {
	case "initialize":
		s.initialized = true
		return c.reply(req.ID, map[string]any{
			"protocolVersion": protocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": "rostam", "version": "0.1.0"},
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
