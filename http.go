// SPDX-License-Identifier: Apache-2.0

package rostam

// HTTPServer is a Direct-backed store exposed over a REST/JSON HTTP API (see the
// httpapi package for the route surface). It is a single-transport convenience
// over NewServer; for a store reachable over several transports at once (HTTP +
// gRPC + TCP sharing one cache) use NewServer with multiple addresses.
type HTTPServer struct{ inner *Server }

// NewHTTPServer constructs a Direct-backed cache and serves the REST API on
// addr. Use "127.0.0.1:0" for an OS-assigned port; read it back via Addr().
// cfg.Authenticator, if set, gates every request by bearer token + op name
// (health is exempt). Close stops the server and the underlying store.
func NewHTTPServer(addr string, cfg DirectConfig) (*HTTPServer, error) {
	inner, err := NewServer(ServerConfig{DirectConfig: cfg, HTTPAddr: addr})
	if err != nil {
		return nil, err
	}
	return &HTTPServer{inner: inner}, nil
}

// Addr returns the bound HTTP address (useful when addr was ":0").
func (s *HTTPServer) Addr() string { return s.inner.HTTPAddr() }

// Close shuts down the HTTP server and the underlying Direct store. Idempotent.
func (s *HTTPServer) Close() error { return s.inner.Close() }
