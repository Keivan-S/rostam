// SPDX-License-Identifier: Apache-2.0

package rostam

// GRPCServer is a Direct-backed store exposed over gRPC (see the grpcapi
// package for the service). It is a single-transport convenience over NewServer;
// for a store reachable over several transports at once (HTTP + gRPC + TCP
// sharing one cache) use NewServer with multiple addresses.
type GRPCServer struct{ inner *Server }

// NewGRPCServer constructs a Direct-backed cache and serves the gRPC API on
// addr. Use "127.0.0.1:0" for an OS-assigned port; read it back via Addr().
// cfg.Authenticator, if set, gates every RPC by the "authorization" metadata
// bearer token + op name (Health is exempt). Close stops the server and store.
func NewGRPCServer(addr string, cfg DirectConfig) (*GRPCServer, error) {
	inner, err := NewServer(ServerConfig{DirectConfig: cfg, GRPCAddr: addr})
	if err != nil {
		return nil, err
	}
	return &GRPCServer{inner: inner}, nil
}

// Addr returns the bound gRPC address (useful when addr was ":0").
func (s *GRPCServer) Addr() string { return s.inner.GRPCAddr() }

// Close gracefully stops the gRPC server and closes the underlying store.
// Idempotent.
func (s *GRPCServer) Close() error { return s.inner.Close() }
