// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/grpcapi/grpcsvc"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
	"github.com/rostamlabs/rostam/vector"
)

// tlsTestEnv bundles a running multi-transport TLS server, its op registry, the
// CA that signed its cert, and a scoped KeyRegistry, for the TLS/mTLS/cert-CN
// tests below.
type tlsTestEnv struct {
	srv    *rostam.Server
	ca     *testcerts.CA
	reg    *ops.Registry
	keyReg *vector.KeyRegistry
}

// newTLSServerEnv stands up a single store fronted by HTTP+gRPC+TCP, all wrapped
// in the SAME server *tls.Config (mTLS = RequireAndVerifyClientCert when
// requireClientCert is set), with the RBAC authorizer over a small scoped
// registry: token "k_read" → read:default/docs; token "k_admin" → *:*; cert CN
// "svcA" → read:default/docs (no token). A "docs" collection (dim 3) is created
// up front via an admin client so reads have something to hit.
func newTLSServerEnv(t *testing.T, requireClientCert bool) *tlsTestEnv {
	t.Helper()
	ca := testcerts.GenCA(t)
	sCert, sKey := ca.ServerCert(t, "localhost")
	serverTLS, err := tlsutil.ServerTLS(sCert, sKey, ca.CAFile, requireClientCert)
	if err != nil {
		t.Fatalf("ServerTLS: %v", err)
	}

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(dir, "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	// Cert-only principals still need a (unique, never-sent) token + tenant for the
	// registry's storage invariant — the token is a placeholder ("cn:<CN>") that no
	// client transmits; resolution happens by CertCN via LookupByCN.
	for _, k := range []vector.APIKey{
		{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}},
		{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}},
		{Token: "cn:admincert", CertCN: "admincert", Tenant: "t", Scopes: []string{"*:*"}},
		{Token: "cn:svcA", CertCN: "svcA", Tenant: "t", Scopes: []string{"read:default/docs"}},
	} {
		if err := keyReg.AddKey(k); err != nil {
			t.Fatalf("AddKey: %v", err)
		}
	}
	auth := authz.NewRBACAuthenticator(keyReg, reg, "")

	srv, err := rostam.NewServer(rostam.ServerConfig{
		DirectConfig: rostam.DirectConfig{
			DataDir:       dir,
			Ops:           reg,
			Cache:         rostam.CacheConfig{NumShardsPerNode: 4},
			Authenticator: auth,
		},
		HTTPAddr:  "127.0.0.1:0",
		GRPCAddr:  "127.0.0.1:0",
		TCPAddr:   "127.0.0.1:0",
		TLSConfig: serverTLS,
	})
	if err != nil {
		t.Fatalf("NewServer(TLS): %v", err)
	}
	t.Cleanup(func() { _ = srv.Close() })

	env := &tlsTestEnv{srv: srv, ca: ca, reg: reg, keyReg: keyReg}
	// Create the "docs" collection via an admin TCP client (mTLS cert "admincert"
	// when required, plus the admin token). Use a fresh admin client so the test
	// bodies start from a known collection.
	env.createDocsCollection(t, requireClientCert)
	return env
}

func (e *tlsTestEnv) createDocsCollection(t *testing.T, mTLS bool) {
	t.Helper()
	var clientTLS *tls.Config
	if mTLS {
		cCert, cKey := e.ca.ClientCert(t, "admincert")
		var err error
		clientTLS, err = tlsutil.ClientTLS(e.ca.CAFile, cCert, cKey, "localhost")
		if err != nil {
			t.Fatal(err)
		}
	} else {
		var err error
		clientTLS, err = tlsutil.ClientTLS(e.ca.CAFile, "", "", "localhost")
		if err != nil {
			t.Fatal(err)
		}
	}
	cli, err := client.New(client.Config{
		Servers:                 []string{e.srv.TCPAddr()},
		Ops:                     e.reg,
		TopologyRefreshInterval: time.Second,
		AuthToken:               "k_admin",
		TLSConfig:               clientTLS,
	})
	if err != nil {
		t.Fatalf("admin client.New: %v", err)
	}
	defer func() { _ = cli.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := cli.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs("docs", vector.Config{Dim: 3, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64})); err != nil {
		t.Fatalf("create docs collection: %v", err)
	}
}

// ---------------------------------------------------------------------------
// HTTP TLS
// ---------------------------------------------------------------------------

func httpsClient(caCfg *tls.Config) *http.Client {
	return &http.Client{
		Timeout:   10 * time.Second,
		Transport: &http.Transport{TLSClientConfig: caCfg},
	}
}

// TestTLSHTTPRightCAWorks: a client trusting the server CA + a valid bearer token
// can search docs over HTTPS; a client trusting a DIFFERENT CA fails the TLS
// handshake (not a 401 — a transport error).
func TestTLSHTTPRightCAWorks(t *testing.T) {
	env := newTLSServerEnv(t, false) // server-auth-only TLS (no client cert required)
	url := "https://" + env.srv.HTTPAddr()

	good, err := tlsutil.ClientTLS(env.ca.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	hc := httpsClient(good)
	body, _ := json.Marshal(map[string]any{"query": []float32{1, 2, 3}, "k": 1})
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points/search", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer k_read")
	resp, err := hc.Do(req)
	if err != nil {
		t.Fatalf("right-CA search over TLS: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("search status = %d, want 200 (%s)", resp.StatusCode, b)
	}

	// Wrong CA → handshake failure (a transport-level error, not an HTTP status).
	otherCA := testcerts.GenCA(t)
	bad, err := tlsutil.ClientTLS(otherCA.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	req2, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points/search", strings.NewReader(string(body)))
	req2.Header.Set("Authorization", "Bearer k_read")
	if _, err := httpsClient(bad).Do(req2); err == nil {
		t.Fatal("wrong-CA HTTPS request must FAIL at the TLS handshake")
	}
}

// TestTLSHTTPScopeEnforcedOverTLS: the read:docs token can search docs but is
// 401 on insert and on searching "other" — RBAC scope enforcement is identical
// over TLS.
func TestTLSHTTPScopeEnforcedOverTLS(t *testing.T) {
	env := newTLSServerEnv(t, false)
	url := "https://" + env.srv.HTTPAddr()
	good, err := tlsutil.ClientTLS(env.ca.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	hc := httpsClient(good)

	// Insert into docs with the read-only token → 401.
	ins, _ := json.Marshal(map[string]any{"id": 1, "vector": []float32{1, 2, 3}})
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points", strings.NewReader(string(ins)))
	req.Header.Set("Authorization", "Bearer k_read")
	resp, err := hc.Do(req)
	if err != nil {
		t.Fatalf("insert over TLS: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("read token insert docs: status = %d, want 401", resp.StatusCode)
	}

	// Search "other" with the docs-only token → 401.
	sb, _ := json.Marshal(map[string]any{"query": []float32{1, 2, 3}, "k": 1})
	req2, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/other/points/search", strings.NewReader(string(sb)))
	req2.Header.Set("Authorization", "Bearer k_read")
	resp2, err := hc.Do(req2)
	if err != nil {
		t.Fatalf("search other over TLS: %v", err)
	}
	_ = resp2.Body.Close()
	if resp2.StatusCode != http.StatusUnauthorized {
		t.Errorf("read:docs token search other: status = %d, want 401", resp2.StatusCode)
	}
}

// TestMTLSHTTPNoCertRejectedAtHandshake: with RequireAndVerifyClientCert, a
// client that presents NO client cert is rejected at the TLS handshake (a
// transport error), NOT served a 401.
func TestMTLSHTTPNoCertRejectedAtHandshake(t *testing.T) {
	env := newTLSServerEnv(t, true) // mTLS required
	url := "https://" + env.srv.HTTPAddr()

	// No client cert: trusts the server CA but presents nothing → handshake fails.
	noCert, err := tlsutil.ClientTLS(env.ca.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]any{"query": []float32{1, 2, 3}, "k": 1})
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points/search", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer k_read")
	if _, err := httpsClient(noCert).Do(req); err == nil {
		t.Fatal("mTLS server must REJECT a no-client-cert request at the handshake")
	}

	// A client WITH a CA-signed client cert + a valid token succeeds.
	cCert, cKey := env.ca.ClientCert(t, "admincert")
	withCert, err := tlsutil.ClientTLS(env.ca.CAFile, cCert, cKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	req2, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points/search", strings.NewReader(string(body)))
	req2.Header.Set("Authorization", "Bearer k_read")
	resp, err := httpsClient(withCert).Do(req2)
	if err != nil {
		t.Fatalf("mTLS client-cert search: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("mTLS client-cert search status = %d, want 200", resp.StatusCode)
	}
}

// TestMTLSHTTPCertCNPrincipal: a cert-only client (CN "svcA", NO bearer token)
// authorizes by its CN's scope (read:default/docs) — it can search docs but is
// 401 on insert. A cert whose CN has no registry entry is denied.
func TestMTLSHTTPCertCNPrincipal(t *testing.T) {
	env := newTLSServerEnv(t, true)
	url := "https://" + env.srv.HTTPAddr()

	svcCert, svcKey := env.ca.ClientCert(t, "svcA")
	svcCfg, err := tlsutil.ClientTLS(env.ca.CAFile, svcCert, svcKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	hc := httpsClient(svcCfg)

	// Search docs with NO token → authorized via cert CN svcA's read:docs scope.
	sb, _ := json.Marshal(map[string]any{"query": []float32{1, 2, 3}, "k": 1})
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points/search", strings.NewReader(string(sb)))
	resp, err := hc.Do(req)
	if err != nil {
		t.Fatalf("cert-CN search: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("cert-CN svcA search docs: status = %d, want 200 (CN scope read:docs)", resp.StatusCode)
	}

	// Insert into docs with the same cert (no token) → 401 (svcA has read only).
	ins, _ := json.Marshal(map[string]any{"id": 1, "vector": []float32{1, 2, 3}})
	req2, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points", strings.NewReader(string(ins)))
	resp2, err := hc.Do(req2)
	if err != nil {
		t.Fatalf("cert-CN insert: %v", err)
	}
	_ = resp2.Body.Close()
	if resp2.StatusCode != http.StatusUnauthorized {
		t.Errorf("cert-CN svcA insert docs: status = %d, want 401 (read-only CN)", resp2.StatusCode)
	}

	// A cert whose CN has NO registry entry → denied even for a read.
	unkCert, unkKey := env.ca.ClientCert(t, "stranger")
	unkCfg, err := tlsutil.ClientTLS(env.ca.CAFile, unkCert, unkKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	req3, _ := http.NewRequest(http.MethodPost, url+"/v1/collections/docs/points/search", strings.NewReader(string(sb)))
	resp3, err := httpsClient(unkCfg).Do(req3)
	if err != nil {
		t.Fatalf("unknown-CN search: %v", err)
	}
	_ = resp3.Body.Close()
	if resp3.StatusCode != http.StatusUnauthorized {
		t.Errorf("unknown cert CN: status = %d, want 401 (no registry entry)", resp3.StatusCode)
	}
}

// ---------------------------------------------------------------------------
// gRPC TLS / mTLS
// ---------------------------------------------------------------------------

func dialGRPCTLS(t *testing.T, addr string, cfg *tls.Config) (grpcsvc.VectorServiceClient, func()) {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(credentials.NewTLS(cfg)))
	if err != nil {
		t.Fatalf("grpc dial: %v", err)
	}
	return grpcsvc.NewVectorServiceClient(conn), func() { _ = conn.Close() }
}

// TestTLSGRPCWrongCAFails: a gRPC client trusting the wrong CA cannot complete a
// call (the handshake fails → an Unavailable/transport error, never a successful
// RPC).
func TestTLSGRPCWrongCAFails(t *testing.T) {
	env := newTLSServerEnv(t, false)
	otherCA := testcerts.GenCA(t)
	bad, err := tlsutil.ClientTLS(otherCA.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	cli, closeConn := dialGRPCTLS(t, env.srv.GRPCAddr(), bad)
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := cli.Health(ctx, &pb.HealthRequest{}); err == nil {
		t.Fatal("gRPC over wrong CA must fail (handshake)")
	}
}

// TestMTLSGRPCCertCNPrincipal: a cert-only gRPC client (CN svcA, no metadata
// token) authorizes a search via its CN scope but is Unauthenticated on upsert.
func TestMTLSGRPCCertCNPrincipal(t *testing.T) {
	env := newTLSServerEnv(t, true)

	svcCert, svcKey := env.ca.ClientCert(t, "svcA")
	svcCfg, err := tlsutil.ClientTLS(env.ca.CAFile, svcCert, svcKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	cli, closeConn := dialGRPCTLS(t, env.srv.GRPCAddr(), svcCfg)
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Search docs with no token → allowed by cert CN svcA's read:docs scope.
	if _, err := cli.Search(ctx, &pb.SearchRequest{Collection: "docs", K: 1, Query: []float32{1, 2, 3}}); err != nil {
		t.Fatalf("cert-CN gRPC search docs: %v", err)
	}
	// Upsert into docs with the same cert (no token) → Unauthenticated (read-only CN).
	_, err = cli.Upsert(ctx, &pb.UpsertRequest{Collection: "docs", Id: 1, Vector: []float32{1, 2, 3}})
	if status.Code(err) != codes.Unauthenticated {
		t.Errorf("cert-CN svcA upsert: code = %v, want Unauthenticated", status.Code(err))
	}

	// A cert with no registry CN → Unauthenticated even for a read.
	strCert, strKey := env.ca.ClientCert(t, "stranger")
	strCfg, err := tlsutil.ClientTLS(env.ca.CAFile, strCert, strKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	cli2, close2 := dialGRPCTLS(t, env.srv.GRPCAddr(), strCfg)
	defer close2()
	_, err = cli2.Search(ctx, &pb.SearchRequest{Collection: "docs", K: 1, Query: []float32{1, 2, 3}})
	if status.Code(err) != codes.Unauthenticated {
		t.Errorf("unknown cert CN gRPC search: code = %v, want Unauthenticated", status.Code(err))
	}
}

// ---------------------------------------------------------------------------
// TCP (Go client) TLS / mTLS
// ---------------------------------------------------------------------------

// TestMTLSTCPCertCNPrincipal: the Go client over mTLS with cert CN svcA (no
// AuthToken) can search docs but is denied an insert; a wrong-CA client cannot
// even dial.
func TestMTLSTCPCertCNPrincipal(t *testing.T) {
	env := newTLSServerEnv(t, true)

	svcCert, svcKey := env.ca.ClientCert(t, "svcA")
	svcCfg, err := tlsutil.ClientTLS(env.ca.CAFile, svcCert, svcKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	cli, err := client.New(client.Config{
		Servers:                 []string{env.srv.TCPAddr()},
		Ops:                     env.reg,
		TopologyRefreshInterval: time.Second,
		TLSConfig:               svcCfg, // NO AuthToken → cert CN is the principal
	})
	if err != nil {
		t.Fatalf("svcA client.New: %v", err)
	}
	defer func() { _ = cli.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if _, err := cli.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs("docs", 1, []float32{1, 2, 3})); err != nil {
		t.Fatalf("cert-CN TCP search docs: %v", err)
	}
	_, err = cli.Call(ctx, "vector_insert", ops.EncodeVectorInsertArgs("docs", 1, []float32{1, 2, 3}))
	if !errors.Is(err, client.ErrUnauthorized) {
		t.Errorf("cert-CN svcA insert: err = %v, want ErrUnauthorized", err)
	}

	// Wrong-CA client: cannot dial (TLS handshake at dial time fails).
	otherCA := testcerts.GenCA(t)
	badCfg, err := tlsutil.ClientTLS(otherCA.CAFile, svcCert, svcKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	badCli, err := client.New(client.Config{
		Servers:                 []string{env.srv.TCPAddr()},
		Ops:                     env.reg,
		TopologyRefreshInterval: time.Second,
		TLSConfig:               badCfg,
	})
	if err != nil {
		t.Fatalf("badCli New (lazy dial): %v", err)
	}
	defer func() { _ = badCli.Close() }()
	if _, err := badCli.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs("docs", 1, []float32{1, 2, 3})); err == nil {
		t.Fatal("wrong-CA TCP client must fail to connect")
	}
}

// TestMTLSTCPNoCertRejected: with mTLS required, a TLS client that presents NO
// client cert is rejected at the handshake (cannot complete an RPC).
func TestMTLSTCPNoCertRejected(t *testing.T) {
	env := newTLSServerEnv(t, true)
	noCertCfg, err := tlsutil.ClientTLS(env.ca.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	cli, err := client.New(client.Config{
		Servers:                 []string{env.srv.TCPAddr()},
		Ops:                     env.reg,
		TopologyRefreshInterval: time.Second,
		AuthToken:               "k_admin", // even WITH a valid token, the handshake blocks first
		TLSConfig:               noCertCfg,
	})
	if err != nil {
		t.Fatalf("noCert client.New: %v", err)
	}
	defer func() { _ = cli.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := cli.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs("docs", 1, []float32{1, 2, 3})); err == nil {
		t.Fatal("mTLS-required server must reject a no-client-cert TCP client at the handshake")
	}
}

// TestTLSTCPTokenWinsOverCert: a client presenting BOTH a cert (CN svcA →
// read-only) AND an admin bearer token authorizes by the TOKEN (token wins), so
// it can insert — proving the token-first precedence holds over TLS.
func TestTLSTCPTokenWinsOverCert(t *testing.T) {
	env := newTLSServerEnv(t, true)
	svcCert, svcKey := env.ca.ClientCert(t, "svcA")
	cfg, err := tlsutil.ClientTLS(env.ca.CAFile, svcCert, svcKey, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	cli, err := client.New(client.Config{
		Servers:                 []string{env.srv.TCPAddr()},
		Ops:                     env.reg,
		TopologyRefreshInterval: time.Second,
		AuthToken:               "k_admin", // admin token + svcA (read-only) cert → token wins
		TLSConfig:               cfg,
	})
	if err != nil {
		t.Fatalf("client.New: %v", err)
	}
	defer func() { _ = cli.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := cli.Call(ctx, "vector_insert", ops.EncodeVectorInsertArgs("docs", 7, []float32{1, 2, 3})); err != nil {
		t.Fatalf("admin-token insert (token must win over read-only cert): %v", err)
	}
}

// TestPlaintextUnchangedWhenTLSNil is the regression guard: a nil TLSConfig
// server + plaintext client behave exactly as before TLS existed.
func TestPlaintextUnchangedWhenTLSNil(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewServer(rostam.ServerConfig{
		DirectConfig: rostam.DirectConfig{
			DataDir: t.TempDir(),
			Ops:     reg,
			Cache:   rostam.CacheConfig{NumShardsPerNode: 4},
		},
		TCPAddr: "127.0.0.1:0",
		// TLSConfig nil ⇒ plaintext, no authenticator ⇒ open.
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = srv.Close() }()

	cli, err := client.New(client.Config{Servers: []string{srv.TCPAddr()}, Ops: reg, TopologyRefreshInterval: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cli.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := cli.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs("docs", vector.Config{Dim: 3, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64})); err != nil {
		t.Fatalf("plaintext create: %v", err)
	}
	if _, err := cli.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs("docs", 1, []float32{1, 2, 3})); err != nil {
		t.Fatalf("plaintext search: %v", err)
	}
}
