// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/llmproxy"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/semcache"
)

// llmproxyDataAuto is the -data sentinel default, mirroring mcpDataAuto:
// "auto" resolves to ~/.rostam/llmcache (created if missing); an explicit
// -data "" selects heap/ephemeral mode instead. See mcpDataAuto for why the
// sentinel exists (a plain "" default would be indistinguishable from an
// operator explicitly asking for heap mode).
const llmproxyDataAuto = "auto"

// exactThreshold is the cosine-similarity hit floor used in "exact" mode (no
// hosted embedder configured): the stub embedder's vector for a given text is
// deterministic, so only a byte-identical prompt should ever score this high.
// It is not 1.0 to leave a hair of floating-point slack.
const exactThreshold = 0.999

// llmproxyFlags holds the parsed `llm-proxy` subcommand flags. It is passed
// to llmproxySetup as plain data so setup logic is unit-testable without
// touching flag.FlagSet or the real environment.
type llmproxyFlags struct {
	storeFlags
	listen     string
	upstream   string
	collection string
	threshold  float64
	maxTemp    float64
	ttl        time.Duration
	insecure   bool
}

// llmproxyRuntime is what llmproxySetup produces: a ready store, the cache
// built on top of it, and the HTTP handler runLlmProxyCmd serves.
type llmproxyRuntime struct {
	store   rostam.Store
	cache   *semcache.Cache
	handler http.Handler
	mode    string // "exact" | "semantic"
}

// runLlmProxyCmd implements `rostam-server llm-proxy`: an OpenAI-compatible
// caching reverse proxy backed by a rostam.Store semantic cache. Like
// runMcpCmd, it owns its own FlagSet and process exit.
func runLlmProxyCmd(args []string) {
	fs := flag.NewFlagSet("llm-proxy", flag.ExitOnError)
	var fl llmproxyFlags
	fs.StringVar(&fl.data, "data", llmproxyDataAuto,
		`data directory for embedded mode: "auto" (default) resolves to `+
			`~/.rostam/llmcache (created if missing); "" runs heap/ephemeral `+
			`(no persistence); any other value is used as given. Mutually `+
			`exclusive with -connect`)
	fs.StringVar(&fl.connect, "connect", "",
		"connect to a remote rostam-server instead of running embedded (host:port); mutually exclusive with -data")
	fs.StringVar(&fl.authToken, "auth-token", "",
		"bearer token for -connect (PREFER the ROSTAM_AUTH_TOKEN environment variable: a flag-passed secret is visible to other local users via /proc and shell history)")
	fs.StringVar(&fl.tlsCA, "tls-ca", "", "CA bundle PEM to verify the remote server's certificate (-connect)")
	fs.StringVar(&fl.tlsCert, "tls-cert", "", "client certificate PEM for mTLS (-connect; requires -tls-key)")
	fs.StringVar(&fl.tlsKey, "tls-key", "", "client private key PEM for mTLS (-connect; requires -tls-cert)")
	fs.StringVar(&fl.tlsServer, "tls-server-name", "", "expected server certificate name (SNI + verification) for -connect")
	fs.StringVar(&fl.listen, "listen", "127.0.0.1:8484", "HTTP listen address for the proxy")
	fs.StringVar(&fl.upstream, "upstream", "https://api.openai.com", "upstream OpenAI-compatible API base URL")
	fs.StringVar(&fl.collection, "collection", "llm-cache", "cache collection name (created if absent)")
	fs.Float64Var(&fl.threshold, "threshold", 0, "cosine-similarity hit floor (0 = mode default: 0.999 in exact mode, 0.97 in semantic mode)")
	fs.Float64Var(&fl.maxTemp, "max-temp", 1.0, "do not cache chat requests with a temperature above this")
	fs.DurationVar(&fl.ttl, "ttl", 168*time.Hour, "per-entry cache expiry")
	fs.BoolVar(&fl.insecure, "insecure", false, "acknowledge running with a non-loopback -listen (the proxy has no auth layer of its own); for dev/trusted-network use only")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}

	// Was -data given explicitly on the command line, as opposed to sitting
	// at its "auto" default? Same trick as runMcpCmd: llmproxySetup's
	// -data/-connect conflict check works on plain non-empty-string tests,
	// and "auto" is itself non-empty, so a caller who passed ONLY -connect
	// must not have -data resolved to a real (equally non-empty) path below.
	dataExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "data" {
			dataExplicit = true
		}
	})

	switch {
	case fl.connect != "" && !dataExplicit:
		// -data is still at its unset "auto" default; treat it as absent so
		// the store-selection logic below cleanly picks the -connect path.
		fl.data = ""
	case fl.data == llmproxyDataAuto:
		// Resolved HERE, not in llmproxySetup: llmproxySetup must stay pure
		// (table-driven-testable with no real filesystem or HOME dependency),
		// so home-dir resolution belongs in the subcommand entry point.
		home, err := os.UserHomeDir()
		if err != nil {
			fatal("llm-proxy: resolving home directory for -data", "err", err)
		}
		fl.data = filepath.Join(home, ".rostam", "llmcache")
	}

	// Claim the data dir before opening it. Only embedded mode with a real
	// directory needs this: heap mode (-data "") has nothing on disk to
	// share, and -connect's concurrency is the remote server's problem.
	unlock := func() error { return nil }
	if fl.connect == "" {
		var lerr error
		unlock, lerr = claimDataDir(fl.data)
		switch {
		case errors.Is(lerr, errDataDirBusy):
			fatal("llm-proxy: another rostam-server llm-proxy process is using this data directory; "+
				"a data dir has one writer, so concurrent proxies must share one server over -connect "+
				"instead of each embedding their own",
				"dir", fl.data, "err", lerr)
		case lerr != nil:
			fatal("llm-proxy: claiming -data directory", "dir", fl.data, "err", lerr)
		}
	}

	rt, err := llmproxySetup(fl, os.LookupEnv)
	if err != nil {
		_ = unlock()
		fatal("llm-proxy: setup failed", "err", err)
	}

	srv := &http.Server{
		Addr:    fl.listen,
		Handler: rt.handler,
		// ReadHeaderTimeout closes the classic slowloris window (a client that
		// dribbles request headers). ReadTimeout/WriteTimeout are deliberately
		// left at 0 (no limit): a streaming chat completion legitimately holds
		// the connection open for as long as the upstream keeps producing SSE
		// chunks, which can run well past any fixed request timeout.
		ReadHeaderTimeout: 10 * time.Second,
	}
	slog.Info("llm-proxy: serving", "addr", fl.listen, "upstream", fl.upstream, "mode", rt.mode, "collection", fl.collection)
	serveErr := srv.ListenAndServe()
	// Close before unlocking, same ordering as runMcpCmd: the store's own
	// shutdown may still write to the data dir, so releasing the lock first
	// would let a waiting process in mid-write.
	closeErr := rt.store.Close()
	_ = unlock()
	if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
		fatal("llm-proxy: serve failed", "err", serveErr)
	}
	if closeErr != nil {
		fatal("llm-proxy: store close failed", "err", closeErr)
	}
}

// llmproxySetup turns parsed flags + an environment lookup into a ready
// llmproxyRuntime. All validation (the upstream URL, the -data/-connect
// conflict, the embedder env combination, the exposed-listen gate) runs
// before any I/O, so a misconfiguration is reported without ever touching
// disk or dialing a remote server — mirrors mcpSetup.
func llmproxySetup(fl llmproxyFlags, lookupEnv func(string) (string, bool)) (llmproxyRuntime, error) {
	upstream, err := url.Parse(fl.upstream)
	if err != nil || !upstream.IsAbs() || (upstream.Scheme != "http" && upstream.Scheme != "https") {
		return llmproxyRuntime{}, fmt.Errorf("llm-proxy: -upstream must be an absolute http(s) URL, got %q", fl.upstream)
	}

	if fl.data != "" && fl.connect != "" {
		return llmproxyRuntime{}, errors.New("llm-proxy: use -data or -connect, not both")
	}

	embedder, err := embedderFromEnv(lookupEnv)
	if err != nil {
		return llmproxyRuntime{}, err
	}

	// exposedBind is the SAME loopback test the main server's own -insecure
	// gate uses (127.0.0.0/8, ::1, and the literal "localhost" are not
	// exposed; an unparseable address or anything else is treated as exposed
	// and refused without -insecure). Reusing it keeps the two gates from
	// silently drifting apart, and it never resolves DNS — a hostname other
	// than "localhost" is conservatively treated as exposed.
	if exposedBind(fl.listen) && !fl.insecure {
		return llmproxyRuntime{}, fmt.Errorf("llm-proxy: refusing to bind non-loopback -listen %q with no auth layer of its own; "+
			"bind a loopback address (127.0.0.1/localhost) or pass -insecure to run open deliberately (dev/trusted-network only)", fl.listen)
	}

	mode, threshold, embedder := llmproxyModeThreshold(embedder, fl.threshold)

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		return llmproxyRuntime{}, fmt.Errorf("llm-proxy: register ops: %w", err)
	}

	var store rostam.Store
	if fl.connect != "" {
		store, err = connectStore(fl.storeFlags, reg, lookupEnv)
	} else {
		// Same sizing precedent as mcpSetup (see its comment): this is one
		// process's cache, reached over ordinary HTTP concurrency, not a
		// multi-tenant server — 8 shards and a 256 MiB KV budget are ample
		// bookkeeping headroom without the server-sized defaults (256
		// shards, a host-RAM-fraction budget) turning a fresh -data dir into
		// hundreds of shard directories before a single entry is cached.
		store, err = rostam.NewDirect(rostam.DirectConfig{
			DataDir: fl.data,
			Ops:     reg,
			Cache:   rostam.CacheConfig{NumShardsPerNode: 8, MaxMemoryBytes: 256 << 20},
		})
	}
	if err != nil {
		return llmproxyRuntime{}, err
	}

	cache, err := semcache.New(context.Background(), semcache.Config{
		Store:      store,
		Embedder:   embedder,
		Collection: fl.collection,
		Threshold:  threshold,
		TTL:        fl.ttl,
		MaxTemp:    fl.maxTemp,
	})
	if err != nil {
		_ = store.Close()
		return llmproxyRuntime{}, fmt.Errorf("llm-proxy: cache init: %w", err)
	}

	proxy, err := llmproxy.NewServer(llmproxy.Config{
		Cache:    cache,
		Upstream: upstream,
		MaxTemp:  fl.maxTemp,
		Mode:     mode,
	})
	if err != nil {
		_ = store.Close()
		return llmproxyRuntime{}, fmt.Errorf("llm-proxy: proxy init: %w", err)
	}

	return llmproxyRuntime{store: store, cache: cache, handler: proxy.Handler(), mode: mode}, nil
}

// llmproxyModeThreshold resolves the cache mode, similarity threshold, and
// effective embedder from the (possibly nil) hosted embedder produced by
// embedderFromEnv and the -threshold flag. Split out from llmproxySetup so
// the defaulting rules are unit-testable without building a store:
//   - embedder == nil (no ROSTAM_EMBED_* env): "exact" mode with a
//     deterministic stub embedder; a zero flagThreshold defaults to
//     exactThreshold (0.999) rather than semcache's looser default, since a
//     stub embedder carries no real semantic meaning to compare on.
//   - embedder != nil: "semantic" mode using the hosted embedder as given; a
//     zero flagThreshold defaults to semcache.DefaultThreshold (0.97).
//   - a non-zero flagThreshold always wins, in either mode.
func llmproxyModeThreshold(embedder semcache.Embedder, flagThreshold float64) (mode string, threshold float64, resolved semcache.Embedder) {
	if embedder != nil {
		threshold = flagThreshold
		if threshold == 0 {
			threshold = semcache.DefaultThreshold
		}
		return "semantic", threshold, embedder
	}
	threshold = flagThreshold
	if threshold == 0 {
		threshold = exactThreshold
	}
	return "exact", threshold, semcache.NewStubEmbedder("exact", 64)
}
