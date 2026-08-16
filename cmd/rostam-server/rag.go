// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/rostamlabs/rostam/rag"
	"github.com/rostamlabs/rostam/semcache"
)

// ragDataDefault mirrors mcpDataAuto/llmproxyDataAuto in spirit, but rag has
// no "connect to the user's existing memory" precedent to default into —
// each corpus is its own thing, so the default is a plain relative directory
// rather than a $HOME-rooted sentinel.
const ragDataDefault = "./.rostam-rag"

// ragFlags holds the parsed `rag` subcommand flags, shared across all three
// sub-verbs (ingest/ask/query) so config resolution is written once.
type ragFlags struct {
	data         string
	endpoint     string
	corpus       string
	k            int
	chunkSize    int
	chunkOverlap int
	embedURL     string
	embedModel   string
	embedDim     int
	llmURL       string
	llmModel     string
	noHybrid     bool
	alpha        float64
}

// runRagCmd implements `rostam-server rag`: an ingest/ask/query CLI over the
// rag package. Like runMcpCmd/runLlmProxyCmd it owns its own FlagSet and
// process exit; runRagCmdE is the test seam that returns errors instead.
func runRagCmd(args []string) {
	if err := runRagCmdE(args); err != nil {
		fatal("rag: " + err.Error())
	}
}

// runRagCmdE is the sub-verb dispatcher. Split out from runRagCmd so tests
// can exercise the full ingest/ask/query flow without a subprocess or
// os.Exit — mirrors the mcpSetup/llmproxySetup test-seam pattern, but at the
// subcommand-dispatch level since rag's sub-verbs each do real I/O (there is
// no single "setup" step to isolate).
func runRagCmdE(args []string) error {
	if len(args) < 1 {
		return errors.New("usage: rostam-server rag <ingest|ask|query> [flags] <args>")
	}
	verb, rest := args[0], args[1:]

	fs := flag.NewFlagSet("rag "+verb, flag.ContinueOnError)
	var fl ragFlags
	fs.StringVar(&fl.data, "data", ragDataDefault, "embedded data directory (ignored when -endpoint is set)")
	fs.StringVar(&fl.endpoint, "endpoint", "", "talk to a running rostam server instead of the local --data dir (host:port); -endpoint takes precedence over -data")
	fs.StringVar(&fl.corpus, "corpus", "default", "corpus (collection) name")
	fs.IntVar(&fl.k, "k", 5, "number of hits to retrieve")
	fs.IntVar(&fl.chunkSize, "chunk-size", 0, "chunk size in words (0 = rag package default, 512)")
	fs.IntVar(&fl.chunkOverlap, "chunk-overlap", 0, "chunk overlap in words (0 = rag package default, 64)")
	fs.StringVar(&fl.embedURL, "embed-url", "", "embedding endpoint URL (overrides ROSTAM_EMBED_URL); needs -embed-model and -embed-dim too, else BM25")
	fs.StringVar(&fl.embedModel, "embed-model", "", "embedding model id (overrides ROSTAM_EMBED_MODEL)")
	fs.IntVar(&fl.embedDim, "embed-dim", 0, "embedding vector dimension (overrides ROSTAM_EMBED_DIM)")
	fs.StringVar(&fl.llmURL, "llm-url", "", "LLM chat-completions endpoint URL (overrides ROSTAM_LLM_URL)")
	fs.StringVar(&fl.llmModel, "llm-model", "", "LLM model id (overrides ROSTAM_LLM_MODEL)")
	fs.BoolVar(&fl.noHybrid, "no-hybrid", false, "disable dense+BM25 fusion; use pure dense KNN when an embedder is configured")
	fs.Float64Var(&fl.alpha, "alpha", -1, "weighted-fusion dense weight in [0,1]; unset (<0) uses RRF")
	if err := fs.Parse(rest); err != nil {
		return err
	}
	// Alpha validation lives in the rag package (Retrieve/Ask), so it covers
	// both the CLI and any library caller of WithHybrid — no need to duplicate
	// the NaN/>1 check here.

	resolveEnv(&fl, os.LookupEnv)
	embedKey, _ := os.LookupEnv("ROSTAM_EMBED_KEY")
	llmKey, _ := os.LookupEnv("ROSTAM_LLM_KEY")

	r, err := buildRetriever(fl)
	if err != nil {
		return err
	}
	defer func() { _ = r.Close() }()

	emb := buildEmbedder(fl, embedKey)
	ctx := context.Background()

	switch verb {
	case "ingest":
		return runRagIngest(ctx, r, emb, fl, fs.Args())
	case "query":
		return runRagQuery(ctx, r, emb, fl, fs.Args())
	case "ask":
		return runRagAsk(ctx, r, emb, fl, llmKey, fs.Args())
	default:
		return fmt.Errorf("unknown rag subcommand %q (want ingest, ask, or query)", verb)
	}
}

// resolveEnv fills any flag left at its zero value from the ROSTAM_EMBED_*/
// ROSTAM_LLM_* environment, per the env-first config rule: flags override
// env for every non-secret value, and API keys are read ONLY from
// ROSTAM_EMBED_KEY/ROSTAM_LLM_KEY (never a flag, to keep secrets out of
// /proc and shell history).
func resolveEnv(fl *ragFlags, lookupEnv func(string) (string, bool)) {
	if fl.embedURL == "" {
		if v, ok := lookupEnv("ROSTAM_EMBED_URL"); ok {
			fl.embedURL = v
		}
	}
	if fl.embedModel == "" {
		if v, ok := lookupEnv("ROSTAM_EMBED_MODEL"); ok {
			fl.embedModel = v
		}
	}
	if fl.embedDim == 0 {
		if v, ok := lookupEnv("ROSTAM_EMBED_DIM"); ok {
			if d, err := strconv.Atoi(v); err == nil {
				fl.embedDim = d
			}
		}
	}
	if fl.llmURL == "" {
		if v, ok := lookupEnv("ROSTAM_LLM_URL"); ok {
			fl.llmURL = v
		}
	}
	if fl.llmModel == "" {
		if v, ok := lookupEnv("ROSTAM_LLM_MODEL"); ok {
			fl.llmModel = v
		}
	}
}

// buildRetriever picks the Retriever backend: -endpoint (networked) wins
// over -data (embedded), mirroring storeFlags' -connect/-data precedent.
func buildRetriever(fl ragFlags) (rag.Retriever, error) {
	if fl.endpoint != "" {
		return rag.NewHTTPRetriever(fl.endpoint)
	}
	return rag.NewEmbeddedRetriever(fl.data)
}

// buildEmbedder constructs a hosted embedder only when URL+model+dim are ALL
// present; otherwise nil, which every rag package entry point (Ingest,
// Retrieve, Ask) already treats as "fall back to BM25". embedKey comes only
// from ROSTAM_EMBED_KEY (there is no -embed-key flag — see resolveEnv's doc
// comment), resolved by the caller so this stays a pure function of its
// arguments.
func buildEmbedder(fl ragFlags, embedKey string) semcache.Embedder {
	if fl.embedURL == "" || fl.embedModel == "" || fl.embedDim <= 0 {
		return nil
	}
	oe := semcache.NewOpenAIEmbedder(embedKey, fl.embedModel, fl.embedDim)
	oe.Endpoint = fl.embedURL
	return oe
}

// ragOptions builds the []rag.Option for Retrieve/Ask from the parsed CLI
// flags: hybrid fusion is on by default, unless -no-hybrid was passed.
func ragOptions(fl ragFlags) []rag.Option {
	if fl.noHybrid {
		return nil
	}
	return []rag.Option{rag.WithHybrid(fl.alpha)}
}

func embedderDim(emb semcache.Embedder) int {
	if emb == nil {
		return 0
	}
	return emb.Dim()
}

func runRagIngest(ctx context.Context, r rag.Retriever, emb semcache.Embedder, fl ragFlags, paths []string) error {
	if len(paths) == 0 {
		return errors.New("usage: rostam-server rag ingest [flags] <paths...>")
	}
	if err := r.EnsureCorpus(ctx, fl.corpus, embedderDim(emb)); err != nil {
		return fmt.Errorf("ensure corpus %q: %w", fl.corpus, err)
	}
	rep, err := rag.Ingest(ctx, r, paths, rag.IngestOptions{
		Corpus:       fl.corpus,
		ChunkSize:    fl.chunkSize,
		ChunkOverlap: fl.chunkOverlap,
		Embedder:     emb,
	})
	if err != nil {
		return fmt.Errorf("ingest: %w", err)
	}
	fmt.Printf("ingested %d file(s), %d chunk(s), skipped %d\n", rep.Files, rep.Chunks, rep.Skipped)
	for _, p := range rep.SkippedPaths {
		fmt.Printf("  skipped: %s\n", p)
	}
	return nil
}

func runRagQuery(ctx context.Context, r rag.Retriever, emb semcache.Embedder, fl ragFlags, args []string) error {
	if len(args) == 0 {
		return errors.New("usage: rostam-server rag query [flags] <text>")
	}
	query := args[0]
	hits, err := rag.Retrieve(ctx, r, emb, fl.corpus, query, fl.k, ragOptions(fl)...)
	if err != nil {
		return fmt.Errorf("query: %w", err)
	}
	printHits(hits)
	return nil
}

func runRagAsk(ctx context.Context, r rag.Retriever, emb semcache.Embedder, fl ragFlags, llmKey string, args []string) error {
	if len(args) == 0 {
		return errors.New("usage: rostam-server rag ask [flags] <question>")
	}
	question := args[0]
	llm := rag.LLMConfig{
		URL:        fl.llmURL,
		APIKey:     llmKey,
		Model:      fl.llmModel,
		HTTPClient: &http.Client{Timeout: 5 * time.Minute},
	}
	res, err := rag.Ask(ctx, r, emb, llm, fl.corpus, question, fl.k, ragOptions(fl)...)
	if err != nil {
		return fmt.Errorf("ask: %w", err)
	}
	fmt.Println(res.Answer)
	fmt.Println()
	fmt.Println("Sources:")
	for i, h := range res.Hits {
		fmt.Printf("  [%d] %s#%d\n", i+1, h.Source, h.Index)
	}
	return nil
}

// printHits renders query hits as `[n] source#index (score)` followed by an
// indented content excerpt, per the brief's output format.
func printHits(hits []rag.Hit) {
	for i, h := range hits {
		fmt.Printf("[%d] %s#%d (%.4f)\n", i+1, h.Source, h.Index, h.Score)
		fmt.Printf("   %s\n", excerpt(h.Content))
	}
}

// excerpt caps a hit's content to a single readable line for query output;
// ask's cited answer carries the full content instead (via the LLM prompt).
func excerpt(s string) string {
	const maxLen = 200
	// Collapse newlines so a multi-line chunk still prints as one indented
	// line rather than breaking the "[n] ..." / "   ..." two-line shape.
	r := []rune(s)
	clean := make([]rune, 0, len(r))
	for _, c := range r {
		if c == '\n' || c == '\r' {
			c = ' '
		}
		clean = append(clean, c)
	}
	if len(clean) > maxLen {
		return string(clean[:maxLen]) + "..."
	}
	return string(clean)
}
