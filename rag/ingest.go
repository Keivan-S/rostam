// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"fmt"
	"hash/fnv"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/rostamlabs/rostam/semcache"
)

// IngestOptions configures a call to Ingest.
type IngestOptions struct {
	Corpus       string
	ChunkSize    int               // words; default 512
	ChunkOverlap int               // words; default 64
	Embedder     semcache.Embedder // nil => BM25-only
}

// IngestReport summarizes the outcome of an Ingest call.
type IngestReport struct {
	Files        int
	Chunks       int
	Skipped      int
	SkippedPaths []string
}

var textExts = map[string]bool{
	".txt": true, ".md": true, ".markdown": true, ".go": true, ".py": true,
	".js": true, ".ts": true, ".rs": true, ".java": true, ".c": true, ".h": true,
	".cpp": true, ".json": true, ".yaml": true, ".yml": true, ".toml": true,
}

// Ingest walks paths (files or directories, recursively), extracts UTF-8 text
// from recognized extensions, chunks each file, optionally embeds the chunks,
// and upserts them into the given corpus. Re-ingesting the same source path
// is idempotent: prior chunks for that source are deleted before the fresh
// ones are written.
func Ingest(ctx context.Context, r Retriever, paths []string, opts IngestOptions) (IngestReport, error) {
	if opts.Corpus == "" {
		opts.Corpus = "default"
	}
	if opts.ChunkSize <= 0 {
		opts.ChunkSize = 512
	}
	if opts.ChunkOverlap <= 0 {
		opts.ChunkOverlap = 64
	}
	var rep IngestReport
	for _, p := range paths {
		err := filepath.WalkDir(p, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			if !textExts[strings.ToLower(filepath.Ext(path))] {
				rep.Skipped++
				rep.SkippedPaths = append(rep.SkippedPaths, path)
				return nil
			}
			body, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if !utf8.Valid(body) {
				rep.Skipped++
				rep.SkippedPaths = append(rep.SkippedPaths, path)
				return nil
			}
			n, err := ingestFile(ctx, r, opts, path, string(body))
			if err != nil {
				return err
			}
			rep.Files++
			rep.Chunks += n
			return nil
		})
		if err != nil {
			return rep, err
		}
	}
	return rep, nil
}

func ingestFile(ctx context.Context, r Retriever, opts IngestOptions, source, body string) (int, error) {
	chunks := SplitText(body, opts.ChunkSize, opts.ChunkOverlap)
	// Idempotent re-ingest: purge this file's previous chunks first, even if
	// the file now yields zero chunks (e.g. it was edited down to empty) —
	// otherwise stale chunks from an earlier ingest orphan in the corpus.
	if _, err := r.DeleteBySource(ctx, opts.Corpus, source); err != nil {
		return 0, err
	}
	if len(chunks) == 0 {
		return 0, nil
	}
	var vecs [][]float32
	if opts.Embedder != nil {
		texts := make([]string, len(chunks))
		for i, c := range chunks {
			texts[i] = c.Content
		}
		var err error
		vecs, err = opts.Embedder.Embed(ctx, texts)
		if err != nil {
			return 0, fmt.Errorf("rag: embed %s: %w", source, err)
		}
	}
	stored := make([]StoredChunk, len(chunks))
	for i, c := range chunks {
		sc := StoredChunk{
			ID:      chunkID(source, c.Index),
			Content: c.Content,
			Source:  source,
			Index:   c.Index,
		}
		if vecs != nil {
			sc.Vector = vecs[i]
		}
		stored[i] = sc
	}
	if err := r.Upsert(ctx, opts.Corpus, stored); err != nil {
		return 0, err
	}
	return len(stored), nil
}

// chunkID is deterministic in (source, index) so re-ingest overwrites in place.
func chunkID(source string, index int) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(source))
	_, _ = h.Write([]byte("#"))
	_, _ = h.Write([]byte(strconv.Itoa(index)))
	id := h.Sum64()
	if id == 0 {
		id = 1 // avoid id 0 (historically special-cased in the engine)
	}
	return id
}
