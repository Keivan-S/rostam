// SPDX-License-Identifier: Apache-2.0
//go:build localembed

// Package local implements the -tags localembed local ONNX embedder: model
// download/verify/cache, tokenization, and (in later tasks) inference.
package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

// Ensure returns local paths to the model's onnx + vocab, downloading and
// SHA-256-verifying any missing artifact. Concurrent processes are serialized
// by a lock on the model directory. A generous client timeout should be set by
// the caller (models are ~90-130 MB).
func Ensure(ctx context.Context, spec localcatalog.ModelSpec, root string, hc *http.Client) (string, string, error) {
	dir := filepath.Join(root, spec.Name)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", "", err
	}
	unlock, err := lockDir(dir)
	if err != nil {
		return "", "", err
	}
	defer func() { _ = unlock() }()

	onnxPath := filepath.Join(dir, "model.onnx")
	vocabPath := filepath.Join(dir, "vocab.txt")
	if err := fetchVerify(ctx, hc, spec.OnnxURL, spec.OnnxSHA, onnxPath); err != nil {
		return "", "", fmt.Errorf("model %q onnx: %w", spec.Name, err)
	}
	if err := fetchVerify(ctx, hc, spec.VocabURL, spec.VocabSHA, vocabPath); err != nil {
		return "", "", fmt.Errorf("model %q vocab: %w", spec.Name, err)
	}
	return onnxPath, vocabPath, nil
}

// fetchVerify downloads url to dest if dest is not already present, verifying
// the download's SHA-256 against wantSHA before installing it. A checksum
// mismatch removes the temp file and returns an error; dest is never left
// holding unverified content.
func fetchVerify(ctx context.Context, hc *http.Client, url, wantSHA, dest string) error {
	if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
		return nil // already cached (verified when first installed)
	}
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := hc.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	tmp, err := os.CreateTemp(filepath.Dir(dest), ".dl-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	h := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(tmp, h), resp.Body)
	closeErr := tmp.Close()
	if copyErr != nil {
		_ = os.Remove(tmpName)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmpName)
		return closeErr
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != wantSHA {
		_ = os.Remove(tmpName)
		return fmt.Errorf("checksum mismatch for %s: got %s want %s", url, got, wantSHA)
	}
	return os.Rename(tmpName, dest)
}
