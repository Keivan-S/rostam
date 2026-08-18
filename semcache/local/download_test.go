// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

func sha256hex(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func TestEnsureDownloadsVerifiesAndReuses(t *testing.T) {
	onnx := []byte("fake-onnx-bytes")
	vocab := []byte("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n")
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if r.URL.Path == "/model.onnx" {
			_, _ = w.Write(onnx)
		} else {
			_, _ = w.Write(vocab)
		}
	}))
	defer srv.Close()

	spec := localcatalog.ModelSpec{
		Name: "test", OnnxURL: srv.URL + "/model.onnx", OnnxSHA: sha256hex(onnx),
		VocabURL: srv.URL + "/vocab.txt", VocabSHA: sha256hex(vocab), Dim: 4,
	}
	root := t.TempDir()

	op, vp, err := Ensure(context.Background(), spec, root, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(op); string(b) != string(onnx) {
		t.Fatal("onnx content mismatch")
	}
	if b, _ := os.ReadFile(vp); string(b) != string(vocab) {
		t.Fatal("vocab content mismatch")
	}
	firstHits := hits

	// Second call: files cached, zero new HTTP hits.
	if _, _, err := Ensure(context.Background(), spec, root, srv.Client()); err != nil {
		t.Fatal(err)
	}
	if hits != firstHits {
		t.Fatalf("expected cache reuse, got %d extra hits", hits-firstHits)
	}
}

func TestEnsureRejectsBadChecksum(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("tampered"))
	}))
	defer srv.Close()
	spec := localcatalog.ModelSpec{
		Name: "bad", OnnxURL: srv.URL + "/model.onnx", OnnxSHA: sha256hex([]byte("expected")),
		VocabURL: srv.URL + "/vocab.txt", VocabSHA: sha256hex([]byte("expected")), Dim: 4,
	}
	root := t.TempDir()
	if _, _, err := Ensure(context.Background(), spec, root, srv.Client()); err == nil {
		t.Fatal("want checksum error, got nil")
	}
	// The tampered temp file must not be left behind as model.onnx.
	if _, statErr := os.Stat(root + "/bad/model.onnx"); statErr == nil {
		t.Fatal("tampered file was installed")
	}
}
