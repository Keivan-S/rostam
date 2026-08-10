// SPDX-License-Identifier: Apache-2.0

package objstore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func testCreds() Credentials {
	return Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret"}
}

// newTestStore builds an S3Store pointed at srv with a fixed signing clock.
func newTestStore(t *testing.T, srv *httptest.Server, pathStyle bool, creds Credentials) *S3Store {
	t.Helper()
	store, err := NewS3Store(Config{
		Endpoint:   srv.URL,
		Region:     "us-east-1",
		Bucket:     "mybucket",
		Creds:      creds,
		PathStyle:  pathStyle,
		HTTPClient: srv.Client(),
		clock:      func() time.Time { return tvTime },
	})
	if err != nil {
		t.Fatalf("NewS3Store: %v", err)
	}
	return store
}

func assertSigned(t *testing.T, r *http.Request) {
	t.Helper()
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/") {
		t.Errorf("missing/malformed Authorization: %q", auth)
	}
	if !strings.Contains(auth, "SignedHeaders=") || !strings.Contains(auth, "Signature=") {
		t.Errorf("Authorization missing components: %q", auth)
	}
	if r.Header.Get("X-Amz-Date") == "" {
		t.Errorf("missing X-Amz-Date")
	}
	if r.Header.Get("X-Amz-Content-Sha256") == "" {
		t.Errorf("missing X-Amz-Content-Sha256")
	}
}

func TestS3_PutGetRoundTrip_PathStyle(t *testing.T) {
	var stored []byte
	var putPath string
	var putContentSha string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r)
		switch r.Method {
		case http.MethodPut:
			putPath = r.URL.Path
			putContentSha = r.Header.Get("X-Amz-Content-Sha256")
			b, _ := io.ReadAll(r.Body)
			stored = b
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			w.WriteHeader(http.StatusOK)
			w.Write(stored)
		default:
			t.Errorf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	store := newTestStore(t, srv, true, testCreds())
	ctx := context.Background()

	body := []byte("snapshot-bytes")
	if err := store.Put(ctx, "t/c/snap.snap", bytes.NewReader(body), int64(len(body))); err != nil {
		t.Fatalf("put: %v", err)
	}
	if putPath != "/mybucket/t/c/snap.snap" {
		t.Errorf("path-style PUT path = %q, want /mybucket/t/c/snap.snap", putPath)
	}
	// The test server is http:// (no TLS), so the PUT must sign the REAL SHA-256 of
	// the body, NOT UNSIGNED-PAYLOAD — UNSIGNED-PAYLOAD over plaintext would leave
	// the body unprotected (see Put: UNSIGNED-PAYLOAD is gated on https).
	wantSha := hashSHA256(body)
	if putContentSha != wantSha {
		t.Errorf("PUT x-amz-content-sha256 = %q, want real body hash %s (http endpoint must not use UNSIGNED-PAYLOAD)", putContentSha, wantSha)
	}

	rc, err := store.Get(ctx, "t/c/snap.snap")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	got, _ := io.ReadAll(rc)
	rc.Close()
	if !bytes.Equal(got, body) {
		t.Errorf("get returned %q, want %q", got, body)
	}
}

func TestS3_VirtualHostURL(t *testing.T) {
	var sawHost string
	var sawPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawHost = r.Host
		sawPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	}))
	defer srv.Close()

	// For virtual-host we must still actually reach the test server; the
	// bucket-subdomain host won't resolve, so we point the HTTP client's
	// transport at the test server via a rewriting client.
	endpointURL, _ := url.Parse(srv.URL)
	rewriting := &http.Client{Transport: rewriteTransport{target: endpointURL, base: srv.Client().Transport}}

	store, err := NewS3Store(Config{
		Endpoint:   srv.URL,
		Region:     "us-east-1",
		Bucket:     "mybucket",
		Creds:      testCreds(),
		PathStyle:  false,
		HTTPClient: rewriting,
		clock:      func() time.Time { return tvTime },
	})
	if err != nil {
		t.Fatalf("NewS3Store: %v", err)
	}

	rc, err := store.Get(context.Background(), "key.snap")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	rc.Close()

	if !strings.HasPrefix(sawHost, "mybucket.") {
		t.Errorf("virtual-host Host = %q, want mybucket.* prefix", sawHost)
	}
	if sawPath != "/key.snap" {
		t.Errorf("virtual-host path = %q, want /key.snap", sawPath)
	}
}

// rewriteTransport sends every request to target regardless of the request URL
// host, while preserving the original Host header (so virtual-host addressing
// can be exercised against a localhost test server).
type rewriteTransport struct {
	target *url.URL
	base   http.RoundTripper
}

func (rt rewriteTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r.URL.Scheme = rt.target.Scheme
	r.URL.Host = rt.target.Host
	return rt.base.RoundTrip(r)
}

func TestS3_Get404IsNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`<?xml version="1.0"?><Error><Code>NoSuchKey</Code><Message>nope</Message></Error>`))
	}))
	defer srv.Close()

	store := newTestStore(t, srv, true, testCreds())
	if _, err := store.Get(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get 404: got %v, want ErrNotFound", err)
	}
}

func TestS3_GetErrorXML(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(`<?xml version="1.0"?><Error><Code>AccessDenied</Code><Message>denied</Message></Error>`))
	}))
	defer srv.Close()

	store := newTestStore(t, srv, true, testCreds())
	_, err := store.Get(context.Background(), "k")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "AccessDenied") || !strings.Contains(err.Error(), "denied") {
		t.Errorf("error should include S3 code/message: %v", err)
	}
}

func TestS3_Delete(t *testing.T) {
	var deleted bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r)
		if r.Method == http.MethodDelete {
			deleted = true
			w.WriteHeader(http.StatusNoContent)
			return
		}
		t.Errorf("unexpected method %s", r.Method)
	}))
	defer srv.Close()

	store := newTestStore(t, srv, true, testCreds())
	if err := store.Delete(context.Background(), "k"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if !deleted {
		t.Error("server never saw DELETE")
	}
}

func TestS3_Delete404IsNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	store := newTestStore(t, srv, true, testCreds())
	if err := store.Delete(context.Background(), "k"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Delete 404: got %v, want ErrNotFound", err)
	}
}

func TestS3_ListPagination(t *testing.T) {
	// Two pages: first IsTruncated with a continuation token, second final.
	var reqCount int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r)
		if r.URL.Query().Get("list-type") != "2" {
			t.Errorf("missing list-type=2: %s", r.URL.RawQuery)
		}
		if r.URL.Query().Get("prefix") != "t/c/" {
			t.Errorf("prefix = %q, want t/c/", r.URL.Query().Get("prefix"))
		}
		token := r.URL.Query().Get("continuation-token")
		reqCount++
		w.Header().Set("Content-Type", "application/xml")
		w.WriteHeader(http.StatusOK)
		switch token {
		case "":
			fmt.Fprint(w, `<?xml version="1.0"?>
<ListBucketResult>
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>TOKEN2</NextContinuationToken>
  <Contents><Key>t/c/1.snap</Key><Size>10</Size><LastModified>2026-01-01T00:00:00.000Z</LastModified></Contents>
  <Contents><Key>t/c/2.snap</Key><Size>20</Size><LastModified>2026-01-02T00:00:00.000Z</LastModified></Contents>
</ListBucketResult>`)
		case "TOKEN2":
			fmt.Fprint(w, `<?xml version="1.0"?>
<ListBucketResult>
  <IsTruncated>false</IsTruncated>
  <Contents><Key>t/c/3.snap</Key><Size>30</Size><LastModified>2026-01-03T00:00:00.000Z</LastModified></Contents>
</ListBucketResult>`)
		default:
			t.Errorf("unexpected continuation-token %q", token)
		}
	}))
	defer srv.Close()

	store := newTestStore(t, srv, true, testCreds())
	got, err := store.List(context.Background(), "t/c/")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if reqCount != 2 {
		t.Errorf("expected 2 paged requests, got %d", reqCount)
	}
	want := []struct {
		key  string
		size int64
	}{{"t/c/1.snap", 10}, {"t/c/2.snap", 20}, {"t/c/3.snap", 30}}
	if len(got) != len(want) {
		t.Fatalf("got %d objects, want %d: %+v", len(got), len(want), got)
	}
	for i, w := range want {
		if got[i].Key != w.key || got[i].Size != w.size {
			t.Errorf("obj[%d] = {%q,%d}, want {%q,%d}", i, got[i].Key, got[i].Size, w.key, w.size)
		}
		if got[i].LastModified.IsZero() {
			t.Errorf("obj[%d] LastModified not parsed", i)
		}
	}
}

func TestS3_SessionTokenHeader(t *testing.T) {
	var sawToken string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawToken = r.Header.Get("X-Amz-Security-Token")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	creds := Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret", SessionToken: "TOKEN-XYZ"}
	store := newTestStore(t, srv, true, creds)
	rc, err := store.Get(context.Background(), "k")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	rc.Close()
	if sawToken != "TOKEN-XYZ" {
		t.Errorf("X-Amz-Security-Token = %q, want TOKEN-XYZ", sawToken)
	}
}

func TestS3_DefaultEndpointAndEnvCreds(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "ENVKEY")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "ENVSECRET")
	t.Setenv("AWS_SESSION_TOKEN", "ENVTOKEN")

	store, err := NewS3Store(Config{Region: "eu-west-1", Bucket: "bkt"})
	if err != nil {
		t.Fatalf("NewS3Store: %v", err)
	}
	if store.endpoint.String() != "https://s3.eu-west-1.amazonaws.com" {
		t.Errorf("default endpoint = %q", store.endpoint.String())
	}
	if store.creds.AccessKeyID != "ENVKEY" || store.creds.SecretAccessKey != "ENVSECRET" || store.creds.SessionToken != "ENVTOKEN" {
		t.Errorf("env creds not loaded: %+v", store.creds)
	}
}
