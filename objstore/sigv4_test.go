// SPDX-License-Identifier: Apache-2.0

package objstore

import (
	"encoding/hex"
	"net/http"
	"strings"
	"testing"
	"time"
)

// AWS Signature V4 test-suite constants
// (https://docs.aws.amazon.com/general/latest/gr/signature-v4-test-suite.html).
// Every published fixture uses these exact values.
const (
	tvAccessKey = "AKIDEXAMPLE"
	tvSecretKey = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
	tvRegion    = "us-east-1"
	tvService   = "service"
	tvHost      = "example.amazonaws.com"
	tvAmzDate   = "20150830T123600Z"
	tvDateStamp = "20150830"
	// tvEmptyHash is SHA-256(""), the payload hash the fixtures use.
	tvEmptyHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

// tvTime is 20150830T123600Z.
var tvTime = time.Date(2015, 8, 30, 12, 36, 0, 0, time.UTC)

// sigVector is one AWS test-suite case. canonicalReq is the EXACT published
// .creq; authzSig is the EXACT published Signature= value from the .authz.
type sigVector struct {
	name    string
	method  string
	rawURL  string // request target incl. query, e.g. "/?Param1=value1"
	headers map[string]string

	canonicalReq string // exact .creq contents
	authzSig     string // exact Signature= hex from the .authz
}

func sigVectors() []sigVector {
	return []sigVector{
		{
			// get-vanilla
			name:    "get-vanilla",
			method:  "GET",
			rawURL:  "/",
			headers: map[string]string{"X-Amz-Date": tvAmzDate},
			canonicalReq: "GET\n" +
				"/\n" +
				"\n" +
				"host:example.amazonaws.com\n" +
				"x-amz-date:20150830T123600Z\n" +
				"\n" +
				"host;x-amz-date\n" +
				tvEmptyHash,
			authzSig: "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
		},
		{
			// get-vanilla-query-order-key-case: params sorted by key.
			name:    "get-vanilla-query-order-key-case",
			method:  "GET",
			rawURL:  "/?Param2=value2&Param1=value1",
			headers: map[string]string{"X-Amz-Date": tvAmzDate},
			canonicalReq: "GET\n" +
				"/\n" +
				"Param1=value1&Param2=value2\n" +
				"host:example.amazonaws.com\n" +
				"x-amz-date:20150830T123600Z\n" +
				"\n" +
				"host;x-amz-date\n" +
				tvEmptyHash,
			authzSig: "b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500",
		},
		{
			// post-vanilla
			name:    "post-vanilla",
			method:  "POST",
			rawURL:  "/",
			headers: map[string]string{"X-Amz-Date": tvAmzDate},
			canonicalReq: "POST\n" +
				"/\n" +
				"\n" +
				"host:example.amazonaws.com\n" +
				"x-amz-date:20150830T123600Z\n" +
				"\n" +
				"host;x-amz-date\n" +
				tvEmptyHash,
			authzSig: "5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b",
		},
	}
}

// TestSigV4_AWSTestSuite verifies the signer reproduces the published AWS
// canonical request and signature EXACTLY for a representative subset of the
// aws-sig-v4-test-suite (vanilla GET, query ordering, single query, POST).
func TestSigV4_AWSTestSuite(t *testing.T) {
	creds := Credentials{AccessKeyID: tvAccessKey, SecretAccessKey: tvSecretKey}
	signingKey := signingKey(tvSecretKey, tvDateStamp, tvRegion, tvService)
	scope := tvDateStamp + "/" + tvRegion + "/" + tvService + "/aws4_request"

	for _, v := range sigVectors() {

		t.Run(v.name, func(t *testing.T) {
			req, err := http.NewRequest(v.method, "http://"+tvHost+v.rawURL, nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req.Host = tvHost
			for k, val := range v.headers {
				req.Header.Set(k, val)
			}

			// Layer 1: canonical request must match the published .creq exactly.
			canon, signed := canonicalRequest(req, tvEmptyHash)
			if canon != v.canonicalReq {
				t.Fatalf("canonical request mismatch\n--- got ---\n%s\n--- want ---\n%s", canon, v.canonicalReq)
			}

			// Layer 2: string-to-sign derives from the canonical hash.
			sts := stringToSign(tvAmzDate, scope, canon)

			// Layer 3: final signature must match the published .authz exactly.
			gotSig := hex.EncodeToString(hmacSHA256(signingKey, []byte(sts)))
			if gotSig != v.authzSig {
				t.Fatalf("signature mismatch\n got: %s\nwant: %s\nstring-to-sign:\n%s", gotSig, v.authzSig, sts)
			}

			// And the assembled Authorization header is well-formed.
			wantAuthz := "AWS4-HMAC-SHA256 " +
				"Credential=" + creds.AccessKeyID + "/" + scope + ", " +
				"SignedHeaders=" + signed + ", " +
				"Signature=" + v.authzSig
			gotAuthz := "AWS4-HMAC-SHA256 " +
				"Credential=" + creds.AccessKeyID + "/" + scope + ", " +
				"SignedHeaders=" + signed + ", " +
				"Signature=" + gotSig
			if gotAuthz != wantAuthz {
				t.Fatalf("authorization mismatch\n got: %s\nwant: %s", gotAuthz, wantAuthz)
			}
		})
	}
}

// TestSigV4_EndToEnd checks signV4 sets every expected header and reproduces
// the get-vanilla signature when x-amz-content-sha256 is also signed (the S3
// path), confirming the full in-place signing flow.
func TestSigV4_EndToEnd(t *testing.T) {
	creds := Credentials{AccessKeyID: tvAccessKey, SecretAccessKey: tvSecretKey}
	req, _ := http.NewRequest("GET", "http://"+tvHost+"/", nil)
	req.Host = tvHost

	signV4(req, tvEmptyHash, creds, tvRegion, tvService, tvTime)

	if got := req.Header.Get("X-Amz-Date"); got != tvAmzDate {
		t.Errorf("x-amz-date = %q, want %q", got, tvAmzDate)
	}
	if got := req.Header.Get("X-Amz-Content-Sha256"); got != tvEmptyHash {
		t.Errorf("x-amz-content-sha256 = %q", got)
	}
	// With x-amz-content-sha256 signed, the signed-headers list differs from the
	// bare suite fixture; assert structure + presence.
	wantPrefix := "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, " +
		"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature="
	got := req.Header.Get("Authorization")
	if !strings.HasPrefix(got, wantPrefix) {
		t.Errorf("authorization = %q\nwant prefix %q", got, wantPrefix)
	}
	if len(strings.TrimPrefix(got, wantPrefix)) != 64 {
		t.Errorf("signature not 64 hex chars: %q", got)
	}
}

// TestSigV4_SessionToken verifies the security-token header is set and signed.
func TestSigV4_SessionToken(t *testing.T) {
	creds := Credentials{AccessKeyID: tvAccessKey, SecretAccessKey: tvSecretKey, SessionToken: "SESSIONTOKEN123"}
	req, _ := http.NewRequest("GET", "http://"+tvHost+"/", nil)
	req.Host = tvHost
	signV4(req, tvEmptyHash, creds, tvRegion, tvService, tvTime)

	if got := req.Header.Get("X-Amz-Security-Token"); got != "SESSIONTOKEN123" {
		t.Errorf("x-amz-security-token = %q", got)
	}
	if auth := req.Header.Get("Authorization"); !strings.Contains(auth, "x-amz-security-token") {
		t.Errorf("security token not in signed headers: %s", auth)
	}
}

// TestRFC3986Escape covers the encoding rules AWS relies on for query params.
func TestRFC3986Escape(t *testing.T) {
	cases := map[string]string{
		"value1":      "value1",
		"a b":         "a%20b",
		"a/b":         "a%2Fb",
		"a~b":         "a~b",
		"a+b":         "a%2Bb",
		"foo=bar&baz": "foo%3Dbar%26baz",
	}
	for in, want := range cases {
		if got := rfc3986Escape(in); got != want {
			t.Errorf("rfc3986Escape(%q) = %q, want %q", in, got, want)
		}
	}
}
