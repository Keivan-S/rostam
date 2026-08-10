// SPDX-License-Identifier: Apache-2.0

package objstore

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// Credentials holds AWS access credentials. SessionToken is optional and only
// set for temporary (STS) credentials.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

// unsignedPayload is the magic content-sha256 value used to skip hashing the
// request body. Safe over HTTPS and lets us stream a snapshot PUT without
// buffering/hashing it whole.
const unsignedPayload = "UNSIGNED-PAYLOAD"

const (
	amzDateFmt = "20060102T150405Z" // ISO8601 basic, e.g. 20150830T123600Z
	dateFmt    = "20060102"
)

// hashSHA256 returns the lowercase hex SHA-256 of b.
func hashSHA256(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// hmacSHA256 computes HMAC-SHA256(key, data).
func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

// signV4 signs req in place per AWS Signature Version 4. It sets x-amz-date,
// x-amz-content-sha256 (= payloadHash), x-amz-security-token (if the session
// token is set) and the Authorization header.
//
// payloadHash is either a precomputed lowercase-hex SHA-256 of the body, or the
// literal UNSIGNED-PAYLOAD constant for streaming bodies over HTTPS.
//
// t is supplied explicitly (never time.Now inside) so signing is deterministic
// and testable against the published AWS test vectors.
func signV4(req *http.Request, payloadHash string, creds Credentials, region, service string, t time.Time) {
	t = t.UTC()
	amzDate := t.Format(amzDateFmt)
	dateStamp := t.Format(dateFmt)

	// AWS-required headers must be present before we build the canonical
	// headers / signed-headers list.
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	if creds.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", creds.SessionToken)
	}

	canonicalReq, signedHeaders := canonicalRequest(req, payloadHash)
	scope := dateStamp + "/" + region + "/" + service + "/aws4_request"
	stringToSign := stringToSign(amzDate, scope, canonicalReq)
	signingKey := signingKey(creds.SecretAccessKey, dateStamp, region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	auth := "AWS4-HMAC-SHA256 " +
		"Credential=" + creds.AccessKeyID + "/" + scope + ", " +
		"SignedHeaders=" + signedHeaders + ", " +
		"Signature=" + signature
	req.Header.Set("Authorization", auth)
}

// canonicalRequest builds the AWS canonical request string and returns it along
// with the semicolon-joined signed-headers list.
//
//	METHOD\nCanonicalURI\nCanonicalQuery\nCanonicalHeaders\n\nSignedHeaders\nHashedPayload
func canonicalRequest(req *http.Request, payloadHash string) (canonical, signedHeaders string) {
	canonHeaders, signedHeaders := canonicalHeaders(req)
	parts := []string{
		req.Method,
		canonicalURI(req.URL),
		canonicalQuery(req.URL),
		canonHeaders,
		signedHeaders,
		payloadHash,
	}
	return strings.Join(parts, "\n"), signedHeaders
}

// canonicalURI returns the URI-encoded path. Per the AWS spec each path segment
// is RFC3986-encoded but the segment separators ("/") are preserved (not
// double-encoded). We rely on url.URL.EscapedPath, which already produces the
// RFC3986 path with slashes intact and the path normalized. An empty path
// canonicalizes to "/".
func canonicalURI(u *url.URL) string {
	path := u.EscapedPath()
	if path == "" {
		return "/"
	}
	// EscapedPath gives the RFC3986 path with segment separators ("/")
	// preserved and each segment percent-encoded, which is exactly the S3
	// canonical-URI rule (S3 does NOT normalize away "." / ".." or collapse
	// "//", and we rely on callers passing already-literal keys).
	return path
}

// canonicalQuery returns the canonical query string: parameters sorted by key
// (and value), each key and value URI-encoded per RFC3986.
//
// Decoding uses url.PathUnescape (NOT url.QueryUnescape) deliberately: a literal
// '+' in RawQuery is a real plus sign on the wire, but QueryUnescape would decode
// it to a space, so the signer would canonicalize a different string than the URL
// actually carries — yielding a 403 signature mismatch for any caller that splices
// a base64 token or a '+'-bearing value into RawQuery unencoded. PathUnescape
// leaves '+' intact, so the canonical form matches the wire regardless of how the
// caller built the query. (%2B still decodes to '+' and re-encodes to %2B.)
func canonicalQuery(u *url.URL) string {
	if u.RawQuery == "" {
		return ""
	}
	// Parse manually rather than url.Values to preserve duplicate keys and to
	// control encoding exactly.
	type kv struct{ k, v string }
	var pairs []kv
	for _, p := range strings.Split(u.RawQuery, "&") {
		if p == "" {
			continue
		}
		k, v, _ := strings.Cut(p, "=")
		dk, _ := url.PathUnescape(k)
		dv, _ := url.PathUnescape(v)
		pairs = append(pairs, kv{rfc3986Escape(dk), rfc3986Escape(dv)})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].k != pairs[j].k {
			return pairs[i].k < pairs[j].k
		}
		return pairs[i].v < pairs[j].v
	})
	var b strings.Builder
	for i, p := range pairs {
		if i > 0 {
			b.WriteByte('&')
		}
		b.WriteString(p.k)
		b.WriteByte('=')
		b.WriteString(p.v)
	}
	return b.String()
}

// canonicalHeaders returns the canonical headers block and the signed-headers
// list. Header names are lowercased and sorted; values are trimmed and
// sequential internal whitespace is collapsed. host, x-amz-date and
// x-amz-content-sha256 are always included when present.
func canonicalHeaders(req *http.Request) (canonical, signed string) {
	// Collect header name -> joined values.
	headers := map[string][]string{}
	for name, vals := range req.Header {
		lower := strings.ToLower(name)
		// Skip headers that must not be signed for our use; we sign everything
		// present, which is valid (and required for x-amz-*). Authorization is
		// not yet set at signing time.
		headers[lower] = append(headers[lower], vals...)
	}
	// host is not part of req.Header; it comes from req.Host / req.URL.Host.
	host := req.Host
	if host == "" {
		host = req.URL.Host
	}
	headers["host"] = []string{host}

	names := make([]string, 0, len(headers))
	for n := range headers {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, n := range names {
		b.WriteString(n)
		b.WriteByte(':')
		// Trim each value and collapse internal runs of spaces, then join
		// multiple values with commas (AWS rule).
		vv := make([]string, len(headers[n]))
		for i, v := range headers[n] {
			vv[i] = collapseSpaces(strings.TrimSpace(v))
		}
		b.WriteString(strings.Join(vv, ","))
		b.WriteByte('\n')
	}
	return b.String(), strings.Join(names, ";")
}

// collapseSpaces replaces runs of internal whitespace with a single space, per
// the SigV4 canonical-header value rule.
func collapseSpaces(s string) string {
	var b strings.Builder
	var prevSpace bool
	for _, r := range s {
		if r == ' ' || r == '\t' {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	return b.String()
}

// stringToSign builds the SigV4 string-to-sign.
func stringToSign(amzDate, scope, canonicalReq string) string {
	return strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzDate,
		scope,
		hashSHA256([]byte(canonicalReq)),
	}, "\n")
}

// signingKey derives the SigV4 signing key via the HMAC chain.
func signingKey(secret, dateStamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	return hmacSHA256(kService, []byte("aws4_request"))
}

// rfc3986Escape percent-encodes s per RFC3986, leaving unreserved characters
// (A-Z a-z 0-9 - _ . ~) intact. This matches AWS's expectation for query
// component encoding (notably it encodes "/" and " " as %2F and %20, unlike
// url.QueryEscape which turns space into "+").
func rfc3986Escape(s string) string {
	const upperhex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if isUnreserved(c) {
			b.WriteByte(c)
		} else {
			b.WriteByte('%')
			b.WriteByte(upperhex[c>>4])
			b.WriteByte(upperhex[c&0x0f])
		}
	}
	return b.String()
}

func isUnreserved(c byte) bool {
	switch {
	case c >= 'A' && c <= 'Z':
		return true
	case c >= 'a' && c <= 'z':
		return true
	case c >= '0' && c <= '9':
		return true
	case c == '-' || c == '_' || c == '.' || c == '~':
		return true
	}
	return false
}
