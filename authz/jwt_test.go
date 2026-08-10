// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/hmac"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"testing"
	"time"
)

// --- test helpers: mint keys, PEM-encode public keys, and sign JWTs ----------

func mustRSAKey(t *testing.T) *rsa.PrivateKey {
	t.Helper()
	k, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("rsa keygen: %v", err)
	}
	return k
}

func mustECKey(t *testing.T) *ecdsa.PrivateKey {
	t.Helper()
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("ec keygen: %v", err)
	}
	return k
}

// pubPEM PKIX-encodes a public key to PEM (the format NewJWTVerifier consumes).
func pubPEM(t *testing.T, pub any) []byte {
	t.Helper()
	der, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		t.Fatalf("marshal pub: %v", err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: der})
}

func b64(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }

// signingInputFor builds "<b64 header>.<b64 payload>" for the given alg+claims.
func signingInputFor(t *testing.T, alg string, claims map[string]any) (string, []byte) {
	t.Helper()
	hdr := map[string]any{"alg": alg, "typ": "JWT"}
	hj, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	pj, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	si := b64(hj) + "." + b64(pj)
	return si, []byte(si)
}

// mintRS256 signs a valid RS256 JWT with key.
func mintRS256(t *testing.T, key *rsa.PrivateKey, claims map[string]any) string {
	t.Helper()
	si, raw := signingInputFor(t, "RS256", claims)
	d := sha256.Sum256(raw)
	sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, d[:])
	if err != nil {
		t.Fatalf("rsa sign: %v", err)
	}
	return si + "." + b64(sig)
}

// mintES256 signs a valid ES256 JWT (raw r||s, 32 bytes each).
func mintES256(t *testing.T, key *ecdsa.PrivateKey, claims map[string]any) string {
	t.Helper()
	si, raw := signingInputFor(t, "ES256", claims)
	d := sha256.Sum256(raw)
	r, s, err := ecdsa.Sign(rand.Reader, key, d[:])
	if err != nil {
		t.Fatalf("ecdsa sign: %v", err)
	}
	sig := make([]byte, 64)
	r.FillBytes(sig[:32])
	s.FillBytes(sig[32:])
	return si + "." + b64(sig)
}

// validClaims returns a fresh, fully-valid claim set; mutate as needed.
func validClaims() map[string]any {
	return map[string]any{
		"sub":    "user-1",
		"tenant": "acme",
		"scopes": "read:default/docs write:default/*",
		"exp":    time.Now().Add(time.Hour).Unix(),
	}
}

func newRS256Verifier(t *testing.T, key *rsa.PrivateKey, iss, aud string) *JWTVerifier {
	t.Helper()
	v, err := NewJWTVerifier(pubPEM(t, &key.PublicKey), iss, aud)
	if err != nil {
		t.Fatalf("NewJWTVerifier: %v", err)
	}
	if v.alg != algRS256 {
		t.Fatalf("expected pinned RS256, got %q", v.alg)
	}
	return v
}

func newES256Verifier(t *testing.T, key *ecdsa.PrivateKey, iss, aud string) *JWTVerifier {
	t.Helper()
	v, err := NewJWTVerifier(pubPEM(t, &key.PublicKey), iss, aud)
	if err != nil {
		t.Fatalf("NewJWTVerifier: %v", err)
	}
	if v.alg != algES256 {
		t.Fatalf("expected pinned ES256, got %q", v.alg)
	}
	return v
}

// --- happy paths -------------------------------------------------------------

func TestVerify_ValidRS256_SpaceScopes(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	tok := mintRS256(t, key, validClaims())

	c, err := v.VerifyAndExtract(tok)
	if err != nil {
		t.Fatalf("expected accept, got %v", err)
	}
	if c.Sub != "user-1" || c.Tenant != "acme" {
		t.Fatalf("bad claims: %+v", c)
	}
	if len(c.Scopes) != 2 || c.Scopes[0] != "read:default/docs" || c.Scopes[1] != "write:default/*" {
		t.Fatalf("bad scopes: %v", c.Scopes)
	}
}

func TestVerify_ValidRS256_ArrayScopes(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	claims["scopes"] = []string{"read:default/docs", "*:*"}
	tok := mintRS256(t, key, claims)

	c, err := v.VerifyAndExtract(tok)
	if err != nil {
		t.Fatalf("expected accept, got %v", err)
	}
	if len(c.Scopes) != 2 || c.Scopes[0] != "read:default/docs" || c.Scopes[1] != "*:*" {
		t.Fatalf("bad array scopes: %v", c.Scopes)
	}
}

func TestVerify_ValidES256(t *testing.T) {
	key := mustECKey(t)
	v := newES256Verifier(t, key, "", "")
	tok := mintES256(t, key, validClaims())

	c, err := v.VerifyAndExtract(tok)
	if err != nil {
		t.Fatalf("expected accept, got %v", err)
	}
	if c.Tenant != "acme" {
		t.Fatalf("bad tenant: %q", c.Tenant)
	}
}

// --- ALG-CONFUSION defenses (the security proof) -----------------------------

func TestVerify_AlgNone_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	// Craft an alg:"none" token with no signature ("x.y.").
	si, _ := signingInputFor(t, "none", validClaims())
	tok := si + "." // empty signature segment

	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("alg:none MUST be rejected")
	}
}

func TestVerify_AlgNone_WithNonEmptySig_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	si, _ := signingInputFor(t, "none", validClaims())
	tok := si + "." + b64([]byte("anything"))
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("alg:none MUST be rejected even with a junk sig")
	}
}

// The classic alg-confusion attack: present an HS256 token whose HMAC secret is
// the RSA public key bytes. Must be rejected purely on the alg-pin (HS256 !=
// pinned RS256), never even reaching an HMAC verify.
func TestVerify_HS256WithRSAPubKey_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	pemBytes := pubPEM(t, &key.PublicKey)
	v, err := NewJWTVerifier(pemBytes, "", "")
	if err != nil {
		t.Fatalf("NewJWTVerifier: %v", err)
	}
	// Forge an HS256 token signing with HMAC-SHA256 over the public key PEM.
	si, raw := signingInputFor(t, "HS256", validClaims())
	mac := hmac.New(sha256.New, pemBytes)
	mac.Write(raw)
	tok := si + "." + b64(mac.Sum(nil))

	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("HS256-with-RSA-pubkey alg-confusion MUST be rejected")
	}
}

func TestVerify_AlgMismatch_ES256AgainstRS256Verifier_Rejected(t *testing.T) {
	rsaKey := mustRSAKey(t)
	ecKey := mustECKey(t)
	v := newRS256Verifier(t, rsaKey, "", "") // pinned RS256
	tok := mintES256(t, ecKey, validClaims())
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("ES256 token against RS256 verifier MUST be rejected")
	}
}

// --- signature integrity -----------------------------------------------------

func TestVerify_WrongKey_Rejected(t *testing.T) {
	signKey := mustRSAKey(t)
	otherKey := mustRSAKey(t)
	v := newRS256Verifier(t, otherKey, "", "") // verifier holds a DIFFERENT key
	tok := mintRS256(t, signKey, validClaims())
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("wrong-key signature MUST be rejected")
	}
}

func TestVerify_WrongKey_ES256_Rejected(t *testing.T) {
	signKey := mustECKey(t)
	otherKey := mustECKey(t)
	v := newES256Verifier(t, otherKey, "", "")
	tok := mintES256(t, signKey, validClaims())
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("wrong-key ES256 signature MUST be rejected")
	}
}

func TestVerify_TamperedPayload_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	tok := mintRS256(t, key, validClaims())

	// Replace the payload segment with a different (valid-JSON) payload while
	// keeping the original header + signature -> signature no longer matches.
	h, _, s, ok := split3(tok)
	if !ok {
		t.Fatal("test token not 3 segments")
	}
	evil := validClaims()
	evil["tenant"] = "attacker"
	pj, _ := json.Marshal(evil)
	tampered := h + "." + b64(pj) + "." + s
	if _, err := v.VerifyAndExtract(tampered); err == nil {
		t.Fatal("tampered payload MUST be rejected")
	}
}

// --- temporal claims ---------------------------------------------------------

func TestVerify_Expired_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	claims["exp"] = time.Now().Add(-time.Hour).Unix()
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("expired token MUST be rejected")
	}
}

func TestVerify_NbfFuture_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	claims["nbf"] = time.Now().Add(time.Hour).Unix()
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("nbf-in-future token MUST be rejected")
	}
}

// --- iss / aud ---------------------------------------------------------------

func TestVerify_IssMismatch_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "https://issuer.example", "")
	claims := validClaims()
	claims["iss"] = "https://evil.example"
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("issuer mismatch MUST be rejected")
	}
}

func TestVerify_IssMatch_Accepted(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "https://issuer.example", "")
	claims := validClaims()
	claims["iss"] = "https://issuer.example"
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err != nil {
		t.Fatalf("matching issuer MUST be accepted, got %v", err)
	}
}

func TestVerify_AudMismatch_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "rostam")
	claims := validClaims()
	claims["aud"] = "other-service"
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("audience mismatch MUST be rejected")
	}
}

func TestVerify_AudArrayContains_Accepted(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "rostam")
	claims := validClaims()
	claims["aud"] = []string{"other-service", "rostam", "third"}
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err != nil {
		t.Fatalf("aud array containing the audience MUST be accepted, got %v", err)
	}
}

func TestVerify_AudStringMatch_Accepted(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "rostam")
	claims := validClaims()
	claims["aud"] = "rostam"
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err != nil {
		t.Fatalf("matching aud string MUST be accepted, got %v", err)
	}
}

// --- required claims ---------------------------------------------------------

func TestVerify_MissingExp_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	delete(claims, "exp")
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("missing exp MUST be rejected")
	}
}

func TestVerify_MissingTenant_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	delete(claims, "tenant")
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("missing tenant MUST be rejected")
	}
}

func TestVerify_EmptyTenant_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	claims["tenant"] = ""
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("empty tenant MUST be rejected")
	}
}

func TestVerify_MissingScopes_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	delete(claims, "scopes")
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("missing scopes MUST be rejected")
	}
}

func TestVerify_EmptyScopesString_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	claims["scopes"] = "   "
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("empty scopes string MUST be rejected")
	}
}

func TestVerify_EmptyScopesArray_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	claims := validClaims()
	claims["scopes"] = []string{}
	tok := mintRS256(t, key, claims)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("empty scopes array MUST be rejected")
	}
}

// --- malformed tokens (no panic) ---------------------------------------------

func TestVerify_TwoSegments_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	if _, err := v.VerifyAndExtract("aaa.bbb"); err == nil {
		t.Fatal("2-segment token MUST be rejected")
	}
}

func TestVerify_NonBase64url_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	if _, err := v.VerifyAndExtract("!!!.@@@.###"); err == nil {
		t.Fatal("non-base64url token MUST be rejected")
	}
}

func TestVerify_NonJSONHeader_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	tok := b64([]byte("not json")) + "." + b64([]byte(`{}`)) + "." + b64([]byte("sig"))
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("non-JSON header MUST be rejected")
	}
}

func TestVerify_NonJSONPayload_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	// Valid header + signature over a non-JSON payload.
	hj, _ := json.Marshal(map[string]any{"alg": "RS256", "typ": "JWT"})
	payload := []byte("not json")
	si := b64(hj) + "." + b64(payload)
	d := sha256.Sum256([]byte(si))
	sig, _ := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, d[:])
	tok := si + "." + b64(sig)
	if _, err := v.VerifyAndExtract(tok); err == nil {
		t.Fatal("non-JSON payload MUST be rejected")
	}
}

func TestVerify_EmptySegments_Rejected(t *testing.T) {
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	if _, err := v.VerifyAndExtract(".."); err == nil {
		t.Fatal("empty segments MUST be rejected")
	}
}

// --- NewJWTVerifier construction ---------------------------------------------

func TestNewJWTVerifier_BadPEM_Error(t *testing.T) {
	if _, err := NewJWTVerifier([]byte("not a pem"), "", ""); err == nil {
		t.Fatal("bad PEM MUST error")
	}
}

func TestNewJWTVerifier_UnsupportedKeyType_Error(t *testing.T) {
	// An Ed25519 public key is a valid PKIX key but unsupported here.
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("ed25519 keygen: %v", err)
	}
	der, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		t.Fatalf("marshal ed25519: %v", err)
	}
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: der})
	if _, err := NewJWTVerifier(pemBytes, "", ""); err == nil {
		t.Fatal("unsupported (Ed25519) key type MUST error")
	}
}

func TestNewJWTVerifier_UnsupportedCurve_Error(t *testing.T) {
	// P-384 ECDSA key -> unsupported curve (only P-256/ES256).
	k, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		t.Fatalf("p384 keygen: %v", err)
	}
	pemBytes := pubPEM(t, &k.PublicKey)
	if _, err := NewJWTVerifier(pemBytes, "", ""); err == nil {
		t.Fatal("unsupported curve (P-384) MUST error")
	}
}

// --- looksLikeJWT ------------------------------------------------------------

func TestLooksLikeJWT(t *testing.T) {
	cases := []struct {
		tok  string
		want bool
	}{
		{"a.b.c", true},
		{"eyJ.payload.sig", true},
		{"a.b", false},
		{"abc", false},
		{"a.b.c.d", false},
		{"..", false},
		{"", false},
	}
	for _, c := range cases {
		if got := looksLikeJWT(c.tok); got != c.want {
			t.Errorf("looksLikeJWT(%q) = %v, want %v", c.tok, got, c.want)
		}
	}
}

// --- aud as big.Int sanity (ensures r||s split width is correct) -------------

func TestES256_SigWidth(t *testing.T) {
	// Sanity: a freshly minted ES256 sig is 64 bytes and verifies.
	key := mustECKey(t)
	v := newES256Verifier(t, key, "", "")
	tok := mintES256(t, key, validClaims())
	_, _, s, _ := split3(tok)
	raw, err := base64.RawURLEncoding.DecodeString(s)
	if err != nil {
		t.Fatalf("decode sig: %v", err)
	}
	if len(raw) != 64 {
		t.Fatalf("ES256 sig width = %d, want 64", len(raw))
	}
	r := new(big.Int).SetBytes(raw[:32])
	ss := new(big.Int).SetBytes(raw[32:])
	if r.Sign() == 0 || ss.Sign() == 0 {
		t.Fatal("ES256 r/s must be non-zero")
	}
	if _, err := v.VerifyAndExtract(tok); err != nil {
		t.Fatalf("self-minted ES256 MUST verify: %v", err)
	}
}
