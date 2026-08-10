// SPDX-License-Identifier: Apache-2.0

package authz

// Hand-rolled, alg-pinned, stdlib-only JWT verifier.
//
// SECURITY DESIGN (this is auth crypto — fail-closed everywhere):
//   - ALG-PINNING is the heart of the alg-confusion defense (the #1 JWT vuln).
//     The verifier is built for exactly ONE algorithm, inferred from the
//     configured public key type at construction (RSA -> RS256, ECDSA P-256 ->
//     ES256). VerifyAndExtract REJECTS any token whose header `alg` != the
//     pinned alg. This single equality check kills:
//       * alg:"none"  (no signature)
//       * alg:"HS256" presented against an RSA/ECDSA public key (the classic
//         "use the public key bytes as the HMAC secret" attack)
//       * any other alg mismatch (e.g. an ES256 token against an RS256 verifier)
//     This verifier NEVER supports HS* — there is no symmetric path at all.
//   - VERIFY-BEFORE-TRUST: the signature is verified over the signing input
//     BEFORE any claim is parsed or trusted. No partial/unverified claims are
//     ever returned.
//   - FAIL-CLOSED: every error path returns an error and NO claims. Missing
//     exp -> deny; expired -> deny; nbf in the future -> deny; iss/aud
//     configured but mismatched -> deny; missing tenant/scopes -> deny;
//     malformed token -> deny. There is no fallthrough-to-grant.

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"time"
)

// jwtClockSkew is the fixed tolerance applied to exp/nbf validation to absorb
// small clock differences between the token issuer and this verifier.
const jwtClockSkew = 30 * time.Second

// Pinned algorithm identifiers. The verifier supports exactly these two.
const (
	algRS256 = "RS256"
	algES256 = "ES256"
)

// rsaMinModulusBits is the minimum accepted RSA modulus size for an RS256 issuer
// key. Below this an RS256 signature is forgeable, so a weak key is rejected at
// construction (fail-closed). 2048 is the OWASP/NIST minimum and the project's
// stated RSA posture.
const rsaMinModulusBits = 2048

// Claims is the validated, verify-before-trust view of a JWT payload. It is
// only ever returned AFTER the signature has been verified and all required
// claims have passed validation. Sub is the principal id (for audit); Tenant
// and Scopes feed the synthetic APIKey that drives RBAC + tenant isolation.
type Claims struct {
	Sub    string   // "sub" claim (principal id; may be empty)
	Tenant string   // "tenant" claim (REQUIRED, non-empty)
	Scopes []string // "scopes" claim (REQUIRED, non-empty; "<action>:<pattern>")

	// Raw is the decoded payload map, retained for any future claim handling.
	Raw map[string]any
}

// JWTVerifier verifies a JWT against a single configured public key with a
// PINNED algorithm. The alg is fixed at construction from the key type and can
// never be influenced by the token header (the alg-confusion defense).
type JWTVerifier struct {
	pub      crypto.PublicKey
	alg      string // pinned: "RS256" or "ES256"
	issuer   string // if non-empty, the token "iss" must equal this
	audience string // if non-empty, the token "aud" must contain this
}

// NewJWTVerifier parses a PEM-encoded public key, pins the algorithm to the key
// type (RSA -> RS256, ECDSA P-256 -> ES256), and stores the optional
// issuer/audience to validate. It fails closed on any unsupported key type or
// curve, or any parse failure.
func NewJWTVerifier(pemBytes []byte, issuer, audience string) (*JWTVerifier, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, errors.New("authz: jwt public key: no PEM block found")
	}

	pub, err := parsePublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("authz: jwt public key: %w", err)
	}

	switch key := pub.(type) {
	case *rsa.PublicKey:
		// Reject weak RSA moduli: RS256 with a factorable/small modulus is forgeable
		// (a 512-bit modulus is trivially factored today; 1024-bit is within reach of
		// well-resourced attackers), after which an attacker can mint arbitrary valid
		// JWTs. Pinning the alg is moot if the key behind it is too small, so enforce
		// the OWASP/NIST RSA-2048+ minimum (the project's stated crypto posture).
		if bits := key.N.BitLen(); bits < rsaMinModulusBits {
			return nil, fmt.Errorf("authz: jwt public key: RSA modulus %d bits < %d minimum", bits, rsaMinModulusBits)
		}
		return &JWTVerifier{pub: key, alg: algRS256, issuer: issuer, audience: audience}, nil
	case *ecdsa.PublicKey:
		if key.Curve != elliptic.P256() {
			return nil, fmt.Errorf("authz: jwt public key: unsupported EC curve %q (only P-256/ES256)", key.Curve.Params().Name)
		}
		return &JWTVerifier{pub: key, alg: algES256, issuer: issuer, audience: audience}, nil
	default:
		return nil, fmt.Errorf("authz: jwt public key: unsupported key type %T (only RSA/ECDSA-P256)", pub)
	}
}

// parsePublicKey tries the PKIX (SubjectPublicKeyInfo) encoding first — the
// standard "PUBLIC KEY" PEM that carries both RSA and EC keys — then falls back
// to the RSA-specific PKCS#1 ("RSA PUBLIC KEY") encoding. EC public keys are
// only standardized in PKIX form, so there is no separate EC fallback. It only
// ever returns an *rsa.PublicKey or *ecdsa.PublicKey on success.
func parsePublicKey(der []byte) (crypto.PublicKey, error) {
	if pub, err := x509.ParsePKIXPublicKey(der); err == nil {
		return pub, nil
	}
	if pub, err := x509.ParsePKCS1PublicKey(der); err == nil {
		return pub, nil
	}
	return nil, errors.New("not a supported public key (tried PKIX, PKCS1)")
}

// jwtHeader is the minimal JOSE header we parse. Only alg and typ are read; alg
// is checked against the pinned alg, typ is checked leniently (allowed missing).
type jwtHeader struct {
	Alg string `json:"alg"`
	Typ string `json:"typ"`
}

// VerifyAndExtract verifies the token's signature against the pinned key/alg and
// validates the claims, returning the validated Claims on full success or a
// fail-closed error otherwise. It NEVER returns partial/unverified claims.
//
// Ordering is deliberate and security-critical: split -> decode -> header
// alg-pin check -> SIGNATURE VERIFY -> only then parse & validate claims.
func (v *JWTVerifier) VerifyAndExtract(token string) (*Claims, error) {
	// 1. Split into exactly 3 dot-separated segments.
	h, p, s, ok := split3(token)
	if !ok {
		return nil, errors.New("authz: jwt: token must have exactly 3 segments")
	}

	// 2. base64url (no-pad) decode each segment.
	headerJSON, err := base64.RawURLEncoding.DecodeString(h)
	if err != nil {
		return nil, fmt.Errorf("authz: jwt: bad base64url header: %w", err)
	}
	payloadJSON, err := base64.RawURLEncoding.DecodeString(p)
	if err != nil {
		return nil, fmt.Errorf("authz: jwt: bad base64url payload: %w", err)
	}
	sig, err := base64.RawURLEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("authz: jwt: bad base64url signature: %w", err)
	}

	// 3. Parse the header and enforce the ALG-PIN. This single check is the
	// alg-confusion defense: it rejects alg:"none", alg:"HS256"-with-public-key,
	// and any alg != the pinned one.
	var hdr jwtHeader
	if err := json.Unmarshal(headerJSON, &hdr); err != nil {
		return nil, fmt.Errorf("authz: jwt: bad header JSON: %w", err)
	}
	if hdr.Alg != v.alg {
		return nil, fmt.Errorf("authz: jwt: alg %q != pinned %q (rejected)", hdr.Alg, v.alg)
	}
	// typ policy (conservative / fail-closed): accept only an absent typ or the
	// exact "JWT". This deliberately REJECTS RFC 9068 access tokens (typ "at+jwt")
	// and any explicitly-typed JWS, so this verifier cannot consume standard OAuth2
	// access tokens — a deployment expectation, not a bug. Extend this allowlist if
	// "at+jwt" acceptance is ever required.
	if hdr.Typ != "" && hdr.Typ != "JWT" {
		return nil, fmt.Errorf("authz: jwt: unsupported typ %q", hdr.Typ)
	}

	// 4. Verify the signature over the signing input (the raw ASCII bytes of
	// "<b64url header>.<b64url payload>"). This happens BEFORE any claim is
	// trusted.
	signingInput := []byte(h + "." + p)
	if err := v.verifySignature(signingInput, sig); err != nil {
		return nil, err
	}

	// 5. ONLY NOW parse and validate claims.
	var raw map[string]any
	if err := json.Unmarshal(payloadJSON, &raw); err != nil {
		return nil, fmt.Errorf("authz: jwt: bad payload JSON: %w", err)
	}
	return v.validateClaims(raw)
}

// verifySignature checks the signature over signingInput using the pinned alg.
func (v *JWTVerifier) verifySignature(signingInput, sig []byte) error {
	digest := sha256.Sum256(signingInput)
	switch v.alg {
	case algRS256:
		pub, ok := v.pub.(*rsa.PublicKey)
		if !ok { // defensive: construction guarantees this
			return errors.New("authz: jwt: internal key/alg mismatch")
		}
		if err := rsa.VerifyPKCS1v15(pub, crypto.SHA256, digest[:], sig); err != nil {
			return fmt.Errorf("authz: jwt: RS256 signature verification failed: %w", err)
		}
		return nil
	case algES256:
		pub, ok := v.pub.(*ecdsa.PublicKey)
		if !ok { // defensive: construction guarantees this
			return errors.New("authz: jwt: internal key/alg mismatch")
		}
		// ES256 JWS signature is the raw concatenation r||s, each 32 bytes.
		if len(sig) != 64 {
			return fmt.Errorf("authz: jwt: ES256 signature must be 64 bytes, got %d", len(sig))
		}
		r := new(big.Int).SetBytes(sig[:32])
		s := new(big.Int).SetBytes(sig[32:])
		// Explicitly reject zero / out-of-range (r,s): a valid ECDSA signature has
		// r,s in [1, n-1]. crypto/ecdsa.Verify already rejects these in current
		// stdlib, but enforce it here so the zero-signature / malleability guarantee
		// does not silently depend on an unstated stdlib behaviour (fail-closed).
		n := pub.Curve.Params().N
		if r.Sign() <= 0 || s.Sign() <= 0 || r.Cmp(n) >= 0 || s.Cmp(n) >= 0 {
			return errors.New("authz: jwt: ES256 signature out of range")
		}
		if !ecdsa.Verify(pub, digest[:], r, s) {
			return errors.New("authz: jwt: ES256 signature verification failed")
		}
		return nil
	default: // defensive: construction guarantees one of the two
		return fmt.Errorf("authz: jwt: unsupported pinned alg %q", v.alg)
	}
}

// validateClaims enforces exp (required), nbf, iss, aud, tenant (required), and
// scopes (required). It is only reached AFTER signature verification.
func (v *JWTVerifier) validateClaims(raw map[string]any) (*Claims, error) {
	now := time.Now()

	// exp is REQUIRED.
	exp, ok := numericClaim(raw, "exp")
	if !ok {
		return nil, errors.New("authz: jwt: missing required claim \"exp\"")
	}
	expTime := time.Unix(exp, 0)
	if now.After(expTime.Add(jwtClockSkew)) {
		return nil, errors.New("authz: jwt: token expired")
	}

	// nbf is optional; if present, enforce it.
	if nbf, ok := numericClaim(raw, "nbf"); ok {
		nbfTime := time.Unix(nbf, 0)
		if now.Before(nbfTime.Add(-jwtClockSkew)) {
			return nil, errors.New("authz: jwt: token not yet valid (nbf)")
		}
	}

	// iss validated only when configured.
	if v.issuer != "" {
		iss, _ := raw["iss"].(string)
		if iss != v.issuer {
			return nil, fmt.Errorf("authz: jwt: issuer mismatch")
		}
	}

	// aud validated only when configured. aud may be a single string or an
	// array of strings; the configured audience must be present.
	if v.audience != "" {
		if !audienceContains(raw["aud"], v.audience) {
			return nil, fmt.Errorf("authz: jwt: audience mismatch")
		}
	}

	// tenant is REQUIRED and non-empty.
	tenant, _ := raw["tenant"].(string)
	if tenant == "" {
		return nil, errors.New("authz: jwt: missing required claim \"tenant\"")
	}

	// scopes is REQUIRED and non-empty (space-delimited string OR JSON array).
	scopes, err := parseScopes(raw["scopes"])
	if err != nil {
		return nil, err
	}
	if len(scopes) == 0 {
		return nil, errors.New("authz: jwt: missing required claim \"scopes\"")
	}

	sub, _ := raw["sub"].(string)

	return &Claims{
		Sub:    sub,
		Tenant: tenant,
		Scopes: scopes,
		Raw:    raw,
	}, nil
}

// split3 splits a JWT into its three dot-separated segments WITHOUT allocating a
// slice, and reports false if there are not exactly 3 segments (or any is
// empty). An empty segment is rejected: a valid JWS must have a non-empty
// header, payload, and signature (this also rejects the alg:"none" "x.y."
// form before decoding).
func split3(token string) (h, p, s string, ok bool) {
	first := -1
	second := -1
	for i := 0; i < len(token); i++ {
		if token[i] != '.' {
			continue
		}
		if first < 0 {
			first = i
		} else if second < 0 {
			second = i
		} else {
			// a third dot -> more than 3 segments
			return "", "", "", false
		}
	}
	if first < 0 || second < 0 {
		return "", "", "", false
	}
	h = token[:first]
	p = token[first+1 : second]
	s = token[second+1:]
	if h == "" || p == "" || s == "" {
		return "", "", "", false
	}
	return h, p, s, true
}

// numericClaim extracts a numeric date claim (exp/nbf) as a Unix seconds int64.
// JSON numbers decode to float64; we accept that and json.Number defensively.
func numericClaim(raw map[string]any, name string) (int64, bool) {
	v, ok := raw[name]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return int64(n), true
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, false
		}
		return i, true
	default:
		return 0, false
	}
}

// audienceContains reports whether the configured audience is present in the
// token "aud" claim, which may be a single string or an array of strings.
func audienceContains(aud any, want string) bool {
	switch a := aud.(type) {
	case string:
		return a == want
	case []any:
		for _, e := range a {
			if s, ok := e.(string); ok && s == want {
				return true
			}
		}
	}
	return false
}

// parseScopes accepts the "scopes" claim as EITHER a space-delimited string OR a
// JSON array of strings (each "<action>:<pattern>", the same format the registry
// uses). Returns the scope list (empty if absent/empty), or an error for a
// wrong-typed claim.
func parseScopes(v any) ([]string, error) {
	switch s := v.(type) {
	case nil:
		return nil, nil
	case string:
		return splitSpace(s), nil
	case []any:
		out := make([]string, 0, len(s))
		for _, e := range s {
			str, ok := e.(string)
			if !ok {
				return nil, errors.New("authz: jwt: scopes array must contain strings")
			}
			if str != "" {
				out = append(out, str)
			}
		}
		return out, nil
	default:
		return nil, errors.New("authz: jwt: scopes must be a string or array of strings")
	}
}

// splitSpace splits on runs of ASCII spaces, dropping empty fields. (A minimal
// stdlib-free split kept local to avoid importing strings for one call; equally
// strings.Fields would do — kept explicit for clarity.)
func splitSpace(s string) []string {
	var out []string
	start := -1
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r' {
			if start >= 0 {
				out = append(out, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, s[start:])
	}
	return out
}

// looksLikeJWT reports whether a bearer token looks like a JWT (three
// dot-separated non-empty segments, typically with the "eyJ" base64url prefix of
// a JSON header). This is a cheap pre-filter for the authorizer's JWT
// branch; it makes NO security decision — the actual verification is in
// VerifyAndExtract.
func looksLikeJWT(token string) bool {
	_, _, _, ok := split3(token)
	return ok
}
