// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/rostamlabs/rostam/vector"
)

// runKeysCmd implements the file-based key-admin CLI:
//
//	rostam-server keys add    -file <path> -token <t> -tenant <T> -scopes <s,s> [-cert-cn <cn>]
//	rostam-server keys revoke -file <path> -token <t>
//	rostam-server keys list   -file <path>
//
// It operates directly on a -keys-file JSON KeyRegistry (the same file the
// server loads via -keys-file). All mutations go through the registry's atomic
// AddKey/RevokeKey, so the on-disk file is never left half-written. This is the
// offline/file-based admin surface; an online admin API is a documented
// follow-up.
//
// args is os.Args[2:] (everything after "rostam-server keys"). On any usage or
// operation error it prints to stderr and exits non-zero (it is the top-level
// entry for the subcommand, so it owns process exit, exactly like main()).
func runKeysCmd(args []string) {
	if len(args) == 0 {
		keysUsage()
		os.Exit(2)
	}
	op, rest := args[0], args[1:]
	var err error
	switch op {
	case "add":
		err = keysAdd(rest)
	case "revoke":
		err = keysRevoke(rest)
	case "list":
		err = keysList(rest)
	case "-h", "--help", "help":
		keysUsage()
		return
	default:
		fmt.Fprintf(os.Stderr, "rostam-server keys: unknown subcommand %q\n", op)
		keysUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "rostam-server keys %s: %v\n", op, err)
		os.Exit(1)
	}
}

func keysUsage() {
	fmt.Fprint(os.Stderr, `rostam-server keys — file-based API-key administration

  keys add    -file <path> -token <t> -tenant <T> -scopes read:default/docs,write:default/* [-cert-cn <cn>]
  keys revoke -file <path> -token <t>
  keys list   -file <path>

Scopes are "<action>:<pattern>" (action ∈ read|write|admin), comma-separated.
A scope missing its ':' is rejected (fail-closed: a typo'd scope must not be
silently stored and then deny everything). list masks tokens by default.
`)
}

// validateScopes fail-closes on malformed scope syntax: every scope must be
// "<action>:<pattern>" with a non-empty action drawn from {read,write,admin} (or
// the superuser "*") and a non-empty pattern. A scope without a ':' (or with an
// empty/unknown action or empty pattern) is REJECTED rather than stored, so a
// typo'd scope cannot be silently persisted and then match nothing (denying
// everything for that key — a confusing, security-relevant foot-gun).
func validateScopes(scopes []string) error {
	for _, s := range scopes {
		action, pattern, ok := strings.Cut(s, ":")
		if !ok {
			return fmt.Errorf("malformed scope %q: want \"<action>:<pattern>\" (e.g. read:default/docs)", s)
		}
		if pattern == "" {
			return fmt.Errorf("malformed scope %q: empty resource pattern", s)
		}
		switch action {
		case vector.PermRead, vector.PermWrite, vector.PermAdmin, "*":
		default:
			return fmt.Errorf("malformed scope %q: action must be one of read|write|admin|* (got %q)", s, action)
		}
	}
	return nil
}

// parseScopes splits a comma-separated -scopes value, trims whitespace, drops
// empty entries, then validates each. An empty/whitespace -scopes yields nil
// (a key with no scopes — denied everything until edited; the operator owns
// that). Returns an error if any scope is malformed.
func parseScopes(csv string) ([]string, error) {
	var out []string
	for _, s := range strings.Split(csv, ",") {
		s = strings.TrimSpace(s)
		if s != "" {
			out = append(out, s)
		}
	}
	if err := validateScopes(out); err != nil {
		return nil, err
	}
	return out, nil
}

func keysAdd(args []string) error {
	fs := flag.NewFlagSet("keys add", flag.ContinueOnError)
	file := fs.String("file", "", "keys-file JSON path (created if missing)")
	token := fs.String("token", "", "the bearer token to register (PREFER the ROSTAM_KEY_TOKEN env var: a flag-passed secret is visible to other local users via /proc and shell history)")
	tenant := fs.String("tenant", "", "the tenant this key belongs to")
	scopes := fs.String("scopes", "", "comma-separated <action>:<pattern> scopes (e.g. read:default/docs,write:default/*)")
	certCN := fs.String("cert-cn", "", "optional mTLS client-cert CommonName this key is bound to")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *file == "" {
		return fmt.Errorf("-file is required")
	}
	// Secret resolution (env preferred over flag): a -token passed on the command
	// line is visible to other local users via /proc and shell history. Prefer the
	// ROSTAM_KEY_TOKEN env var; the env value wins when both are set, and a
	// flag-passed token emits a warning. The token is NEVER logged in full.
	tokenVal := *token
	if env := os.Getenv("ROSTAM_KEY_TOKEN"); env != "" {
		tokenVal = env
	} else if tokenVal != "" {
		fmt.Fprintln(os.Stderr, "rostam-server keys add: WARNING: -token passed on the command line is visible to other local users via /proc and shell history; prefer the ROSTAM_KEY_TOKEN environment variable")
	}
	// Validate scope syntax BEFORE touching the registry so a malformed scope
	// never reaches disk. AddKey itself enforces non-empty token+tenant.
	parsed, err := parseScopes(*scopes)
	if err != nil {
		return err
	}
	reg, err := vector.OpenKeyRegistry(*file)
	if err != nil {
		return err
	}
	if err := reg.AddKey(vector.APIKey{
		Token:  tokenVal,
		Tenant: *tenant,
		Scopes: parsed,
		CertCN: *certCN,
	}); err != nil {
		return err
	}
	fmt.Printf("added key %s (tenant=%s, scopes=[%s], cert-cn=%q)\n",
		maskToken(tokenVal), *tenant, strings.Join(parsed, ","), *certCN)
	return nil
}

func keysRevoke(args []string) error {
	fs := flag.NewFlagSet("keys revoke", flag.ContinueOnError)
	file := fs.String("file", "", "keys-file JSON path")
	token := fs.String("token", "", "the bearer token to revoke")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *file == "" {
		return fmt.Errorf("-file is required")
	}
	if *token == "" {
		return fmt.Errorf("-token is required")
	}
	reg, err := vector.OpenKeyRegistry(*file)
	if err != nil {
		return err
	}
	if err := reg.RevokeKey(*token); err != nil {
		return err
	}
	fmt.Printf("revoked key %s\n", maskToken(*token))
	return nil
}

func keysList(args []string) error {
	fs := flag.NewFlagSet("keys list", flag.ContinueOnError)
	file := fs.String("file", "", "keys-file JSON path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *file == "" {
		return fmt.Errorf("-file is required")
	}
	reg, err := vector.OpenKeyRegistry(*file)
	if err != nil {
		return err
	}
	keys := reg.ListKeys()
	sort.Slice(keys, func(i, j int) bool { return keys[i].Token < keys[j].Token })
	if len(keys) == 0 {
		fmt.Println("(no keys)")
		return nil
	}
	for _, k := range keys {
		scopes := k.Scopes
		if len(scopes) == 0 {
			// A legacy Permissions-only file loads with synthesized scopes, but if
			// neither is present show the (empty) scope list explicitly.
			scopes = nil
		}
		fmt.Printf("token=%s  tenant=%s  scopes=[%s]  cert-cn=%q\n",
			maskToken(k.Token), k.Tenant, strings.Join(scopes, ","), k.CertCN)
	}
	return nil
}

// maskTokenMinLen is the minimum token length before maskToken reveals its 4-char
// prefix: below this a 4-char prefix would disclose a meaningful FRACTION of the
// secret (e.g. 4 of 8 chars), so a short token is fully masked instead.
const maskTokenMinLen = 12

// maskToken renders a token for display without leaking it in full. This is
// defense-in-depth even though `keys list` is an admin-local op operating on the
// same file the admin already holds — the output may land in logs/terminal
// scrollback/CI, so we never echo the full secret. The 4-char prefix is shown
// ONLY for a sufficiently long token (>= maskTokenMinLen) where 4 chars is a small
// fraction; any shorter token is fully masked so a short/low-entropy token never
// leaks a meaningful prefix.
func maskToken(token string) string {
	if token == "" {
		return "(empty)"
	}
	if len(token) < maskTokenMinLen {
		return strings.Repeat("*", len(token))
	}
	return token[:4] + strings.Repeat("*", len(token)-4)
}
