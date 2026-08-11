// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"testing"
	"time"
)

// newTestFlags builds a flag set shaped like the real one: a string, a bool, an
// int and a duration, which between them cover every Value.Set path a mistyped
// variable can take.
func newTestFlags() (*flag.FlagSet, *string, *bool, *int, *time.Duration) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	s := fs.String("http", ":8080", "")
	b := fs.Bool("pb-auto-failover", false, "")
	i := fs.Int("shards", 0, "")
	d := fs.Duration("wasm-blob-retention", 0, "")
	return fs, s, b, i, d
}

func env(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

func TestEnvNameForMapsDashesAndCase(t *testing.T) {
	for flagName, want := range map[string]string{
		"http":                "ROSTAM_HTTP",
		"api-key":             "ROSTAM_API_KEY",
		"pb-auto-failover":    "ROSTAM_PB_AUTO_FAILOVER",
		"wasm-blob-retention": "ROSTAM_WASM_BLOB_RETENTION",
	} {
		if got := envNameFor(flagName); got != want {
			t.Errorf("envNameFor(%q) = %q, want %q", flagName, got, want)
		}
	}
}

func TestEnvFillsFlagsNotGivenOnTheCommandLine(t *testing.T) {
	fs, s, b, i, d := newTestFlags()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	err := applyEnvDefaults(fs, env(map[string]string{
		"ROSTAM_HTTP":                "127.0.0.1:9999",
		"ROSTAM_PB_AUTO_FAILOVER":    "true",
		"ROSTAM_SHARDS":              "16",
		"ROSTAM_WASM_BLOB_RETENTION": "72h",
	}))
	if err != nil {
		t.Fatalf("applyEnvDefaults: %v", err)
	}
	if *s != "127.0.0.1:9999" || !*b || *i != 16 || *d != 72*time.Hour {
		t.Errorf("env not applied: http=%q failover=%v shards=%d retention=%v", *s, *b, *i, *d)
	}
}

// The precedence rule the whole design rests on: an explicit flag beats the
// environment. A container that sets ROSTAM_HTTP but is then run with an
// explicit -http must honour the command line.
func TestExplicitFlagBeatsEnv(t *testing.T) {
	fs, s, _, _, _ := newTestFlags()
	if err := fs.Parse([]string{"-http", "127.0.0.1:1234"}); err != nil {
		t.Fatal(err)
	}
	if err := applyEnvDefaults(fs, env(map[string]string{"ROSTAM_HTTP": "0.0.0.0:9999"})); err != nil {
		t.Fatal(err)
	}
	if *s != "127.0.0.1:1234" {
		t.Errorf("env overrode an explicit flag: got %q, want the command-line value", *s)
	}
}

// Set-but-empty is meaningful: -grpc "" disables the listener, so ROSTAM_GRPC=""
// must too. Treating empty as "unset" would make a listener impossible to turn
// off from the environment.
func TestEmptyEnvValueIsHonouredNotIgnored(t *testing.T) {
	fs, s, _, _, _ := newTestFlags()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if err := applyEnvDefaults(fs, env(map[string]string{"ROSTAM_HTTP": ""})); err != nil {
		t.Fatal(err)
	}
	if *s != "" {
		t.Errorf("empty env value ignored: got %q, want \"\"", *s)
	}
}

// A mistyped value must fail loudly. Silently keeping the default is how a node
// ends up running a configuration nobody chose.
func TestUnparseableEnvValueIsAnError(t *testing.T) {
	fs, _, _, _, _ := newTestFlags()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	err := applyEnvDefaults(fs, env(map[string]string{"ROSTAM_SHARDS": "sixteen"}))
	if err == nil {
		t.Fatal("expected an error for ROSTAM_SHARDS=sixteen, got nil")
	}
	// The message must name the variable, the value and the flag: the operator
	// reading it has a pod spec, not a stack trace.
	for _, want := range []string{"ROSTAM_SHARDS", "sixteen", "-shards"} {
		if !contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestUnsetEnvLeavesTheFlagDefault(t *testing.T) {
	fs, s, _, _, _ := newTestFlags()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if err := applyEnvDefaults(fs, env(nil)); err != nil {
		t.Fatal(err)
	}
	if *s != ":8080" {
		t.Errorf("default lost: got %q, want \":8080\"", *s)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
