// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"strings"
)

// envPrefix namespaces every variable this binary reads, so a ROSTAM_-prefixed
// name in a shared environment (a Kubernetes pod spec, a docker --env-file) is
// unambiguously ours.
const envPrefix = "ROSTAM_"

// envNameFor maps a flag name onto its environment variable: -pb-auto-failover
// becomes ROSTAM_PB_AUTO_FAILOVER. The mapping is mechanical so there is no
// table to keep in sync with the 59 flags, and no flag that quietly lacks one.
func envNameFor(flagName string) string {
	return envPrefix + strings.ToUpper(strings.ReplaceAll(flagName, "-", "_"))
}

// applyEnvDefaults fills every flag that was NOT given on the command line from
// its ROSTAM_* variable. Precedence is therefore:
//
//	command-line flag  >  ROSTAM_* variable  >  flag default
//
// The one deliberate exception is secrets (-api-key, -internal-token), where
// resolveSecret runs afterwards and lets the environment win: a secret on the
// command line is visible to other local users via /proc and lands in shell
// history, so the environment is the safer source and should not be silently
// overridden by a worse one.
//
// An empty value is honoured rather than ignored, because empty is meaningful
// here — ROSTAM_GRPC="" disables the gRPC listener, exactly as -grpc "" does.
// That is why this uses LookupEnv semantics (set-but-empty ≠ unset) instead of
// treating "" as absent.
//
// An unparseable value is an error, not a warning. A mistyped ROSTAM_SHARDS
// that silently left the default in place is precisely the failure this exists
// to prevent.
func applyEnvDefaults(fs *flag.FlagSet, lookup func(string) (string, bool)) error {
	explicit := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

	var firstErr error
	fs.VisitAll(func(f *flag.Flag) {
		// VisitAll cannot stop early; report the first failure and skip the rest.
		if firstErr != nil || explicit[f.Name] {
			return
		}
		env := envNameFor(f.Name)
		v, ok := lookup(env)
		if !ok {
			return
		}
		if err := f.Value.Set(v); err != nil {
			firstErr = fmt.Errorf("%s=%q is not a valid value for -%s: %w", env, v, f.Name, err)
		}
	})
	return firstErr
}
