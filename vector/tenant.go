// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"strings"
)

// DefaultTenant is the implicit tenant name used when a collection is
// referenced without a path prefix (e.g., "docs" instead of "acme/docs").
// Existing single-tenant deployments effectively run under "default".
const DefaultTenant = "default"

// ErrInvalidTenantName is returned when a collection name contains more than
// one path separator (e.g., "a/b/c") or is empty.
var ErrInvalidTenantName = errors.New("vector: collection name must be <tenant>/<collection> or just <collection>")

// splitTenant parses a collection name into its tenant and collection parts.
//
//   - "docs"         → ("default", "docs")    bare name, implicit tenant
//   - "acme/docs"    → ("acme", "docs")       explicit tenant
//   - "a/b/c"        → error                  no nested tenants in subsystem 2
//   - ""             → error                  empty name
//   - "acme/"        → error                  empty collection
//   - "/docs"        → error                  empty tenant
//
// The returned tenant and collection are always non-empty on success.
func splitTenant(name string) (tenant, collection string, err error) {
	if name == "" {
		return "", "", ErrInvalidTenantName
	}
	idx := strings.Index(name, "/")
	if idx < 0 {
		return DefaultTenant, name, nil
	}
	// At most one separator allowed in subsystem 2.
	if strings.Count(name, "/") > 1 {
		return "", "", ErrInvalidTenantName
	}
	tenant = name[:idx]
	collection = name[idx+1:]
	if tenant == "" || collection == "" {
		return "", "", ErrInvalidTenantName
	}
	return tenant, collection, nil
}

// TenantOf returns the tenant component of a canonical resource name — the
// single source of truth for "which tenant does this resource belong to",
// shared with the storage-layout parse so the authz tenant guard can never
// disagree with how a collection is actually laid out on disk.
//
// It is intentionally TOLERANT (it never errors) because the authz guard
// already operates on a resource that resourceFor has canonicalized
// ("<tenant>/<collection>", suffixes stripped):
//
//   - "acme/docs" → "acme"   (explicit tenant prefix)
//   - "docs"      → ""       (no '/': not a tenant-prefixed resource)
//   - ""          → ""       (cluster/no-collection resource)
//
// Returning "" for a bare/empty name (rather than DefaultTenant) keeps the
// guard's decision explicit: a resource with no tenant prefix matches no
// tenant-scoped key (it is handled by the guard's cluster-resource branch),
// rather than being silently attributed to "default".
//
// The tenant boundary is the FIRST '/' — identical to splitTenant's rule (and
// thus to the on-disk storage layout): the substring before the first '/' is
// the tenant. This shared rule is the whole point: the guard's tenant parse
// can never diverge from how the collection is actually keyed/stored, so a key
// cannot be fooled by a malformed-but-clever resource name.
func TenantOf(resource string) string {
	idx := strings.Index(resource, "/")
	if idx <= 0 {
		// "" / no separator / leading '/' (empty tenant) -> not tenant-prefixed.
		return ""
	}
	return resource[:idx]
}

// canonicalName returns the full canonical form of a collection name,
// always including the tenant prefix. Bare names get "default/" prepended.
//
//   - "docs"      → "default/docs"
//   - "acme/docs" → "acme/docs"
//
// Returns an error if the input fails splitTenant.
func canonicalName(name string) (string, error) {
	tenant, collection, err := splitTenant(name)
	if err != nil {
		return "", err
	}
	return tenant + "/" + collection, nil
}
