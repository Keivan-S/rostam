// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"testing"
)

// TestCallSwitchOpsAreClassified is the drift guard for finding 006: EVERY op the
// fanoutDispatcher.Call switch handles must be classified as either a DATA-PLANE
// alias-resolved op (present in dataPlaneAliasOps) or an explicitly-listed
// admin/real-name/control op. It parses the switch's case labels from source, so
// ADDING a new switch case without classifying it FAILS this test — closing the
// hand-sync gap that let vector_get_batch / vector_mv_get_batch /
// vector_named_get_batch / vector_mv_scroll ship missing from dataPlaneAliasOps
// (silent alias resolution failures on the networked unpartitioned path).
func TestCallSwitchOpsAreClassified(t *testing.T) {
	// Admin / real-name / control ops that MUST NOT be alias-resolved: they carry
	// real collection names, physical '#'/'@' names, or are non-shard-routed
	// coordinator ops. Listing them here (rather than in dataPlaneAliasOps)
	// documents the intentional exclusion so the two sets stay mutually exclusive.
	adminOrRealName := map[string]struct{}{
		"vector_create_collection": {}, "vector_drop_collection": {},
		"vector_mv_create_collection": {}, "vector_mv_drop_collection": {},
		"vector_named_create_collection": {}, "vector_named_drop_collection": {},
		"vector_named_get_config": {},
		"vector_resplit":          {}, "vector_mv_resplit": {},
		"vector_resplit_cleanup": {}, "vector_mv_resplit_cleanup": {},
		"vector_reshard": {}, "vector_mv_reshard": {},
		"vector_reshard_abort": {}, "vector_mv_reshard_abort": {},
		"alias_batch": {}, "alias_list": {},
	}

	switchOps := callSwitchOps(t)
	if len(switchOps) < 40 {
		t.Fatalf("parsed only %d Call-switch ops; the parser likely failed to find the switch", len(switchOps))
	}
	for _, op := range switchOps {
		_, isData := dataPlaneAliasOps[op]
		_, isAdmin := adminOrRealName[op]
		switch {
		case isData && isAdmin:
			t.Errorf("op %q is in BOTH dataPlaneAliasOps and the admin exception set — classify it as exactly one", op)
		case !isData && !isAdmin:
			t.Errorf("op %q handled by Call switch is UNCLASSIFIED — add it to dataPlaneAliasOps "+
				"(data-plane, alias-resolved) or to the admin/real-name exception set in this test", op)
		}
	}
}

// callSwitchOps parses fanout_dispatcher.go and returns the string-literal case
// labels of the fanoutDispatcher.Call method's `switch name` statement. Non-literal
// cases (e.g. ops.WCEnvelopeOp, a named const) are skipped — the alias-drift class
// is entirely string-named vector_* ops.
func callSwitchOps(t *testing.T) []string {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "fanout_dispatcher.go", nil, 0)
	if err != nil {
		t.Fatalf("parse fanout_dispatcher.go: %v", err)
	}
	var out []string
	ast.Inspect(file, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "Call" || !isFanoutDispatcherRecv(fn.Recv) {
			return true
		}
		ast.Inspect(fn.Body, func(m ast.Node) bool {
			sw, ok := m.(*ast.SwitchStmt)
			if !ok {
				return true
			}
			for _, stmt := range sw.Body.List {
				cc, ok := stmt.(*ast.CaseClause)
				if !ok {
					continue
				}
				for _, expr := range cc.List {
					lit, ok := expr.(*ast.BasicLit)
					if !ok || lit.Kind != token.STRING {
						continue
					}
					if s, err := strconv.Unquote(lit.Value); err == nil {
						out = append(out, s)
					}
				}
			}
			return true
		})
		return false
	})
	return out
}

func isFanoutDispatcherRecv(recv *ast.FieldList) bool {
	if recv == nil || len(recv.List) != 1 {
		return false
	}
	star, ok := recv.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	id, ok := star.X.(*ast.Ident)
	return ok && id.Name == "fanoutDispatcher"
}
