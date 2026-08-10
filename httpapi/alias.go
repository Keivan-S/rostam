// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"

	"github.com/rostamlabs/rostam/ops"
)

// Collection aliases over HTTP. An alias is a name that resolves to a real
// (canonical) collection, so data-plane ops on the alias transparently route to
// the target (the resolution lives in the engine — /v1/collections/{alias}/...
// already works). These endpoints are the alias-MANAGEMENT surface: create,
// delete, list, and an ATOMIC batch (the zero-downtime swap). They mirror the
// reshard transport: each handler lowers its JSON request into the existing
// alias ops codec and dispatches the coordinator op (alias_batch / alias_list).
//
// Validation errors from the engine (target missing / shadow / reserved char /
// target-is-alias) all carry the "rostam: alias " message prefix and map to HTTP
// 400 via statusForError — never a 500. A delete of an absent alias is a no-op
// (idempotent 200).

// aliasCreateReq is the body for POST /v1/aliases: point alias at collection.
type aliasCreateReq struct {
	Alias      string `json:"alias"`
	Collection string `json:"collection"`
}

// createAlias creates (or overwrites — upsert) an alias. It lowers to a single-
// action alias_batch so the create/delete/swap paths share one atomic op.
func (a *api) createAlias(w http.ResponseWriter, r *http.Request) {
	var req aliasCreateReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Alias == "" || req.Collection == "" {
		writeError(w, http.StatusBadRequest, "alias and collection are required")
		return
	}
	if !validName(w, req.Alias) || !validName(w, req.Collection) {
		return
	}
	if _, ok := a.call(w, r, "alias_batch", ops.EncodeAliasCreateArgs(req.Alias, req.Collection)); !ok {
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"alias": req.Alias, "collection": req.Collection})
}

// deleteAlias removes an alias (a single-action delete batch). An absent alias
// is a no-op — the engine treats delete as idempotent, so this is always 200.
func (a *api) deleteAlias(w http.ResponseWriter, r *http.Request) {
	alias := r.PathValue("alias")
	if _, ok := a.call(w, r, "alias_batch", ops.EncodeAliasDeleteArgs(alias)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"deleted": alias})
}

// aliasEntryJSON is one {alias, collection} pair in the list response.
type aliasEntryJSON struct {
	Alias      string `json:"alias"`
	Collection string `json:"collection"`
}

// listAliases returns all aliases, optionally filtered by ?collection=docs. The
// list is a local FSM read (alias_list); the result is decoded from the binary
// codec into JSON {"aliases":[{"alias":...,"collection":...}]}.
func (a *api) listAliases(w http.ResponseWriter, r *http.Request) {
	filter := r.URL.Query().Get("collection")
	body, ok := a.call(w, r, "alias_list", ops.EncodeAliasListArgs(filter))
	if !ok {
		return
	}
	entries, err := ops.DecodeAliasListResult(body)
	if err != nil {
		writeInternalError(w, "alias_list decode", err)
		return
	}
	out := make([]aliasEntryJSON, 0, len(entries))
	for _, e := range entries {
		out = append(out, aliasEntryJSON{Alias: e.Alias, Collection: e.Collection})
	}
	writeJSON(w, http.StatusOK, map[string]any{"aliases": out})
}

// aliasActionJSON is one action in an atomic batch. Exactly one of Create/Delete
// is set: Create points an alias at a collection (upsert), Delete removes one.
type aliasActionJSON struct {
	Create *aliasCreateReq    `json:"create,omitempty"`
	Delete *aliasDeleteAction `json:"delete,omitempty"`
}

type aliasDeleteAction struct {
	Alias string `json:"alias"`
}

// aliasBatchReq is the body for POST /v1/aliases/batch: a list of create/delete
// actions applied ATOMICALLY in one alias_batch op. A swap is expressed as
// [{delete prod},{create prod→docs2}] — both land in the same Raft entry, so a
// concurrent reader never sees an undefined intermediate state.
type aliasBatchReq struct {
	Actions []aliasActionJSON `json:"actions"`
}

// aliasBatch applies a list of create/delete actions in ONE atomic alias_batch
// op (the zero-downtime swap). Each action must set exactly one of create/delete.
func (a *api) aliasBatch(w http.ResponseWriter, r *http.Request) {
	var req aliasBatchReq
	if !decodeBody(w, r, &req) {
		return
	}
	actions := make([]ops.AliasAction, 0, len(req.Actions))
	for i := range req.Actions {
		act := req.Actions[i]
		switch {
		case act.Create != nil && act.Delete != nil:
			writeError(w, http.StatusBadRequest, "each action must set exactly one of create/delete")
			return
		case act.Create != nil:
			if act.Create.Alias == "" || act.Create.Collection == "" {
				writeError(w, http.StatusBadRequest, "create action requires alias and collection")
				return
			}
			if !validName(w, act.Create.Alias) || !validName(w, act.Create.Collection) {
				return
			}
			actions = append(actions, ops.AliasAction{Alias: act.Create.Alias, Canonical: act.Create.Collection})
		case act.Delete != nil:
			if act.Delete.Alias == "" {
				writeError(w, http.StatusBadRequest, "delete action requires alias")
				return
			}
			if !validName(w, act.Delete.Alias) {
				return
			}
			actions = append(actions, ops.AliasAction{Alias: act.Delete.Alias, Delete: true})
		default:
			writeError(w, http.StatusBadRequest, "each action must set exactly one of create/delete")
			return
		}
	}
	if _, ok := a.call(w, r, "alias_batch", ops.EncodeAliasBatchArgs(actions)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"applied": len(actions)})
}
