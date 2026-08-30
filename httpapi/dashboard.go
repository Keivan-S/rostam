// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// topologyMember is one node in the topology JSON: its NodeID + client-facing
// ServerAddr (the Raft transport addr is intentionally never exposed).
type topologyMember struct {
	NodeID     string `json:"node_id"`
	ServerAddr string `json:"server_addr"`
}

// topologyResponse is the JSON shape of GET /v1/topology: the cluster routing
// snapshot the dashboard renders (shard count, members, per-shard leader addr,
// per-shard owner placement).
type topologyResponse struct {
	NumShards int              `json:"num_shards"`
	Members   []topologyMember `json:"members"`
	Leaders   []string         `json:"leaders"`
	Placement [][]string       `json:"placement"`
}

// topology serves the cluster routing snapshot as JSON. It dispatches the
// existing __topology__ read op (scope-gated like any other read via a.call —
// open when no authenticator is configured), decodes the gob-encoded result the
// smart client uses, and renders it. In single-node / Direct mode the op reports
// a one-member, single-shard map.
func (a *api) topology(w http.ResponseWriter, r *http.Request) {
	body, ok := a.call(w, r, "__topology__", nil)
	if !ok {
		return
	}
	t, err := ops.DecodeTopology(body)
	if err != nil {
		writeInternalError(w, "__topology__ decode", err)
		return
	}
	members := make([]topologyMember, 0, len(t.Members))
	for _, m := range t.Members {
		members = append(members, topologyMember{NodeID: m.NodeID, ServerAddr: m.ServerAddr})
	}
	// Normalize the two shard-indexed slices to non-nil so the JSON carries [] not
	// null on an empty/legacy map — a stable shape for the dashboard to consume.
	leaders := t.Leaders
	if leaders == nil {
		leaders = []string{}
	}
	placement := t.Placement
	if placement == nil {
		placement = [][]string{}
	}
	writeJSON(w, http.StatusOK, topologyResponse{
		NumShards: t.NumShards,
		Members:   members,
		Leaders:   leaders,
		Placement: placement,
	})
}

// collectionRef names one collection in the GET /v1/collections list. It is a
// struct (not a bare string) so the shape can grow per-collection fields later
// without breaking the response contract.
type collectionRef struct {
	Name string `json:"name"`
}

// collections lists this node's dense collections as {"collections":[{"name":..}]}.
// It dispatches the __collections__ read op (scope-gated via a.call, mirroring
// a.metrics) which reads the SAME CollectionStore.CollectionNames() source the
// Prometheus __metrics__ op enumerates, so the two never disagree.
func (a *api) collections(w http.ResponseWriter, r *http.Request) {
	body, ok := a.call(w, r, ops.CollectionsOp, nil)
	if !ok {
		return
	}
	names, err := ops.DecodeCollectionsResult(body)
	if err != nil {
		writeInternalError(w, "__collections__ decode", err)
		return
	}
	refs := make([]collectionRef, 0, len(names))
	for _, n := range names {
		refs = append(refs, collectionRef{Name: n})
	}
	writeJSON(w, http.StatusOK, map[string]any{"collections": refs})
}

// collectionConfigView serves a single collection's configuration as JSON. It
// wraps the existing vector_get_config read op (scope-gated via a.call), decodes
// the engine Config, and renders it through the same friendly collectionConfig
// shape create accepts — metric/quant/index_type as readable strings. An unknown
// collection surfaces as the op's 404 through the standard dispatch-error mapping.
func (a *api) collectionConfigView(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	body, ok := a.call(w, r, "vector_get_config", ops.EncodeGetConfigArgs(name))
	if !ok {
		return
	}
	cfg, err := ops.DecodeGetConfigResult(body)
	if err != nil {
		writeInternalError(w, "vector_get_config decode", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "config": fromConfig(cfg)})
}

// fromConfig renders an engine vector.Config into the friendly collectionConfig
// (the inverse of collectionConfig.toConfig): the integer metric/quant/index-type
// enums become the readable strings the create API accepts, and the remaining
// index-build knobs are copied straight across. It is the read-side companion to
// toConfig so a config round-trips (GET then re-POST) through the HTTP surface.
func fromConfig(c vector.Config) collectionConfig {
	var fullText *fullTextConfig
	if c.FullText != nil {
		fullText = &fullTextConfig{Analyzer: c.FullText.Analyzer, K1: c.FullText.K1, B: c.FullText.B}
	}
	return collectionConfig{
		Dim:                   c.Dim,
		Metric:                metricString(c.Metric),
		M:                     c.M,
		EfConstruction:        c.EfConstruction,
		EfSearch:              c.EfSearch,
		Seed:                  c.Seed,
		Quant:                 quantString(c.Quant),
		Persistent:            c.Persistent,
		RescoreFactor:         c.RescoreFactor,
		SQBits:                c.SQBits,
		PRQLayers:             c.PRQLayers,
		Partitions:            c.Partitions,
		ExtendCandidates:      c.ExtendCandidates,
		ExtendCandidatesMax:   c.ExtendCandidatesMax,
		Level0FullDegree:      c.Level0FullDegree,
		QuantizedBuild:        c.QuantizedBuild,
		IndexType:             indexTypeString(c.IndexType),
		IVFNlist:              c.IVFNlist,
		IVFNprobe:             c.IVFNprobe,
		VamanaR:               c.VamanaR,
		VamanaL:               c.VamanaL,
		VamanaAlpha:           c.VamanaAlpha,
		IVFPQ:                 c.IVFPQ,
		IVFPQM:                c.IVFPQM,
		IVFRerank:             c.IVFRerank,
		QuantPQM:              c.QuantPQM,
		OPQ:                   c.OPQ,
		OPQIters:              c.OPQIters,
		PQDropVecs:            c.PQDropVecs,
		IVFTrainThreshold:     c.IVFTrainThreshold,
		IVFDriftRetrain:       c.IVFDriftRetrain,
		IVFDriftGrowthFactor:  c.IVFDriftGrowthFactor,
		IVFDriftFactor:        c.IVFDriftFactor,
		FilterFirstRelativeBP: c.FilterFirstRelativeBP,
		FullText:              fullText,
		AnisotropicEta:        c.AnisotropicEta,
		SOAR:                  c.SOAR,
		SOARLambda:            c.SOARLambda,
		PQNBits:               c.PQNBits,
	}
}

// metricString is the inverse of parseMetric: the readable name for a metric
// enum. An unrecognized value falls back to the default "cosine".
func metricString(m vector.Metric) string {
	switch m {
	case vector.L2:
		return "l2"
	case vector.DotProduct:
		return "dot"
	default:
		return "cosine"
	}
}

// quantString is the inverse of parseQuant: the readable name for a quant enum.
// QuantNone renders as "none". An unrecognized value falls back to "none".
func quantString(q vector.QuantMode) string {
	switch q {
	case vector.QuantSQ8:
		return "sq8"
	case vector.QuantBQ1:
		return "bq1"
	case vector.QuantPQ:
		return "pq"
	case vector.QuantSQ:
		return "sq"
	case vector.QuantPRQ:
		return "prq"
	default:
		return "none"
	}
}

// indexTypeString is the inverse of parseIndexType: the readable name for an
// index-type enum. An unrecognized value falls back to the default "hnsw".
func indexTypeString(t vector.IndexType) string {
	switch t {
	case vector.IndexIVF:
		return "ivf"
	case vector.IndexVamana:
		return "vamana"
	default:
		return "hnsw"
	}
}
