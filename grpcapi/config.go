// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"fmt"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// toConfig maps a protobuf Config onto vector.Config, resolving the friendly
// string metric/quant and filling the standard HNSW defaults for zero m/ef_*
// (the engine rejects zeros rather than defaulting them itself).
func toConfig(c *pb.Config) (vector.Config, error) {
	if c == nil {
		return vector.Config{}, fmt.Errorf("config is required")
	}
	metric, err := parseMetric(c.GetMetric())
	if err != nil {
		return vector.Config{}, err
	}
	quant, err := parseQuant(c.GetQuant())
	if err != nil {
		return vector.Config{}, err
	}
	if c.GetPartitions() < 0 {
		return vector.Config{}, fmt.Errorf("partitions must be non-negative")
	}
	indexType, err := parseIndexType(c.GetIndexType())
	if err != nil {
		return vector.Config{}, err
	}
	if c.GetIvfNlist() < 0 || c.GetIvfNprobe() < 0 {
		return vector.Config{}, fmt.Errorf("ivf_nlist and ivf_nprobe must be non-negative")
	}
	// Caught early (before the uint32 create-wire encoding silently wraps a
	// negative to a large positive) so a bad threshold fails loud, mirroring the
	// engine's Validate rule (IVFTrainThreshold >= 0).
	if c.GetIvfTrainThreshold() < 0 {
		return vector.Config{}, fmt.Errorf("ivf_train_threshold must be non-negative")
	}
	// Drift-retrain factors must be > 1.0 (0 = engine default), mirroring the engine's
	// Validate rule. Caught here so a bad factor fails loud before the create wire.
	if c.GetIvfDriftGrowthFactor() != 0 && c.GetIvfDriftGrowthFactor() <= 1.0 {
		return vector.Config{}, fmt.Errorf("ivf_drift_growth_factor must be > 1.0")
	}
	if c.GetIvfDriftFactor() != 0 && c.GetIvfDriftFactor() <= 1.0 {
		return vector.Config{}, fmt.Errorf("ivf_drift_factor must be > 1.0")
	}
	// FilterFirstRelativeBP is basis points (0..10000); fail loud before the create
	// wire, mirroring the engine's Validate rule.
	if c.GetFilterFirstRelativeBp() < 0 || c.GetFilterFirstRelativeBp() > 10000 {
		return vector.Config{}, fmt.Errorf("filter_first_relative_bp must be in [0, 10000]")
	}
	// OPQIters drives full-OPQ iterative refinement; fail loud on out-of-range
	// before the create wire, mirroring the engine's Validate rule ([0, 20]).
	if c.GetOpqIters() < 0 || c.GetOpqIters() > 20 {
		return vector.Config{}, fmt.Errorf("opq_iters must be in [0, 20]")
	}
	// Map the optional full-text config onto vector.FullTextConfig (nil = disabled).
	// The engine Validate enforces HNSW-only + a registered analyzer name.
	var fullText *vector.FullTextConfig
	if ft := c.GetFullText(); ft != nil {
		fullText = &vector.FullTextConfig{
			Analyzer: ft.GetAnalyzer(),
			K1:       ft.GetK1(),
			B:        ft.GetB(),
		}
	}
	m, efc, efs := int(c.GetM()), int(c.GetEfConstruction()), int(c.GetEfSearch())
	if m <= 0 {
		m = 16
	}
	if efc <= 0 {
		efc = 200
	}
	if efs <= 0 {
		efs = 64
	}
	return vector.Config{
		Dim:                   int(c.GetDim()),
		Metric:                metric,
		M:                     m,
		EfConstruction:        efc,
		EfSearch:              efs,
		Seed:                  c.GetSeed(),
		Quant:                 quant,
		SQBits:                int(c.GetSqBits()),
		PRQLayers:             int(c.GetPrqLayers()),
		Persistent:            c.GetPersistent(),
		RescoreFactor:         int(c.GetRescoreFactor()),
		Partitions:            int(c.GetPartitions()),
		IndexType:             indexType,
		VamanaR:               int(c.GetVamanaR()),
		VamanaL:               int(c.GetVamanaL()),
		VamanaAlpha:           c.GetVamanaAlpha(),
		IVFNlist:              int(c.GetIvfNlist()),
		IVFNprobe:             int(c.GetIvfNprobe()),
		IVFPQ:                 c.GetIvfPq(),
		IVFPQM:                int(c.GetIvfPqM()),
		IVFRerank:             c.GetIvfRerank(),
		QuantPQM:              int(c.GetQuantPqM()),
		OPQ:                   c.GetOpq(),
		OPQIters:              int(c.GetOpqIters()),
		PQDropVecs:            c.GetPqDropVecs(),
		IVFTrainThreshold:     int(c.GetIvfTrainThreshold()),
		IVFDriftRetrain:       c.GetIvfDriftRetrain(),
		IVFDriftGrowthFactor:  c.GetIvfDriftGrowthFactor(),
		IVFDriftFactor:        c.GetIvfDriftFactor(),
		FilterFirstRelativeBP: int(c.GetFilterFirstRelativeBp()),
		FullText:              fullText,
		AnisotropicEta:        c.GetAnisotropicEta(),
		SOAR:                  c.GetSoar(),
		SOARLambda:            c.GetSoarLambda(),
		PQNBits:               int(c.GetPqNbits()),
	}, nil
}

func parseIndexType(s string) (vector.IndexType, error) {
	switch s {
	case "", "hnsw":
		return vector.IndexHNSW, nil
	case "ivf", "ivf-flat", "ivfflat":
		return vector.IndexIVF, nil
	case "vamana", "diskann":
		return vector.IndexVamana, nil
	}
	return 0, fmt.Errorf("unknown index_type %q", s)
}

func parseMetric(s string) (vector.Metric, error) {
	switch s {
	case "", "cosine":
		return vector.Cosine, nil
	case "l2", "euclidean":
		return vector.L2, nil
	case "dot", "dotproduct", "ip":
		return vector.DotProduct, nil
	}
	return 0, fmt.Errorf("unknown metric %q", s)
}

func parseQuant(s string) (vector.QuantMode, error) {
	switch s {
	case "", "none":
		return vector.QuantNone, nil
	case "sq8":
		return vector.QuantSQ8, nil
	case "bq1":
		return vector.QuantBQ1, nil
	case "pq":
		return vector.QuantPQ, nil
	case "sq":
		return vector.QuantSQ, nil
	case "prq":
		return vector.QuantPRQ, nil
	}
	return 0, fmt.Errorf("unknown quant %q", s)
}

func parseFusion(s string) (vector.FusionMethod, error) {
	switch s {
	case "", "rrf":
		return vector.FusionRRF, nil
	case "weighted":
		return vector.FusionWeighted, nil
	case "dbsf":
		return vector.FusionDBSF, nil
	}
	return 0, fmt.Errorf("unknown fusion method %q", s)
}
