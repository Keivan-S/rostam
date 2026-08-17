// Command _oracle dumps golden hex for Rostam's op arg encoders, so the Python
// native client can be differential-tested byte-for-byte against the Go
// reference. It is a test oracle, not shipped: `go run` it from the repo root.
//
//	go run ./clients/python/tests/_oracle > clients/python/tests/_oracle/golden.txt
//
// Each line is  name<TAB>hex.  The Python test recomputes each and asserts equal.
package main

import (
	"encoding/hex"
	"fmt"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

func emit(name string, b []byte) { fmt.Printf("%s\t%s\n", name, hex.EncodeToString(b)) }

func main() {
	// ---- create_collection: a matrix that exercises the config trailer -------
	base := vector.Config{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64}
	emit("create/basic", ops.EncodeCreateCollectionArgs("docs", base))

	c := base
	c.Metric = vector.L2
	c.Seed = 42
	emit("create/l2_seed", ops.EncodeCreateCollectionArgs("docs", c))

	c = base
	c.Quant = vector.QuantSQ8
	emit("create/sq8", ops.EncodeCreateCollectionArgs("c-sq8", c))

	c = base
	c.Quant = vector.QuantBQ1
	emit("create/bq1", ops.EncodeCreateCollectionArgs("c", c))

	c = base
	c.Persistent = true
	c.RescoreFactor = 3
	emit("create/persistent_rescore", ops.EncodeCreateCollectionArgs("p", c))

	c = base
	c.Partitions = 4
	emit("create/partitions", ops.EncodeCreateCollectionArgs("part", c))

	c = base
	c.ExtendCandidates = true
	c.ExtendCandidatesMax = 200
	c.Level0FullDegree = true
	c.QuantizedBuild = true
	emit("create/hnsw_levers", ops.EncodeCreateCollectionArgs("lev", c))

	c = base
	c.IndexType = vector.IndexIVF
	c.IVFNlist = 64
	c.IVFNprobe = 8
	emit("create/ivf", ops.EncodeCreateCollectionArgs("ivf", c))

	c = base
	c.IndexType = vector.IndexVamana
	c.VamanaR = 64
	c.VamanaL = 100
	c.VamanaAlpha = 1.2
	emit("create/vamana", ops.EncodeCreateCollectionArgs("vam", c))

	c = base
	c.FullText = &vector.FullTextConfig{Analyzer: "english", K1: 1.2, B: 0.75}
	emit("create/fulltext", ops.EncodeCreateCollectionArgs("ft", c))

	c = base
	c.Dim = 128
	c.M = 32
	c.EfConstruction = 400
	c.EfSearch = 128
	emit("create/tuned", ops.EncodeCreateCollectionArgs("longname-collection", c))

	c = base
	c.AnisotropicEta = 0.2
	emit("create/aniso", ops.EncodeCreateCollectionArgs("an", c))

	c = base
	c.SOAR = true
	c.SOARLambda = 1.5
	emit("create/soar", ops.EncodeCreateCollectionArgs("so", c))

	c = base
	c.Quant = vector.QuantSQ
	c.SQBits = 6
	emit("create/sqbits", ops.EncodeCreateCollectionArgs("sq", c))

	c = base
	c.IVFTrainThreshold = 10000
	c.IndexType = vector.IndexIVF
	emit("create/ivf_threshold", ops.EncodeCreateCollectionArgs("ivt", c))

	// ---- data-plane ops ------------------------------------------------------
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	emit("insert/plain", ops.EncodeVectorInsertArgs("docs", 1, vec))
	emit("insert/ttl_meta", ops.EncodeVectorInsertArgsExt("docs", 7, vec, 5*time.Minute,
		vector.Metadata{"tenant": vector.NewString("acme"), "year": vector.NewInt(2020)},
		vector.SparseVector{}))
	emit("insert/sparse", ops.EncodeVectorInsertArgsExt("docs", 9, vec, 0, nil,
		vector.SparseVector{Indices: []uint32{3, 17}, Values: []float32{0.8, 0.4}}))

	emit("upsert/plain", ops.EncodeVectorUpsertArgs("docs", 1, vec, "hello", 0, nil, vector.SparseVector{}))
	emit("upsert/content_meta", ops.EncodeVectorUpsertArgs("docs", 2, vec, "world", 0,
		vector.Metadata{"tenant": vector.NewString("acme")}, vector.SparseVector{}))

	emit("search/plain", ops.EncodeVectorSearchArgs("docs", 10, vec))
	emit("search/filter", ops.EncodeVectorSearchArgsExt("docs", 5, vec,
		vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}))

	emit("delete", ops.EncodeVectorDeleteArgs("docs", 42))
	emit("exists", ops.EncodeExistsArgs("docs", 42))
	emit("get/plain", ops.EncodeVectorGetArgs("docs", 1, 0))
	emit("get/withvec_payload", ops.EncodeVectorGetArgs("docs", 1, 0x03))
}
