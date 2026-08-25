module github.com/rostamlabs/rostam

go 1.26.1

// Pinned to a PATCHED toolchain (production-readiness #2). go1.26.1 is affected by
// standard-library vulnerabilities reachable from this server's TLS/mTLS and HTTP
// paths — govulncheck ./cmd/rostam-server/ on 1.26.1 reports crypto/tls
// GO-2026-5856 (fixed 1.26.5), crypto/x509 GO-2026-5037 and net/textproto
// GO-2026-5039 (fixed 1.26.4), and html/template GO-2026-4982/4980 (fixed 1.26.3).
// 1.26.5 covers all of them. CI honors this line via setup-go's
// `go-version-file: go.mod`, so the build and the release binaries move together.
// Raise it (and re-run `govulncheck ./cmd/rostam-server/`) when the next patch
// release lands; keep the `go` directive at the language version.
toolchain go1.26.5

require github.com/cespare/xxhash/v2 v2.3.0

require (
	github.com/bytecodealliance/wasmtime-go/v45 v45.0.0
	github.com/hashicorp/raft v1.7.3
	github.com/jackc/puddle/v2 v2.2.2
	github.com/panjf2000/gnet/v2 v2.10.0
	github.com/rostamlabs/rembed v0.3.0
	golang.org/x/sys v0.47.0
	google.golang.org/grpc v1.81.1
	google.golang.org/protobuf v1.36.11
)

require (
	github.com/armon/go-metrics v0.4.1 // indirect
	github.com/fatih/color v1.13.0 // indirect
	github.com/hashicorp/go-hclog v1.6.2 // indirect
	github.com/hashicorp/go-immutable-radix v1.0.0 // indirect
	github.com/hashicorp/go-metrics v0.5.4 // indirect
	github.com/hashicorp/go-msgpack/v2 v2.1.2 // indirect
	github.com/hashicorp/golang-lru v0.5.0 // indirect
	github.com/mattn/go-colorable v0.1.12 // indirect
	github.com/mattn/go-isatty v0.0.14 // indirect
	github.com/panjf2000/ants/v2 v2.12.1 // indirect
	github.com/valyala/bytebufferpool v1.0.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.uber.org/zap v1.28.0 // indirect
	golang.org/x/net v0.55.0 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260226221140-a57be14db171 // indirect
	gopkg.in/natefinch/lumberjack.v2 v2.2.1 // indirect
)
