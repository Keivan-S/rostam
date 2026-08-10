# Multi-stage build for rostam-server (the REST/gRPC/TCP front door).
# Build context must be the repo root:  docker build -f cmd/rostam-server/Dockerfile .
#
# cgo is required (wasmtime-go is pulled in transitively), so the build image
# carries a C toolchain and the runtime image is debian-slim (not scratch).
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/rostam-server ./cmd/rostam-server

FROM debian:bookworm-slim
RUN useradd -m -u 10001 rostam && mkdir -p /data && chown rostam:rostam /data
COPY --from=build /out/rostam-server /usr/local/bin/rostam-server
USER rostam
# Writable working dir: the vector store creates a "vectors/" dir relative to
# the data dir (empty data dir → relative to CWD), so CWD must be writable by
# the rostam user even in the in-memory default.
WORKDIR /data
EXPOSE 8080
# Single-node, in-memory by default (fast HNSW, no disk). Override the flags to
# enable persistence / mmap (-persistent-vectors -cluster ... -data /data) or
# other transports (-grpc :9090 -tcp :7000).
ENTRYPOINT ["rostam-server"]
CMD ["-http", "0.0.0.0:8080"]
