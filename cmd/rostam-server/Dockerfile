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
#
# AUTHENTICATION IS REQUIRED. The default CMD binds 0.0.0.0, which the server
# refuses to serve unauthenticated, so pass a key:
#
#     docker run -p 8080:8080 -e ROSTAM_API_KEY=<token> rostam-server:latest
#
# ROSTAM_API_KEY is read instead of -api-key so the secret never appears in the
# process table or in `docker inspect`'s Cmd. For a deliberately open dev
# container, append -insecure to the command instead.
ENTRYPOINT ["rostam-server"]
CMD ["-http", "0.0.0.0:8080"]
