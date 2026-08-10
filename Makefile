.PHONY: all build docker lint test test-serial race bench test-python dist-python publish-python tidy clean

GO ?= go
PYTHON ?= python3
PKG := ./...
BIN ?= bin/rostam-server
IMAGE ?= rostam-server:latest

all: lint test race bench

# Build the server binary. cgo is required (wasmtime-go is pulled in
# transitively), so a C toolchain must be on PATH; the vector/cache/ops packages
# themselves build with CGO_ENABLED=0 if you only need the library.
build:
	$(GO) build -trimpath -ldflags="-s -w" -o $(BIN) ./cmd/rostam-server

# Build the container image. Context is the repo root; the Dockerfile lives with
# the server command.
docker:
	docker build -f cmd/rostam-server/Dockerfile -t $(IMAGE) .

lint:
	@command -v golangci-lint >/dev/null || (echo "install: https://golangci-lint.run/"; exit 1)
	golangci-lint run $(PKG)

# The slow cluster/cross-process integration tests live in ./inttest/ (their own
# test binary). -timeout is applied PER PACKAGE, so each binary gets the full budget
# and packages run in parallel: the root binary finishes fast while ./inttest/ uses
# its own budget. Keep the timeout >= the inttest suite's wall-time (~7m, more under -race).
test:
	$(GO) test -count=1 -timeout 20m $(PKG)

# test-serial limits cross-package parallelism (-p 1: one test binary at a time).
# The in-process RF=3 cluster tests in ./inttest/ (and the root/cluster cluster
# tests) each spin many raft groups (~12-27) / >100 goroutines; running several test
# BINARIES concurrently on a busy machine CPU-oversubscribes them, starving raft of
# the cycles it needs to elect/replicate within the (already widened, finite) setup
# deadlines -> rare load-only flakes (a shard that gets no CPU can't elect, so a seed
# upsert sees "not leader" until its budget expires). These pass in isolation; the
# residual is contention, not a logic bug. Use this target to chase or avoid that
# flake without changing the default ./... semantics above.
test-serial:
	$(GO) test -count=1 -p 1 -timeout 25m $(PKG)

race:
	$(GO) test -count=1 -race -short -timeout 30m $(PKG)

bench:
	$(GO) test -run=^$$ -bench=. -benchmem -benchtime=2s $(PKG)

# The Python client suite (mirrors the python-client CI job). Runs against an
# in-process FakeRostam; the cross-stack module auto-skips without a server
# binary. Install deps once with: pip install -e "clients/python[test]".
test-python:
	cd clients/python && $(PYTHON) -m pytest tests

# Build the Python client distribution (wheel + sdist) into clients/python/dist.
# Requires the PEP 517 frontend: python -m pip install build
dist-python:
	cd clients/python && rm -rf dist build && $(PYTHON) -m build

# Upload the built distribution to PyPI. Provide credentials via a ~/.pypirc or
# TWINE_USERNAME=__token__ TWINE_PASSWORD=<pypi-token>. Requires: pip install twine
# Run `make dist-python` first. Use TWINE_REPOSITORY=testpypi to dry-run on TestPyPI.
publish-python:
	cd clients/python && $(PYTHON) -m twine check dist/* && $(PYTHON) -m twine upload dist/*

tidy:
	$(GO) mod tidy

clean:
	rm -f *.test *.out *.prof
