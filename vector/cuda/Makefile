# Builds libknn.a — the CUDA exact-KNN kernel linked into the -tags cuda GPU
# index (vector/gpu_cuda.go) via cgo. The DEFAULT (pure-Go, CGO_ENABLED=0,
# !cuda) build NEVER invokes this; it is only needed before a
# `CGO_ENABLED=1 go build -tags cuda ./vector/`.
#
# Usage (from the repo root):
#   make -C vector/cuda            # -> vector/cuda/libknn.a
#   CGO_ENABLED=1 go build -tags cuda ./vector/
#   CGO_ENABLED=1 go test  -tags cuda ./vector/ -run GPU -count=1
#
# The .o/.a outputs are gitignored (see vector/cuda/.gitignore); only the
# .cu/.h/Makefile are committed.

NVCC ?= nvcc
# -O2 for the distance loops; --compiler-options -fPIC so the static lib links
# into the cgo-produced position-independent objects.
NVCCFLAGS ?= -O2 --compiler-options -fPIC

AR ?= ar

libknn.a: knn.o
	$(AR) rcs $@ $<

knn.o: knn.cu knn.h
	$(NVCC) $(NVCCFLAGS) -c knn.cu -o knn.o

.PHONY: clean
clean:
	rm -f knn.o libknn.a
