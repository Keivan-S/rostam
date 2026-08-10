// SPDX-License-Identifier: Apache-2.0

package rlog

import (
	"context"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// grpcBearer extracts the bearer token from the "authorization" gRPC metadata
// (stripping a "Bearer " prefix), matching the grpcapi transport. Used only to
// derive the redacted principal fingerprint.
func grpcBearer(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	vals := md.Get("authorization")
	if len(vals) == 0 {
		return ""
	}
	if after, ok := strings.CutPrefix(vals[0], "Bearer "); ok {
		return after
	}
	return vals[0]
}

// grpcClientCN returns the VERIFIED mTLS client-cert CommonName from the peer's
// TLS info, or "" — the same verified-chain source the grpcapi transport uses.
func grpcClientCN(ctx context.Context) string {
	p, ok := peer.FromContext(ctx)
	if !ok || p.AuthInfo == nil {
		return ""
	}
	tlsInfo, ok := p.AuthInfo.(credentials.TLSInfo)
	if !ok {
		return ""
	}
	chains := tlsInfo.State.VerifiedChains
	if len(chains) == 0 || len(chains[0]) == 0 {
		return ""
	}
	return chains[0][0].Subject.CommonName
}

// grpcInboundRequestID reads a client-supplied request id from the
// "x-request-id" metadata, or "" if absent.
func grpcInboundRequestID(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	vals := md.Get(strings.ToLower(RequestIDHeader))
	if len(vals) == 0 {
		return ""
	}
	return vals[0]
}

// UnaryInterceptor returns a grpc.UnaryServerInterceptor that assigns a request
// id (reusing an inbound x-request-id metadata value or generating one), puts it
// in the handler context, echoes it as response metadata, and emits one access
// line per RPC with a REDACTED principal (token fingerprint or cert CN, never the
// raw token). It is only chained when -access-log is on; a nil/disabled AccessLog
// returns nil so the caller chains nothing and pays zero cost.
func (a *AccessLog) UnaryInterceptor() grpc.UnaryServerInterceptor {
	if !a.Enabled() {
		return nil
	}
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		start := time.Now()
		ctx, id := EnsureRequestID(ctx, grpcInboundRequestID(ctx))
		_ = grpc.SetHeader(ctx, metadata.Pairs(strings.ToLower(RequestIDHeader), id))
		resp, err := handler(ctx, req)
		a.Log(Entry{
			RequestID: id,
			Transport: "grpc",
			Op:        info.FullMethod,
			Status:    status.Code(err).String(),
			Latency:   time.Since(start),
			Principal: Principal(grpcBearer(ctx), grpcClientCN(ctx)),
			Bytes:     0, // gRPC marshals the response after the interceptor returns; size is not available here.
		})
		return resp, err
	}
}
