// SPDX-License-Identifier: Apache-2.0

// Package client implements the Rostam TCP client with a chpool-style
// connection pool, NotLeader auto-retry, and ping-on-stale validation.
package client

import (
	"encoding/binary"
	"errors"
)

// These constants and helpers MIRROR server/protocol.go intentionally.
// Duplication keeps the client free of any server-side imports
// (which transitively pull cache/raft/shard).

// Status codes carried in the response frame.
const (
	StatusOK           uint8 = 0
	StatusNotFound     uint8 = 1
	StatusNotLeader    uint8 = 2
	StatusError        uint8 = 3
	StatusUnauthorized uint8 = 4
)

// MaxFrameSize bounds each individual frame.
const MaxFrameSize = 16 << 20

// maxOpNameLen is the maximum length of an operation name.
const maxOpNameLen = 255

// ErrFrameTruncated indicates a frame buffer is shorter than its header claims.
var ErrFrameTruncated = errors.New("client: frame truncated")

// ErrFrameTooLarge indicates a frame exceeds MaxFrameSize.
var ErrFrameTooLarge = errors.New("client: frame too large")

// encodeRequestFrame builds one complete request frame into a fresh []byte —
// the same wire the inline Conn.doCall writer emits, but materialized so the
// pipelined connection (client/pipeline.go) can write it under its send lock
// without replicating the multi-step framing. v1 (no token):
//
//	[frameLen u32][opNameLen u8][opName][argsLen u32][args]
//
// v2 (authToken set): frameLen then [version=2 u8][tokenLen u8][token] before
// the opName. Errors on an over-long op name / token or an over-size frame.
func encodeRequestFrame(authToken, op string, args []byte) ([]byte, error) {
	if len(op) > maxOpNameLen {
		return nil, ErrFrameTooLarge
	}
	v2 := 0
	if authToken != "" {
		if len(authToken) > 255 {
			return nil, ErrFrameTooLarge
		}
		v2 = 1 + 1 + len(authToken)
	}
	bodyLen := v2 + 1 + len(op) + 4 + len(args)
	if bodyLen > MaxFrameSize {
		return nil, ErrFrameTooLarge
	}
	buf := make([]byte, 4+bodyLen)
	binary.BigEndian.PutUint32(buf[0:4], uint32(bodyLen)) //nolint:gosec // bounded above
	p := 4
	if authToken != "" {
		buf[p] = 0x02 // ProtocolV2
		buf[p+1] = byte(len(authToken))
		p += 2
		p += copy(buf[p:], authToken)
	}
	buf[p] = byte(len(op))
	p++
	p += copy(buf[p:], op)
	binary.BigEndian.PutUint32(buf[p:p+4], uint32(len(args))) //nolint:gosec // bounded above
	p += 4
	copy(buf[p:], args)
	return buf, nil
}

// decodeResponse reads a response body. Wire: [status u8][payloadLen u32][payload].
func decodeResponse(frame []byte) (status uint8, payload []byte, err error) {
	if len(frame) < 1+4 {
		return 0, nil, ErrFrameTruncated
	}
	status = frame[0]
	payloadLen := int(binary.BigEndian.Uint32(frame[1:5]))
	if payloadLen > MaxFrameSize {
		return 0, nil, ErrFrameTooLarge
	}
	if len(frame) < 5+payloadLen {
		return 0, nil, ErrFrameTruncated
	}
	payload = frame[5 : 5+payloadLen]
	return status, payload, nil
}

// decodeLeaderAddr extracts the leader address from a StatusNotLeader payload.
func decodeLeaderAddr(payload []byte) (string, error) {
	if len(payload) < 2 {
		return "", ErrFrameTruncated
	}
	n := int(binary.BigEndian.Uint16(payload[0:2]))
	if len(payload) < 2+n {
		return "", ErrFrameTruncated
	}
	return string(payload[2 : 2+n]), nil
}

// decodeErrorMsg extracts a server-side error message from a StatusError payload.
func decodeErrorMsg(payload []byte) (string, error) {
	if len(payload) < 2 {
		return "", ErrFrameTruncated
	}
	n := int(binary.BigEndian.Uint16(payload[0:2]))
	if len(payload) < 2+n {
		return "", ErrFrameTruncated
	}
	return string(payload[2 : 2+n]), nil
}
