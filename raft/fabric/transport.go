// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"bufio"
	"errors"
	"io"
	"time"

	hraft "github.com/hashicorp/raft"
)

// groupTransport is the per-group facade returned by Fabric.For. It implements
// hraft.Transport, hraft.WithPreVote, and hraft.WithClose, tagging every
// outbound frame with its group ID and sharing the Fabric's listener, link
// manager, reqID counter, and per-link pending map.
type groupTransport struct {
	fabric  *Fabric
	g       *group
	timeout time.Duration
}

var (
	_ hraft.Transport   = (*groupTransport)(nil)
	_ hraft.WithPreVote = (*groupTransport)(nil)
	_ hraft.WithClose   = (*groupTransport)(nil)
)

// Consumer implements hraft.Transport.
func (t *groupTransport) Consumer() <-chan hraft.RPC { return t.g.consumeCh }

// LocalAddr implements hraft.Transport. Every group on a node shares the one
// listener address (distinguished by Raft server ID), matching the mux path.
func (t *groupTransport) LocalAddr() hraft.ServerAddress {
	return hraft.ServerAddress(t.fabric.localAddr)
}

// AppendEntries implements hraft.Transport (synchronous over the shared link).
func (t *groupTransport) AppendEntries(_ hraft.ServerID, target hraft.ServerAddress, args *hraft.AppendEntriesRequest, resp *hraft.AppendEntriesResponse) error {
	fr := &frame{
		kind:    rpcAppendEntries,
		groupID: t.g.id,
		reqID:   t.fabric.nextReqID(),
		payload: encodeAppendEntriesRequest(getPayload(), args),
		pooled:  true,
	}
	out, err := t.fabric.roundTrip(string(target), fr, t.timeout)
	if err != nil {
		return err
	}
	appErr, err := decodeAppendEntriesResponse(out, resp)
	if err != nil {
		return err
	}
	if appErr != "" {
		return errors.New(appErr)
	}
	return nil
}

// RequestVote implements hraft.Transport.
func (t *groupTransport) RequestVote(_ hraft.ServerID, target hraft.ServerAddress, args *hraft.RequestVoteRequest, resp *hraft.RequestVoteResponse) error {
	fr := &frame{
		kind:    rpcRequestVote,
		groupID: t.g.id,
		reqID:   t.fabric.nextReqID(),
		payload: encodeRequestVoteRequest(getPayload(), args),
		pooled:  true,
	}
	out, err := t.fabric.roundTrip(string(target), fr, t.timeout)
	if err != nil {
		return err
	}
	appErr, err := decodeRequestVoteResponse(out, resp)
	if err != nil {
		return err
	}
	if appErr != "" {
		return errors.New(appErr)
	}
	return nil
}

// RequestPreVote implements hraft.WithPreVote.
func (t *groupTransport) RequestPreVote(_ hraft.ServerID, target hraft.ServerAddress, args *hraft.RequestPreVoteRequest, resp *hraft.RequestPreVoteResponse) error {
	fr := &frame{
		kind:    rpcRequestPreVote,
		groupID: t.g.id,
		reqID:   t.fabric.nextReqID(),
		payload: encodeRequestPreVoteRequest(getPayload(), args),
		pooled:  true,
	}
	out, err := t.fabric.roundTrip(string(target), fr, t.timeout)
	if err != nil {
		return err
	}
	appErr, err := decodeRequestPreVoteResponse(out, resp)
	if err != nil {
		return err
	}
	if appErr != "" {
		return errors.New(appErr)
	}
	return nil
}

// TimeoutNow implements hraft.Transport.
func (t *groupTransport) TimeoutNow(_ hraft.ServerID, target hraft.ServerAddress, args *hraft.TimeoutNowRequest, resp *hraft.TimeoutNowResponse) error {
	fr := &frame{
		kind:    rpcTimeoutNow,
		groupID: t.g.id,
		reqID:   t.fabric.nextReqID(),
		payload: encodeTimeoutNowRequest(getPayload(), args),
		pooled:  true,
	}
	out, err := t.fabric.roundTrip(string(target), fr, t.timeout)
	if err != nil {
		return err
	}
	appErr, err := decodeTimeoutNowResponse(out, resp)
	if err != nil {
		return err
	}
	if appErr != "" {
		return errors.New(appErr)
	}
	return nil
}

// AppendEntriesPipeline implements hraft.Transport. It returns a pipeline over
// the shared link that correlates responses by reqID (see pipeline.go).
func (t *groupTransport) AppendEntriesPipeline(_ hraft.ServerID, target hraft.ServerAddress) (hraft.AppendPipeline, error) {
	link := t.fabric.getLink(string(target))
	return newPipeline(t, link), nil
}

// InstallSnapshot implements hraft.Transport. It dials a dedicated one-shot conn,
// writes the request frame, streams the snapshot body, and reads the response —
// so a large snapshot never head-of-line-blocks the shared mux link.
func (t *groupTransport) InstallSnapshot(_ hraft.ServerID, target hraft.ServerAddress, args *hraft.InstallSnapshotRequest, resp *hraft.InstallSnapshotResponse, data io.Reader) error {
	conn, err := t.fabric.dialConn(string(target), dialTimeout)
	if err != nil {
		return err
	}
	defer conn.Close() //nolint:errcheck // one-shot conn

	// Deadline scaled by snapshot size, mirroring hashicorp's net_transport.
	if t.timeout > 0 {
		scaled := t.timeout * time.Duration(args.Size/int64(hraft.DefaultTimeoutScale))
		if scaled < t.timeout {
			scaled = t.timeout
		}
		_ = conn.SetDeadline(time.Now().Add(scaled))
	}

	if _, err := conn.Write([]byte{connSnapshot}); err != nil {
		return err
	}
	bw := bufio.NewWriterSize(conn, writeBufSize)
	fr := &frame{
		kind:    rpcInstallSnapshot,
		groupID: t.g.id,
		reqID:   t.fabric.nextReqID(),
		payload: encodeInstallSnapshotRequest(nil, args),
	}
	if err := writeFrame(bw, fr); err != nil {
		return err
	}
	if _, err := io.Copy(bw, data); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}

	rd := frameReader{r: bufio.NewReaderSize(conn, readBufSize)}
	msg, err := rd.read()
	if err != nil {
		return err
	}
	appErr, err := decodeInstallSnapshotResponse(msg.payload, resp)
	if err != nil {
		return err
	}
	if appErr != "" {
		return errors.New(appErr)
	}
	return nil
}

// EncodePeer implements hraft.Transport (raw address bytes, like hashicorp).
func (t *groupTransport) EncodePeer(_ hraft.ServerID, addr hraft.ServerAddress) []byte {
	return []byte(addr)
}

// DecodePeer implements hraft.Transport.
func (t *groupTransport) DecodePeer(buf []byte) hraft.ServerAddress {
	return hraft.ServerAddress(buf)
}

// SetHeartbeatHandler implements hraft.Transport.
func (t *groupTransport) SetHeartbeatHandler(cb func(rpc hraft.RPC)) {
	t.g.hbMu.Lock()
	t.g.hbHandler = cb
	t.g.hbMu.Unlock()
}

// Close implements hraft.WithClose. It closes only this group (stopping Consumer
// delivery); the shared Fabric listener and peer links stay up for the other
// groups and are torn down by Fabric.Close. This mirrors the mux path, where a
// per-group NetworkTransport.Close leaves the shared listener alone.
func (t *groupTransport) Close() error {
	t.g.closeOnce.Do(func() { close(t.g.closed) })
	return nil
}
