// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"errors"
	"sync"
	"time"

	hraft "github.com/hashicorp/raft"
)

// maxInFlight follows hashicorp's DefaultMaxRPCsInFlight semantics: the channel
// buffers are sized maxInFlight-2 because a request is sent before it blocks on
// the channel and the decode goroutine unblocks the channel as soon as it waits
// on the first request — so even a zero-length buffer keeps two requests in
// flight.
const maxInFlight = hraft.DefaultMaxRPCsInFlight

var errPipelineShutdown = errors.New("fabric: append pipeline closed")

// pipeline implements hraft.AppendPipeline over the shared peer link. Unlike
// hashicorp's per-conn in-order pipeline, responses correlate by reqID (groups
// interleave on the shared conn). A single decode goroutine drains inprogressCh
// in submission order and waits on each future's own result channel, so Consumer
// yields futures in submission order regardless of wire arrival order.
type pipeline struct {
	gt   *groupTransport
	link *peerLink

	inprogressCh chan *appendFuture
	doneCh       chan hraft.AppendFuture

	shutdownCh   chan struct{}
	shutdownOnce sync.Once
}

func newPipeline(gt *groupTransport, link *peerLink) *pipeline {
	p := &pipeline{
		gt:           gt,
		link:         link,
		inprogressCh: make(chan *appendFuture, maxInFlight-2),
		doneCh:       make(chan hraft.AppendFuture, maxInFlight-2),
		shutdownCh:   make(chan struct{}),
	}
	go p.decodeResponses()
	return p
}

// AppendEntries implements hraft.AppendPipeline. It allocates a reqID, submits
// the frame (registering a pending waiter), and hands the future to the decode
// goroutine — blocking on inprogressCh capacity as back-pressure.
func (p *pipeline) AppendEntries(args *hraft.AppendEntriesRequest, resp *hraft.AppendEntriesResponse) (hraft.AppendFuture, error) {
	fr := &frame{
		kind:    rpcAppendEntries,
		groupID: p.gt.g.id,
		reqID:   p.gt.fabric.nextReqID(),
		payload: encodeAppendEntriesRequest(getPayload(), args),
		pooled:  true,
	}
	ch, err := p.link.submit(fr)
	if err != nil {
		return nil, err
	}
	fut := &appendFuture{start: time.Now(), args: args, resp: resp, resultCh: ch}
	fut.init()
	select {
	case p.inprogressCh <- fut:
		return fut, nil
	case <-p.shutdownCh:
		return nil, errPipelineShutdown
	}
}

// Consumer implements hraft.AppendPipeline.
func (p *pipeline) Consumer() <-chan hraft.AppendFuture { return p.doneCh }

// Close implements hraft.AppendPipeline. It stops the decode goroutine; the
// shared link stays up for other traffic. In-flight futures are abandoned (Raft
// stops consuming after Close).
func (p *pipeline) Close() error {
	p.shutdownOnce.Do(func() { close(p.shutdownCh) })
	return nil
}

// decodeResponses drains futures in submission order, waiting on each one's
// correlated result before decoding and forwarding it to Consumer.
func (p *pipeline) decodeResponses() {
	for {
		select {
		case fut := <-p.inprogressCh:
			select {
			case res := <-fut.resultCh:
				if res.err != nil {
					fut.respond(res.err)
				} else {
					appErr, derr := decodeAppendEntriesResponse(res.payload, fut.resp)
					switch {
					case derr != nil:
						fut.respond(derr)
					case appErr != "":
						fut.respond(errors.New(appErr))
					default:
						fut.respond(nil)
					}
				}
			case <-p.shutdownCh:
				return
			}
			select {
			case p.doneCh <- fut:
			case <-p.shutdownCh:
				return
			}
		case <-p.shutdownCh:
			return
		}
	}
}

// appendFuture implements hraft.AppendFuture. resultCh is the peer link's pending
// channel for this request's reqID.
type appendFuture struct {
	start    time.Time
	args     *hraft.AppendEntriesRequest
	resp     *hraft.AppendEntriesResponse
	resultCh chan result

	err   error
	errCh chan struct{}
	once  sync.Once
}

func (f *appendFuture) init() { f.errCh = make(chan struct{}) }

func (f *appendFuture) respond(err error) {
	f.once.Do(func() {
		f.err = err
		close(f.errCh)
	})
}

// Error blocks until the response (or an error) has landed.
func (f *appendFuture) Error() error {
	<-f.errCh
	return f.err
}

func (f *appendFuture) Start() time.Time                       { return f.start }
func (f *appendFuture) Request() *hraft.AppendEntriesRequest   { return f.args }
func (f *appendFuture) Response() *hraft.AppendEntriesResponse { return f.resp }
