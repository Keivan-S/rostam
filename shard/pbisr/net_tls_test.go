// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"crypto/tls"
	"testing"

	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
)

// pbTLSConfigs builds the strict-mTLS server + client configs for a node whose
// identity cert CN is nodeCN (both EKUs, loopback SANs) signed off ca — mirroring
// how cmd/rostam-server wires the PB replication transport.
func pbTLSConfigs(t *testing.T, ca *testcerts.CA, nodeCN string) (server, client *tls.Config) {
	t.Helper()
	certFile, keyFile := ca.NodeCert(t, nodeCN)
	server, err := tlsutil.ServerTLS(certFile, keyFile, ca.CAFile, true)
	if err != nil {
		t.Fatalf("ServerTLS: %v", err)
	}
	client, err = tlsutil.ClientTLS(ca.CAFile, certFile, keyFile, "")
	if err != nil {
		t.Fatalf("ClientTLS: %v", err)
	}
	return server, client
}

// TestPBTLSReplicateRoundTrip: with matching-CA strict mTLS and an allowlisted
// client CN, a real replicate request travels the TLS pipelined link to the backup
// and its ack returns — the happy-path round trip over encrypted, authenticated
// inter-node PB replication.
func TestPBTLSReplicateRoundTrip(t *testing.T) {
	const shard = 5
	ca := testcerts.GenCA(t)
	allow := map[string]bool{"n1": true, "n2": true}

	srvS, srvC := pbTLSConfigs(t, ca, "n2") // backup identity n2
	backup, err := NewNetTransport("127.0.0.1:0", srvS, srvC, allow)
	if err != nil {
		t.Fatalf("backup NewNetTransport: %v", err)
	}
	defer backup.Close()
	rcv := &captureReceiver{}
	backup.Register(shard, rcv)

	_, cliC := pbTLSConfigs(t, ca, "n1") // primary identity n1 (allowlisted)
	primary, err := NewNetTransport("127.0.0.1:0", nil, cliC, allow)
	if err != nil {
		t.Fatalf("primary NewNetTransport: %v", err)
	}
	defer primary.Close()

	ack, err := syncReplicate(primary.For(shard), backup.Addr(), ReplicateMsg{Epoch: 2, Seq: 3, PrevSeq: 2, Data: []byte("x")})
	if err != nil {
		t.Fatalf("replicate over mTLS: %v", err)
	}
	if !ack.OK || ack.Epoch != 2 || ack.Seq != 3 {
		t.Fatalf("ack: %+v", ack)
	}
	if last := rcv.Last(); last.Seq != 3 || string(last.Data) != "x" {
		t.Fatalf("backup receiver saw: %+v", last)
	}
}

// TestPBPlaintextDialToTLSListenerFails: a plaintext primary dialing a TLS backup
// must fail — the backup's handshake rejects it, no silent plaintext fallback.
func TestPBPlaintextDialToTLSListenerFails(t *testing.T) {
	const shard = 5
	ca := testcerts.GenCA(t)
	srvS, srvC := pbTLSConfigs(t, ca, "n2")
	backup, err := NewNetTransport("127.0.0.1:0", srvS, srvC, nil)
	if err != nil {
		t.Fatalf("backup NewNetTransport: %v", err)
	}
	defer backup.Close()
	backup.Register(shard, &captureReceiver{})

	// Plaintext primary (nil clientTLS): its dial connects but the frame bytes are
	// rejected at the backup's TLS handshake, failing the replicate.
	primary, err := NewNetTransport("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		t.Fatalf("primary NewNetTransport: %v", err)
	}
	defer primary.Close()

	if _, err := syncReplicate(primary.For(shard), backup.Addr(), ReplicateMsg{Epoch: 1, Seq: 1, Data: []byte("x")}); err == nil {
		t.Fatal("plaintext dial to a TLS PB listener must fail, but replicate succeeded")
	}
}

// TestPBNonAllowlistedCNRejected: a CA-valid primary whose CN is NOT allowlisted
// completes the handshake but is dropped in serveConn before any replicate frame
// reaches a receiver, so the replicate fails.
func TestPBNonAllowlistedCNRejected(t *testing.T) {
	const shard = 5
	ca := testcerts.GenCA(t)
	srvS, srvC := pbTLSConfigs(t, ca, "n2")
	backup, err := NewNetTransport("127.0.0.1:0", srvS, srvC, map[string]bool{"n1": true, "n2": true})
	if err != nil {
		t.Fatalf("backup NewNetTransport: %v", err)
	}
	defer backup.Close()
	rcv := &captureReceiver{}
	backup.Register(shard, rcv)

	_, cliC := pbTLSConfigs(t, ca, "rogue") // CA-valid but CN not allowlisted
	primary, err := NewNetTransport("127.0.0.1:0", nil, cliC, nil)
	if err != nil {
		t.Fatalf("primary NewNetTransport: %v", err)
	}
	defer primary.Close()

	if _, err := syncReplicate(primary.For(shard), backup.Addr(), ReplicateMsg{Epoch: 1, Seq: 1, Data: []byte("x")}); err == nil {
		t.Fatal("a non-allowlisted CN must be rejected before any replicate frame is served")
	}
	if last := rcv.Last(); last.Seq != 0 || last.Data != nil {
		t.Fatalf("rejected peer must not reach the receiver, but it saw: %+v", last)
	}
}

// TestPBWrongCARejected: a primary signed by a DIFFERENT CA cannot verify the
// backup's cert (nor present a cert the backup trusts), so the dial handshake
// fails and the replicate errors.
func TestPBWrongCARejected(t *testing.T) {
	const shard = 5
	serverCA := testcerts.GenCA(t)
	otherCA := testcerts.GenCA(t)
	srvS, srvC := pbTLSConfigs(t, serverCA, "n2")
	backup, err := NewNetTransport("127.0.0.1:0", srvS, srvC, nil)
	if err != nil {
		t.Fatalf("backup NewNetTransport: %v", err)
	}
	defer backup.Close()
	backup.Register(shard, &captureReceiver{})

	_, cliC := pbTLSConfigs(t, otherCA, "n1") // wrong CA
	primary, err := NewNetTransport("127.0.0.1:0", nil, cliC, nil)
	if err != nil {
		t.Fatalf("primary NewNetTransport: %v", err)
	}
	defer primary.Close()

	if _, err := syncReplicate(primary.For(shard), backup.Addr(), ReplicateMsg{Epoch: 1, Seq: 1, Data: []byte("x")}); err == nil {
		t.Fatal("a primary signed by a different CA must be rejected at the handshake")
	}
}

// TestPBNilConfigPlaintextUnchanged proves the default (nil/nil/nil) path is a
// plain-TCP transport: a plaintext replicate round-trips exactly as before.
func TestPBNilConfigPlaintextUnchanged(t *testing.T) {
	const shard = 5
	backup, err := NewNetTransport("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		t.Fatalf("backup NewNetTransport: %v", err)
	}
	defer backup.Close()
	rcv := &captureReceiver{}
	backup.Register(shard, rcv)

	primary, err := NewNetTransport("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		t.Fatalf("primary NewNetTransport: %v", err)
	}
	defer primary.Close()

	ack, err := syncReplicate(primary.For(shard), backup.Addr(), ReplicateMsg{Epoch: 1, Seq: 1, Data: []byte("x")})
	if err != nil {
		t.Fatalf("plaintext replicate: %v", err)
	}
	if !ack.OK {
		t.Fatalf("plaintext ack not OK: %+v", ack)
	}
}
