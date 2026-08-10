// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"crypto/x509"
	"crypto/x509/pkix"
	"strings"
	"testing"
)

// certWithCN returns a minimal *x509.Certificate carrying only a Subject
// CommonName — enough to exercise the peer-CN verify callback, which reads
// verifiedChains[0][0].Subject.CommonName.
func certWithCN(cn string) *x509.Certificate {
	return &x509.Certificate{Subject: pkix.Name{CommonName: cn}}
}

// TestPeerCNVerifierAcceptReject exercises the inter-node CLIENT peer-CN verify
// callback (peerCNVerifier) directly: it accepts a peer whose verified leaf-cert
// CN is in the allowlist and FAILS the handshake (returns an error) for an
// unlisted CN or an absent/empty verified chain.
func TestPeerCNVerifierAcceptReject(t *testing.T) {
	allow := map[string]bool{"n1": true, "n2": true}
	verify := peerCNVerifier(allow)

	// Accept: a verified chain whose leaf CN is allowlisted.
	chain := [][]*x509.Certificate{{certWithCN("n2"), certWithCN("root-ca")}}
	if err := verify(nil, chain); err != nil {
		t.Errorf("allowlisted peer CN n2 must be accepted, got error: %v", err)
	}

	// Reject: leaf CN not in the allowlist (the security property).
	badChain := [][]*x509.Certificate{{certWithCN("rogue")}}
	if err := verify(nil, badChain); err == nil {
		t.Error("non-allowlisted peer CN must be rejected (handshake fail)")
	} else if !strings.Contains(err.Error(), "rogue") {
		t.Errorf("reject error should name the rejected CN, got: %v", err)
	}

	// Reject: empty leaf CN (a cert with no CommonName).
	if err := verify(nil, [][]*x509.Certificate{{certWithCN("")}}); err == nil {
		t.Error("empty peer CN must be rejected")
	}

	// Reject fail-loud: no verified chain at all.
	if err := verify(nil, nil); err == nil {
		t.Error("nil verified chain must be rejected")
	}
	if err := verify(nil, [][]*x509.Certificate{}); err == nil {
		t.Error("empty verified chains must be rejected")
	}
	// Reject fail-loud: a chain entry with no certs.
	if err := verify(nil, [][]*x509.Certificate{{}}); err == nil {
		t.Error("verified chain with no leaf cert must be rejected")
	}
}
