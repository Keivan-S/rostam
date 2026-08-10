// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"reflect"
	"testing"
)

func TestAliasBatchArgsRoundtrip(t *testing.T) {
	in := []AliasAction{
		{Alias: "prod", Canonical: "docs_v1", Delete: false},
		{Alias: "stale", Canonical: "", Delete: true},
		{Alias: "prod", Canonical: "docs_v2", Delete: false},
	}
	args := EncodeAliasBatchArgs(in)
	got, err := DecodeAliasBatchArgs(args)
	if err != nil {
		t.Fatalf("DecodeAliasBatchArgs: %v", err)
	}
	if !reflect.DeepEqual(got, in) {
		t.Fatalf("roundtrip mismatch:\n got = %+v\nwant = %+v", got, in)
	}
}

func TestAliasBatchArgsEmpty(t *testing.T) {
	args := EncodeAliasBatchArgs(nil)
	got, err := DecodeAliasBatchArgs(args)
	if err != nil {
		t.Fatalf("DecodeAliasBatchArgs(empty): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got %d actions, want 0", len(got))
	}
}

func TestAliasCreateDeleteArgsLowerToBatch(t *testing.T) {
	create, err := DecodeAliasBatchArgs(EncodeAliasCreateArgs("prod", "docs"))
	if err != nil {
		t.Fatalf("decode create: %v", err)
	}
	want := []AliasAction{{Alias: "prod", Canonical: "docs", Delete: false}}
	if !reflect.DeepEqual(create, want) {
		t.Fatalf("create = %+v, want %+v", create, want)
	}
	del, err := DecodeAliasBatchArgs(EncodeAliasDeleteArgs("prod"))
	if err != nil {
		t.Fatalf("decode delete: %v", err)
	}
	wantDel := []AliasAction{{Alias: "prod", Canonical: "", Delete: true}}
	if !reflect.DeepEqual(del, wantDel) {
		t.Fatalf("delete = %+v, want %+v", del, wantDel)
	}
}

func TestAliasListArgsRoundtrip(t *testing.T) {
	for _, coll := range []string{"", "docs_v1"} {
		got, err := DecodeAliasListArgs(EncodeAliasListArgs(coll))
		if err != nil {
			t.Fatalf("DecodeAliasListArgs(%q): %v", coll, err)
		}
		if got != coll {
			t.Fatalf("collection = %q, want %q", got, coll)
		}
	}
}

func TestAliasListResultRoundtrip(t *testing.T) {
	in := []AliasEntry{
		{Alias: "prod", Collection: "docs_v1"},
		{Alias: "stage", Collection: "docs_v2"},
	}
	body := EncodeAliasListResult(in)
	got, err := DecodeAliasListResult(body)
	if err != nil {
		t.Fatalf("DecodeAliasListResult: %v", err)
	}
	if !reflect.DeepEqual(got, in) {
		t.Fatalf("roundtrip mismatch:\n got = %+v\nwant = %+v", got, in)
	}
}

func TestAliasBatchArgsTruncated(t *testing.T) {
	full := EncodeAliasBatchArgs([]AliasAction{{Alias: "prod", Canonical: "docs"}})
	// Every strict prefix shorter than the full encoding must fail-loud.
	for i := 0; i < len(full); i++ {
		if _, err := DecodeAliasBatchArgs(full[:i]); err == nil {
			t.Fatalf("DecodeAliasBatchArgs(prefix len=%d) = nil err, want truncation error", i)
		}
	}
}

func TestAliasListArgsTruncated(t *testing.T) {
	// A 1-byte buffer can't hold the u16 length header.
	if _, err := DecodeAliasListArgs([]byte{0x00}); err == nil {
		t.Fatal("DecodeAliasListArgs(1 byte) = nil err, want truncation error")
	}
}

func TestAliasListResultTruncated(t *testing.T) {
	full := EncodeAliasListResult([]AliasEntry{{Alias: "prod", Collection: "docs"}})
	for i := 0; i < len(full); i++ {
		if _, err := DecodeAliasListResult(full[:i]); err == nil {
			t.Fatalf("DecodeAliasListResult(prefix len=%d) = nil err, want truncation error", i)
		}
	}
}
