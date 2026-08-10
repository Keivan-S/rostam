// SPDX-License-Identifier: Apache-2.0

package analysis

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"simple", "hello world", []string{"hello", "world"}},
		{"lowercasing", "Hello WORLD FoO", []string{"hello", "world", "foo"}},
		{"punctuation", "hello, world! it's fine.", []string{"hello", "world", "it", "s", "fine"}},
		{"digits", "abc123 45 6x", []string{"abc123", "45", "6x"}},
		{"mixed separators", "a-b_c/d.e", []string{"a", "b", "c", "d", "e"}},
		{"leading/trailing seps", "  ...foo--- ", []string{"foo"}},
		{"only separators", "--- ... !!!", nil},
		{"unicode letters", "naïve café Ünïcödé", []string{"naïve", "café", "ünïcödé"}},
		{"unicode non-latin", "héllo мир 日本語", []string{"héllo", "мир", "日本語"}},
		{"runs of separators collapse", "a,,,b", []string{"a", "b"}},
		{"newlines and tabs", "a\n\tb\r\nc", []string{"a", "b", "c"}},
		{"underscores split", "snake_case_name", []string{"snake", "case", "name"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Tokenize(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Tokenize(%q) = %#v, want %#v", tc.in, got, tc.want)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	tests := map[string]string{
		"":          "",
		"foo":       "foo",
		"FOO":       "foo",
		"FooBar":    "foobar",
		"café":      "café",
		"CAFÉ":      "café",
		"Ünïcödé":   "ünïcödé",
		"already12": "already12",
	}
	for in, want := range tests {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}
