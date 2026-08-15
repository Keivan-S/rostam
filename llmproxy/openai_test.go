// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"encoding/json"
	"testing"
)

func TestParseChatRequest_StringContent(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}
	if req.Model != "gpt-4" {
		t.Fatalf("Model = %q, want gpt-4", req.Model)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(req.Messages))
	}

	_, _, ok := req.cacheIdentity("")
	if !ok {
		t.Fatalf("cacheIdentity ok = false, want true for string content")
	}
}

func TestParseChatRequest_StashesRawBody(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}
	if string(req.Raw) != string(body) {
		t.Fatalf("Raw = %q, want %q", req.Raw, body)
	}
}

func TestCacheIdentity_ArrayContentUncacheable(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}

	_, _, ok := req.cacheIdentity("")
	if ok {
		t.Fatalf("cacheIdentity ok = true, want false for array content")
	}
}

func TestTemperature_NilDefaultsToOne(t *testing.T) {
	req := &chatRequest{}
	if got := req.temperature(); got != 1.0 {
		t.Fatalf("temperature() = %v, want 1.0", got)
	}
}

func TestTemperature_ExplicitZeroStaysZero(t *testing.T) {
	zero := 0.0
	req := &chatRequest{Temperature: &zero}
	if got := req.temperature(); got != 0.0 {
		t.Fatalf("temperature() = %v, want 0.0", got)
	}
}

func TestTemperature_ExplicitValuePreserved(t *testing.T) {
	half := 0.5
	req := &chatRequest{Temperature: &half}
	if got := req.temperature(); got != 0.5 {
		t.Fatalf("temperature() = %v, want 0.5", got)
	}
}

func TestCacheIdentity_PromptSerializationExact(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[
		{"role":"system","content":"be terse"},
		{"role":"user","content":"hello"},
		{"role":"assistant","content":"hi there"},
		{"role":"user","content":"bye"}
	]}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}

	prompt, scope, ok := req.cacheIdentity("")
	if !ok {
		t.Fatalf("cacheIdentity ok = false, want true")
	}

	want := "user\x00hello\x1eassistant\x00hi there\x1euser\x00bye\x1e"
	if prompt != want {
		t.Fatalf("prompt = %q, want %q", prompt, want)
	}
	if scope.System != "be terse" {
		t.Fatalf("Scope.System = %q, want %q", scope.System, "be terse")
	}
}

func TestCacheIdentity_SystemMessagesExcludedAndJoined(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[
		{"role":"system","content":"first"},
		{"role":"user","content":"q"},
		{"role":"system","content":"second"}
	]}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}

	prompt, scope, ok := req.cacheIdentity("")
	if !ok {
		t.Fatalf("cacheIdentity ok = false, want true")
	}

	wantPrompt := "user\x00q\x1e"
	if prompt != wantPrompt {
		t.Fatalf("prompt = %q, want %q", prompt, wantPrompt)
	}

	wantSystem := "first\nsecond"
	if scope.System != wantSystem {
		t.Fatalf("Scope.System = %q, want %q", scope.System, wantSystem)
	}
}

func TestCacheIdentity_ToolNamesExtracted(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],
		"tools":[{"function":{"name":"get_weather"}},{"function":{"name":"search"}}]}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}

	_, scope, ok := req.cacheIdentity("")
	if !ok {
		t.Fatalf("cacheIdentity ok = false, want true")
	}

	want := []string{"get_weather", "search"}
	if len(scope.Tools) != len(want) {
		t.Fatalf("Tools = %v, want %v", scope.Tools, want)
	}
	for i := range want {
		if scope.Tools[i] != want[i] {
			t.Fatalf("Tools[%d] = %q, want %q", i, scope.Tools[i], want[i])
		}
	}
}

func TestCacheIdentity_NGreaterThanOneUncacheable(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"n":2}`)
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}

	_, _, ok := req.cacheIdentity("")
	if ok {
		t.Fatalf("cacheIdentity ok = true, want false when n=2")
	}
}

func TestCacheIdentity_ScopeFieldsPopulated(t *testing.T) {
	half := 0.5
	body, err := json.Marshal(chatRequest{
		Model:       "gpt-4o",
		Messages:    []chatMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		Temperature: &half,
		MaxTokens:   256,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := parseChatRequest(body)
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}

	_, scope, ok := req.cacheIdentity("tenant-a")
	if !ok {
		t.Fatalf("cacheIdentity ok = false, want true")
	}
	if scope.Model != "gpt-4o" {
		t.Fatalf("Scope.Model = %q, want gpt-4o", scope.Model)
	}
	if scope.Temperature != 0.5 {
		t.Fatalf("Scope.Temperature = %v, want 0.5", scope.Temperature)
	}
	if scope.MaxTokens != 256 {
		t.Fatalf("Scope.MaxTokens = %d, want 256", scope.MaxTokens)
	}
	if scope.Tenant != "tenant-a" {
		t.Fatalf("Scope.Tenant = %q, want tenant-a", scope.Tenant)
	}
}

func TestCacheIdentity_StreamOptionsUncacheable(t *testing.T) {
	for name, body := range map[string]string{
		"include_usage": `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`,
		"empty object":  `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{}}`,
	} {
		req, err := parseChatRequest([]byte(body))
		if err != nil {
			t.Fatalf("%s: parseChatRequest: %v", name, err)
		}
		if _, _, ok := req.cacheIdentity(""); ok {
			t.Errorf("%s: cacheIdentity ok = true, want false (a replay can't reproduce the requested stream shape)", name)
		}
	}

	// An explicit null is the same as absent — several OpenAI-compatible
	// backends send it — and must stay cacheable.
	req, err := parseChatRequest([]byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream_options":null}`))
	if err != nil {
		t.Fatalf("parseChatRequest: %v", err)
	}
	if _, _, ok := req.cacheIdentity(""); !ok {
		t.Fatalf("cacheIdentity ok = false for stream_options:null, want true")
	}
}

// extraOf is a test shorthand: parse a body and return the Extra
// discriminator its scope carries.
func extraOf(t *testing.T, body string) string {
	t.Helper()
	req, err := parseChatRequest([]byte(body))
	if err != nil {
		t.Fatalf("parseChatRequest(%s): %v", body, err)
	}
	_, scope, ok := req.cacheIdentity("")
	if !ok {
		t.Fatalf("cacheIdentity ok = false for %s", body)
	}
	return scope.Extra
}

func TestCacheIdentity_ExtraPartitionsOnSamplingParams(t *testing.T) {
	const base = `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	baseExtra := extraOf(t, base)

	for name, body := range map[string]string{
		"response_format":   `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`,
		"seed":              `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"seed":7}`,
		"top_p":             `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"top_p":0.1}`,
		"stop":              `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stop":["\n"]}`,
		"frequency_penalty": `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"frequency_penalty":1.5}`,
	} {
		if got := extraOf(t, body); got == baseExtra {
			t.Errorf("%s: Extra = %q, same as the base request — the two must not share a cache entry", name, got)
		}
	}
}

func TestCacheIdentity_ExtraIgnoresKeyOrderAndPromptOnlyFields(t *testing.T) {
	// Byte-different, semantically equal: reordered keys and extra whitespace.
	a := extraOf(t, `{"model":"gpt-4","seed":7,"messages":[{"role":"user","content":"hi"}],"top_p":0.5}`)
	b := extraOf(t, `{ "top_p": 0.5, "messages":[{"role":"user","content":"hi"}], "model":"gpt-4",  "seed":7 }`)
	if a != b {
		t.Fatalf("Extra = %q vs %q, want equal (key order and whitespace are not request surface)", a, b)
	}

	// stream and messages are deliberately excluded: the streaming and
	// non-streaming forms of one call share a cached answer, and the messages
	// ARE the prompt.
	plain := extraOf(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	streamed := extraOf(t, `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if plain != streamed {
		t.Fatalf("Extra = %q (plain) vs %q (stream:true), want equal", plain, streamed)
	}
	differentPrompt := extraOf(t, `{"model":"gpt-4","messages":[{"role":"user","content":"something else"}]}`)
	if plain != differentPrompt {
		t.Fatalf("Extra = %q vs %q, want equal (messages belong to the prompt, not the discriminator)", plain, differentPrompt)
	}
}

func TestSynthesizeResponse_RoundTrips(t *testing.T) {
	body := synthesizeResponse("gpt-4", "the answer", 42)

	var resp chatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("len(Choices) = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "the answer" {
		t.Fatalf("Content = %q, want %q", resp.Choices[0].Message.Content, "the answer")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.CompletionTokens != 42 {
		t.Fatalf("CompletionTokens = %d, want 42", resp.Usage.CompletionTokens)
	}

	// Also check the raw JSON shape for the fields not carried by chatResponse.
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("Unmarshal raw: %v", err)
	}
	if raw["id"] != "rostam-cache" {
		t.Fatalf("id = %v, want rostam-cache", raw["id"])
	}
	if raw["object"] != "chat.completion" {
		t.Fatalf("object = %v, want chat.completion", raw["object"])
	}
	if raw["model"] != "gpt-4" {
		t.Fatalf("model = %v, want gpt-4", raw["model"])
	}
	if created, ok := raw["created"].(float64); !ok || created == 0 {
		t.Fatalf("created = %v, want a non-zero Unix timestamp", raw["created"])
	}
	usage, _ := raw["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(0) {
		t.Fatalf("prompt_tokens = %v, want 0", usage["prompt_tokens"])
	}
	if usage["total_tokens"] != float64(42) {
		t.Fatalf("total_tokens = %v, want 42", usage["total_tokens"])
	}
}

func TestTenantOf_EmptyHeadersYieldEmptyTenant(t *testing.T) {
	if got := tenantOf("", "", ""); got != "" {
		t.Fatalf("tenantOf with no credential headers = %q, want empty", got)
	}
}

func TestTenantOf_DifferingHeadersDiffer(t *testing.T) {
	a := tenantOf("Bearer sk-aaaa", "", "")
	b := tenantOf("Bearer sk-bbbb", "", "")
	if a == "" || b == "" {
		t.Fatalf("tenantOf returned empty for non-empty header: a=%q b=%q", a, b)
	}
	if a == b {
		t.Fatalf("tenantOf(a) == tenantOf(b) = %q, want distinct hashes", a)
	}
}

func TestTenantOf_SameHeaderSameTenant(t *testing.T) {
	a := tenantOf("Bearer sk-aaaa", "", "")
	b := tenantOf("Bearer sk-aaaa", "", "")
	if a != b {
		t.Fatalf("tenantOf(same) = %q vs %q, want equal", a, b)
	}
}

// Azure OpenAI sends "api-key" and Anthropic-style surfaces send "x-api-key";
// neither sets Authorization. Hashing only Authorization put all of them in
// the one empty "no auth" tenant, so every such caller shared a cache.
func TestTenantOf_AlternateCredentialHeadersIsolate(t *testing.T) {
	for name, pair := range map[string][2]string{
		"api-key":   {"azure-key-a", "azure-key-b"},
		"x-api-key": {"anthropic-key-a", "anthropic-key-b"},
	} {
		var a, b string
		if name == "api-key" {
			a, b = tenantOf("", pair[0], ""), tenantOf("", pair[1], "")
		} else {
			a, b = tenantOf("", "", pair[0]), tenantOf("", "", pair[1])
		}
		if a == "" || b == "" {
			t.Errorf("%s: tenantOf returned empty for a set credential header: a=%q b=%q", name, a, b)
		}
		if a == b {
			t.Errorf("%s: two different keys share tenant %q", name, a)
		}
	}

	// The three headers must not be interchangeable: the same secret in a
	// different header is a different caller.
	if tenantOf("k", "", "") == tenantOf("", "k", "") {
		t.Fatal("Authorization and api-key with the same value share a tenant")
	}
	if tenantOf("", "k", "") == tenantOf("", "", "k") {
		t.Fatal("api-key and x-api-key with the same value share a tenant")
	}
}
