// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/rostamlabs/rostam/semcache"
)

const systemPrompt = "You answer questions using ONLY the numbered context passages provided. " +
	"For each passage you rely on, cite its bracketed number in the form [n] (e.g. [1]). " +
	"If the context does not contain the answer, say you don't know."

// BuildPrompt returns the system+user messages: numbered context then question.
func BuildPrompt(question string, hits []Hit) (string, string) {
	var b strings.Builder
	b.WriteString("Context:\n")
	for i, h := range hits {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, h.Content)
	}
	if len(hits) == 0 {
		b.WriteString("(no relevant context found)\n")
	}
	fmt.Fprintf(&b, "\nQuestion: %s", question)
	return systemPrompt, b.String()
}

// LLMConfig configures an OpenAI-compatible chat completion endpoint.
type LLMConfig struct {
	URL        string
	APIKey     string
	Model      string
	HTTPClient *http.Client
}

// AskResult is a synthesized answer plus the hits it was grounded in.
type AskResult struct {
	Answer string
	Hits   []Hit
}

// Ask retrieves then synthesizes a cited answer.
func Ask(ctx context.Context, r Retriever, emb semcache.Embedder, llm LLMConfig, corpus, question string, k int) (AskResult, error) {
	if llm.URL == "" || llm.Model == "" {
		return AskResult{}, fmt.Errorf("rag: ask requires an LLM URL and model (set ROSTAM_LLM_URL/ROSTAM_LLM_MODEL); use `rag query` for retrieval only")
	}
	hits, err := Retrieve(ctx, r, emb, corpus, question, k)
	if err != nil {
		return AskResult{}, err
	}
	sys, user := BuildPrompt(question, hits)
	answer, err := chatComplete(ctx, llm, sys, user)
	if err != nil {
		return AskResult{}, err
	}
	return AskResult{Answer: answer, Hits: hits}, nil
}

// chatComplete posts to an OpenAI-compatible /chat/completions endpoint. URL may
// be the base (…/v1) or the full path; we append the standard suffix if absent.
func chatComplete(ctx context.Context, cfg LLMConfig, system, user string) (string, error) {
	url := cfg.URL
	if !strings.Contains(url, "/chat/completions") {
		url = strings.TrimRight(url, "/") + "/chat/completions"
	}
	reqBody, err := json.Marshal(map[string]any{
		"model": cfg.Model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
	})
	if err != nil {
		return "", fmt.Errorf("rag: marshal chat request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("rag: LLM returned status %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("rag: LLM returned no choices")
	}
	return out.Choices[0].Message.Content, nil
}
