// SPDX-License-Identifier: Apache-2.0

// Package rag turns local files into a Rostam corpus and answers questions over
// it with grounded, cited LLM responses. It orchestrates the vector engine, an
// optional embedder, and an OpenAI-compatible chat endpoint behind a small
// Retriever interface with embedded and HTTP backends.
package rag
