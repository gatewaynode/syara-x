# Task Planning

<!-- This file is the authoritative source for phase numbering. -->
<!-- Mark items complete as you go and add a review section when done. -->

## Porting Phases (authoritative)

| Phase | Feature flag | Area | Status |
|---|---|---|---|
| 1 | _(none)_ | Core: parser, condition AST, string matching, cleaners, chunkers, registry, cache, public API | ✅ Complete |
| 2 | `sbert` / `sbert-onnx` | Semantic similarity (ONNX + HTTP embedding endpoint) | ✅ Complete |
| 3 | `classifier` | ML text classifiers | ⬜ Pending |
| 4 | `llm` | HTTP LLM evaluators (OpenAI / Ollama) | ✅ Complete |
| 5 | `burn-llm` / `burn-llm-gpu` | Local LLM inference (Qwen3 + Nemotron, CPU + GPU) | ⚠️ Fixture tests pass; real-model bugs (see below) |
| 6 | `phash` | Perceptual image hashing (`image` crate + dHash) | ⬜ Pending |
| 7 | `capi` | C FFI via `cbindgen` | ⬜ Pending |

---

## Completed Features

### Publish to crates.io ✅

- [x] Audit `Cargo.toml` metadata (name, version, description, license, repository, keywords, categories)
- [x] Ensure `LICENSE` file is present and matches `Cargo.toml` license field
- [x] Review public API surface — only expose what should be public (5 modules made `pub(crate)`)
- [x] Add `README.md` that crates.io will render (set `readme = "../README.md"` in both Cargo.toml)
- [x] Run `cargo publish --dry-run` to catch packaging issues (clean, 0 warnings)
- [x] Verify all pinned dependencies resolve cleanly from crates.io
- [x] Check for any path dependencies that need to be published first (publish order: syara-x then capi)
- [x] Set up crate ownership / team access on crates.io
- [x] Publish initial version (published 2026-04-13)

### Phase 5: Integrated Local LLM Processing via Burn Framework ✅

Local inference using [Burn](https://github.com/tracel-ai/burn) as an additional LLM backend alongside the existing HTTP-based evaluator. Target models: Qwen3.5-0.8B (hybrid DeltaNet/attention) and Nemotron-3-Nano-4B (hybrid Mamba/attention). Full plan: `~/.claude/plans/shiny-mapping-frost.md`

- [x] Research Burn's current model ecosystem — available architectures, ONNX import, tokenizer support
- [x] Decide on target model(s): Qwen3.5-0.8B-Base, NVIDIA-Nemotron-3-Nano-4B-BF16

Sub-phases (internal to the Burn LLM plan):

- [x] **5.1** Feature flag, shared utilities, stub evaluator
- [x] **5.2** Common building blocks in Burn (attention, FFN)
- [x] **5.3** Gated DeltaNet + Qwen3.5 model assembly
- [x] **5.4** Weight loading, tokenizer, inference pipeline (Qwen3 end-to-end)
- [x] **5.5** Nemotron model (Mamba2 + hybrid pattern), GPU backend, BurnEvaluatorBuilder, test fixtures

Final state: 131 lib tests + 25 integration tests + 4 doc-tests, clippy clean with all features, `burn-llm-gpu` compiles.

### Phase 5 Known Bugs (on hold)

Real-model integration tests (`--ignored`) revealed issues that don't affect fixture tests:

**Qwen3.5-0.8B — tensor shape crash:**
```
Mul: Incompatible size at dimension '2' => '2048 != 128'
Lhs [1, 70, 2048], Rhs [1, 1, 128]
```
Root cause: `models/Qwen3.5-0.8B-Base/config.json` is actually **Qwen3_5ForConditionalGeneration** (multimodal VL model) with features our Burn code doesn't implement:
- `attn_output_gate: true` — attention output gating (not in our `FullAttention`)
- `mrope_interleaved: true` + `mrope_section: [11,11,10]` — multi-dimensional RoPE for vision, we use plain RoPE
- `layer_types` explicit list (replaces `full_attention_interval` derivation)
- `head_dim: 256`, `linear_key_head_dim: 128`, `linear_num_key_heads: 16` → DeltaNet v_dim=2048, much larger than tiny fixture dims — exposed a broadcast axis bug in DeltaNet recurrence
- Architecture string `Qwen3_5ForConditionalGeneration` may not match our detector

**Nemotron-3-Nano-4B — extreme slowness (not a crash):**
Model loads and runs, but NdArray CPU backend on 4B params with no KV cache is impractical. Each token requires a full forward pass through 42 layers × 3136-dim. Expected to take hours per generation.

**Options for future fix (pick when revisiting):**
1. **Replace Burn inference with candle or mistral.rs** — both have Metal/MLX support, KV cache, and existing Qwen3/Nemotron implementations. Keep Burn fixture tests as architecture validation.
2. **Find a non-VL Qwen3 model** that matches what we've built (no output gate, plain RoPE).
3. **Implement missing features in Burn** — output-gated attention, mrope, KV cache. Weeks of work to reach where candle/mistral.rs already are.

MLX-level Apple Silicon performance will require candle (Metal) or mistral.rs (MLX bindings) regardless — Burn's wgpu/Metal path uses generic shaders.

---

## Planned Features

<!-- Phase 2 shipped 2026-04-17 — kept below for provenance, but it is complete. -->

### Phase 2: Semantic Similarity (`sbert` + `sbert-onnx`) ✅

Port the Python `engine/semantic_matcher.py` to Rust. Provides `SemanticMatcher` trait + backends for rules that use `similarity:` blocks (cosine similarity between embeddings). Full plan: `~/.claude/plans/zazzy-soaring-token.md`.

**Already complete (audit 2026-04-17):**

- [x] `SemanticMatcher` trait + default `match_chunks` (`engine/semantic_matcher.rs:20-64`)
- [x] `cosine_similarity` helper with edge cases (`engine/semantic_matcher.rs:72-83`)
- [x] `HttpEmbeddingMatcher` (Ollama-compatible `/api/embed`) (`engine/semantic_matcher.rs:93-113`)
- [x] Shared `HttpEmbedder` w/ timeouts + embedding cache (`engine/mod.rs:34-111`, BUG-011/012/033)
- [x] Registry wiring, default `"sbert"` → HTTP (`config.rs:85-93, 150-172`)
- [x] Execution hook in cost-ordered scan (`compiled_rules.rs:118-126, 194-206`)
- [x] Parser + `SimilarityRule` model (Phase 1)
- [x] 8 unit tests: cosine math, match_chunks, HTTP timeouts, cache

**Remaining (this phase closes):**

- [x] `sbert-onnx` sub-feature + `ort` / `ndarray` deps pinned in `Cargo.toml` (done 2026-04-17)
- [x] `engine/onnx_embedder.rs` — `OnnxEmbeddingMatcher` (MiniLM-L6-v2, mean-pool + L2-norm) (done 2026-04-17)
- [x] `scripts/fetch_minilm.sh` + document `../models/all-MiniLM-L6-v2/` layout (done 2026-04-17)
- [x] Integration test `tests/similarity_integration.rs` (deterministic `FixedMatcher` fixture) (done 2026-04-17)
- [x] `#[ignore]` `integration_real_http_embed` test (hits real Ollama) (done 2026-04-17)
- [x] `#[ignore]` `integration_real_onnx_embed` test (loads real MiniLM ONNX) (done 2026-04-17)
- [x] README feature table + rustdoc for the ONNX backend (done 2026-04-17)
- [x] `CLAUDE.md` real-model test section: add the new command (done 2026-04-17)
- [x] Real-model tests run against LM Studio + local ONNX (done 2026-04-17):
      - `integration_real_openai_embed` ✅ (LM Studio `/v1/embeddings`, `text-embedding-nomic-embed-text-v1.5`)
      - `integration_real_onnx_embed` ✅ (local MiniLM-L6-v2, 9.06s real inference)
      - `integration_real_ollama_embed` ⏭️ skipped (user moved off Ollama to LM Studio)

**Known limitations (MVP):**
- MiniLM-L6-v2 only; `max_length=256` hard-coded
- No quantized/int8 support in MVP (fp32 ONNX only)
- `ort` uses `load-dynamic` → requires system `libonnxruntime` at runtime
  (macOS: `brew install onnxruntime` + set `ORT_DYLIB_PATH` — see README)

### Phase 3: ML Classifiers (`classifier`)

- [ ] _(to be scoped)_

### Phase 6: Perceptual Hashing (`phash`)

- [ ] _(to be scoped)_ — `image` crate + dHash for image/audio/video perceptual hashing

### Phase 7: C FFI (`capi`)

- [ ] _(to be scoped)_ — `capi/` crate already stubbed, needs cbindgen integration
