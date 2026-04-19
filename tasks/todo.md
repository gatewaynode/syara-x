# Task Planning

<!-- This file is the authoritative source for phase numbering. -->
<!-- Mark items complete as you go and add a review section when done. -->

## Porting Phases (authoritative)

| Phase | Feature flag | Area | Status |
|---|---|---|---|
| 1 | _(none)_ | Core: parser, condition AST, string matching, cleaners, chunkers, registry, cache, public API | ✅ Complete |
| 2 | `sbert` / `sbert-onnx` | Semantic similarity (ONNX + HTTP embedding endpoint) | ✅ Complete |
| 3 | `classifier` / `classifier-onnx` | ML text classifiers (HTTP + local ONNX) | ✅ Complete |
| 4 | `llm` | HTTP LLM evaluators (OpenAI-compatible + Ollama legacy) | ✅ Complete |
| 5 | `burn-llm` / `burn-llm-gpu` | Local LLM inference (Qwen3 + Nemotron, CPU + GPU) | ⚠️ Fixture tests pass; real-model bugs (see below) |
| 6 | `phash` | Perceptual image hashing (`image` crate + dHash) | ✅ Complete |
| 7 | `capi` | C FFI via `cbindgen` | ✅ Complete |

---

## Completed Features

### HTTP LLM Configurability + OpenAI-Compatible Default ✅ (2026-04-19)

Refinement recorded in `tasks/TASK-REFINEMENT.md`.

- [x] Added `OpenAiChatEvaluator` + `OpenAiChatEvaluatorBuilder` targeting `/v1/chat/completions` (LM Studio / vLLM / llama-server / openai.com / Ollama shim)
- [x] Builder knobs: endpoint, model, api_key (bearer), temperature (default 0.0), max_tokens (default 8192 — reasoning-model headroom, BUG-035), system_prompt, extra headers, connect_timeout (20s default), read_timeout (60s default)
- [x] BUG-035: reasoning-model truncation — detect `finish_reason=length` with empty `content` and surface an actionable `SyaraError::LlmError` (pointing at `.max_tokens(…)`) instead of silently falling through `parse_response` as "Ambiguous LLM response:". Pure extractor helper `extract_openai_content` with 5 unit tests. Confirmed live against `qwen/qwen3.6-35b-a3b` via LM Studio. (Originally filed as BUG-029 in session notes; renumbered to avoid collision with the WAV RIFF padding bug.)
- [x] Response cache keyed on `(pattern, chunk)` when `temperature==0.0`; bounded at 1024 entries; cleared per-scan via `LLMEvaluator::clear_cache` + `Registry::clear_llm_caches`
- [x] Env-var auto-detection for default registration: `SYARA_LLM_{ENDPOINT,MODEL,API_KEY}` (preferred) → `OPENAI_{BASE_URL,MODEL,API_KEY}` (fallback) → hardcoded `localhost:1234` fallback; opt-out via `SYARA_LLM_NO_ENV=1`
- [x] Default registry entry renamed `ollama` → `openai-api-compatible`; `ollama` kept as legacy for users on Ollama's native `/api/chat`
- [x] `LLMRule::default` and parser default both updated to `openai-api-compatible`
- [x] 5 new unit tests: empty-input short-circuit, doubled timeouts, builder validation, builder knob composition, clear_cache safety
- [x] `#[ignore]` `integration_real_openai_chat` test hitting LM Studio (reads env vars)
- [x] README updates: feature-flag row, LLM example, env-var table + opt-out docs + scoped-token guidance, built-in component list
- [x] CLAUDE.md real-model-tests block updated with `integration_real_openai_chat`
- [x] `cargo test --features llm` all green; `cargo clippy --all-features -- -D warnings` clean

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

Migrated to `tasks/BUGS.md` on 2026-04-19 as **BUG-036** (Qwen3.5 shape
crash) and **BUG-037** (Nemotron CPU slowness). Both blocked on a candle
or mistral.rs backend migration — see `ROADMAP.md`.

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

<!-- Phase 3 shipped 2026-04-18 — kept below for provenance, but it is complete. -->

### Phase 3: ML Classifiers (`classifier` + `classifier-onnx`) ✅

Port the Python `engine/classifier.py` to Rust. Provides `TextClassifier` trait
plus three backends — two HTTP variants and a local ONNX backend that composes
the Phase 2 `OnnxEmbeddingMatcher` so we don't duplicate ~200 lines of inference
plumbing.

- [x] `TextClassifier` trait + default `classify_chunks` (threshold + `MatchDetail`) — `engine/classifier.rs`
- [x] `OpenAiEmbeddingClassifier` (registry default for `"tuned-sbert"`) + `OllamaEmbeddingClassifier` HTTP backends, both reusing the shared `HttpEmbedder`
- [x] `OnnxEmbeddingClassifier` behind the `classifier-onnx` feature, wraps `OnnxEmbeddingMatcher::embed` + cosine
- [x] `Cargo.toml`: `classifier-onnx = ["classifier", "sbert-onnx"]`; included in `all`
- [x] Registry wiring (`get_classifier` / `register_classifier`) — `config.rs`
- [x] Cost-ordered execution slot 4 in `compiled_rules.rs::execute_classifier`; pattern-map seeded for `is_identifier_needed`
- [x] Parser `parse_classifier_section` (threshold/cleaner/chunker/classifier params)
- [x] Public `CompiledRules::register_classifier` API
- [x] 6 module/cross-module unit tests passing
- [x] Integration test `tests/classifier_integration.rs` — deterministic `FixedClassifier` + 2 `#[ignore]` real-backend tests
- [x] README: `classifier:` rule example + `classifier-onnx` row in feature table + System dependencies updated
- [x] CLAUDE.md real-model test block: `integration_real_openai_classifier` + `integration_real_onnx_classifier`
- [x] Real-backend tests run 2026-04-18:
      - `integration_real_openai_classifier` ✅ (LM Studio nomic-embed)
      - `integration_real_onnx_classifier` ✅ (local MiniLM-L6-v2; rule pattern aligned to MiniLM-friendly phrasing — see `project_phase3_progress.md`)
- [x] `scripts/install_onnxruntime_xdg.sh` — installs `~/.local/bin/with-onnxruntime` wrapper for ergonomic ONNX-feature test runs

**Known limitations (MVP):**
- HTTP classifier shares the `HttpEmbedder` cache and timeouts with sbert — same retry posture (none, beyond reqwest defaults).
- ONNX classifier inherits `max_length=256` from `OnnxEmbeddingMatcher`.
- Python's `train()` threshold-calibration loop is intentionally not ported — Rust users tune the rule's `threshold` field directly. Re-evaluate if a real fine-tuning use case appears.

<!-- Phase 6 shipped 2026-04-18 — kept below for provenance, but it is complete. -->

### Phase 6: Perceptual Hashing (`phash`) ✅

Port the Python `engine/phash_matcher.py` to Rust. Provides `PHashMatcher`
trait plus three backends — `ImageHashMatcher` (dHash via the `image` crate),
`AudioHashMatcher` (pure-Rust WAV reader), `VideoHashMatcher` (byte-sampling
fingerprint). The core engine landed in earlier porting passes; the
2026-04-18 closeout brought it up to the Phase 2/3 standard.

**Already complete (pre-closeout audit 2026-04-18):**

- [x] `PHashMatcher` trait + default `match_rule` (`engine/phash_matcher.rs`, 604 lines)
- [x] `ImageHashMatcher` — dHash, Lanczos resize, grayscale (`image` crate)
- [x] `AudioHashMatcher` — RIFF/fmt/data chunk parser, multi-channel, BUG-014/015/029 fixes
- [x] `VideoHashMatcher` — 65-sample byte fingerprint, zero external deps
- [x] `PHashRule` model + `Default { threshold: 0.9, phash_name: "imagehash" }`
- [x] Registry defaults (`imagehash` / `audiohash` / `videohash`) + `register_phash_matcher` API
- [x] Cost slot 3 in `execute_phash`; `scan_file` reads file as text so string + phash can compose (BUG-008)
- [x] Parser `parse_phash_section` (inline `hasher=` syntax) + `test_parse_phash`
- [x] Compiler duplicate-identifier validation includes phash
- [x] `Cargo.toml`: `phash = ["dep:image"]`; `image` pinned to 0.24.9
- [x] 14 unit tests in `engine::phash_matcher::tests` passing

**Closeout (this phase closes):**

- [x] Integration test `tests/phash_integration.rs` — deterministic `FixedHashMatcher` + 2 `#[ignore]` real-backend tests (done 2026-04-18)
- [x] `[dev-dependencies]` += `image` so integration tests can synthesize PNGs at runtime (done 2026-04-18)
- [x] README: replace broken YAML-block `phash:` example with inline `hasher="..."` syntax that matches the parser (done 2026-04-18)
- [x] CLAUDE.md real-model test block: add `integration_real_image_phash` + `integration_real_wav_phash` rows (done 2026-04-18)
- [x] Real-backend tests run 2026-04-18:
      - `integration_real_image_phash` ✅ (generated 32×32 PNGs via `image` crate; identical → match, inverted → miss)
      - `integration_real_wav_phash` ✅ (synthesized 128-sample 16-bit mono PCM WAV; identical → match)

**Design notes:**

- Real-backend tests synthesize their inputs in a tempdir so the repo does not grow committed binary fixtures.
- Parser grammar is single-line (`$id = "<file_path>" key=value key="value"`), not YAML-block; parameter name is `hasher=`, not `phash=` (the `phash:` section header is the block name, but the matcher-selection key inside the rule is `hasher=`).

<!-- Phase 7 shipped 2026-04-18 — kept below for provenance, but it is complete. -->

### Phase 7: C FFI (`capi`) ✅

Expose the `syara-x` library to C callers via cbindgen-generated headers and
a combined `cdylib` + `staticlib`. The FFI surface, header generator, and
unit tests all landed in earlier porting passes; the 2026-04-18 closeout
brought it up to the Phase 2/3/6 standard.

**Already complete (pre-closeout audit 2026-04-18):**

- [x] `capi/Cargo.toml` — `crate-type = ["cdylib", "staticlib"]`, cbindgen 0.26.0 as build-dep
- [x] `capi/build.rs` — inline cbindgen config (Language::C, `SYARA_X_H` include guard, documentation=true, tab_width=4)
- [x] `capi/src/lib.rs` — 8 public `extern "C"` fns (`syara_compile_str` / `compile_file` / `scan` / `scan_file` / `rule_count` / `rules_free` / `matches_free` / `last_error`) + `SyaraStatus` enum + `SyaraRules` opaque handle + `SyaraMatchArray` (with stored `capacity`)
- [x] Thread-local error store via `thread_local! RefCell<CString>`
- [x] BUG-018 regression test — `SyaraMatchArray::capacity` roundtrips so `Vec::from_raw_parts` in `syara_matches_free` is sound
- [x] BUG-030 regression test — `*out` is null-initialised on all error paths so callers see null instead of garbage
- [x] `capi/syara_x.h` — checked-in, ~4.6 KB, matches API
- [x] 11 unit tests in `capi/src/lib.rs::tests` all passing
- [x] README C example (`README.md:182-203`) — verified to match the actual FFI

**Closeout (this phase closes):**

- [x] Integration test `capi/tests/ffi_integration.rs`:
      - `header_matches_regenerated_cbindgen_output` (default) — regenerates the header in-process with the same cbindgen builder config as `build.rs`, diffs against the checked-in `capi/syara_x.h`; catches any drift from hand-edits
      - `integration_real_c_link` (`#[ignore]`, Unix-only) — builds `libsyara_x_capi.a` via `cargo build --release`, writes a small C driver to a tempdir, invokes `cc` with platform-appropriate link flags (macOS: `-framework CoreFoundation -framework Security`; Linux: `-lpthread -ldl -lm`), runs the binary, asserts `HIT:r` in stdout
- [x] `capi/Cargo.toml` `[dev-dependencies]` += `tempfile = { workspace = true }` and `cbindgen = "0.26.0"` (needed for the integration test)
- [x] `capi/src/lib.rs:478` — one-line clippy fix: `0x1 as *mut T` → `std::ptr::dangling_mut::<T>()` (same non-null placeholder semantics, satisfies `manual_dangling_ptr`)
- [x] CLAUDE.md crate-layout row: `(Phase 6)` → `(Phase 7)` for capi
- [x] CLAUDE.md real-model-tests block: append `integration_real_c_link` row
- [x] Real-backend test run 2026-04-18:
      - `integration_real_c_link` ✅ (macOS, Apple `cc`, 6.34s including release build)

**Design notes:**

- The C-link test synthesises its `driver.c` at runtime in a tempdir — no committed `.c` fixture that would need its own maintenance.
- `build.rs` unconditionally regenerates `capi/syara_x.h` on every `cargo build`, so the header drift test is tight: any divergence implies someone hand-edited the header (or the source) outside the normal build cycle.
- The `#[ignore]` test is `#[cfg(unix)]`. Windows (MSVC) is deliberately out of scope for this closeout — would need a different link-flag set and a different `cc` detection path.
- No `capi` feature-flag plumbing: `syara-x` is depended on with default features, so `phash` / `sbert` / `llm` C-level entry points are **not** wired up. If needed later, gate new `extern "C"` fns behind capi-local features that re-export syara-x's.
