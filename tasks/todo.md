# Task Planning

<!-- Use this file to plan out tasks with checkable items. -->
<!-- Mark items complete as you go and add a review section when done. -->

## Planned Features

### Publish to crates.io

- [x] Audit `Cargo.toml` metadata (name, version, description, license, repository, keywords, categories)
- [x] Ensure `LICENSE` file is present and matches `Cargo.toml` license field
- [x] Review public API surface — only expose what should be public (5 modules made `pub(crate)`)
- [x] Add `README.md` that crates.io will render (set `readme = "../README.md"` in both Cargo.toml)
- [x] Run `cargo publish --dry-run` to catch packaging issues (clean, 0 warnings)
- [x] Verify all pinned dependencies resolve cleanly from crates.io
- [x] Check for any path dependencies that need to be published first (publish order: syara-x then capi)
- [x] Set up crate ownership / team access on crates.io
- [x] Publish initial version (published 2026-04-13)

### Integrated Local LLM Processing via Burn Framework

Adds local inference using [Burn](https://github.com/tracel-ai/burn) as an additional LLM backend alongside the existing HTTP-based evaluator. Target models: Qwen3.5-0.8B (hybrid DeltaNet/attention) and Nemotron-3-Nano-4B (hybrid Mamba/attention). Full plan: `~/.claude/plans/shiny-mapping-frost.md`

- [x] Research Burn's current model ecosystem — available architectures, ONNX import, tokenizer support
- [x] Decide on target model(s): Qwen3.5-0.8B-Base, NVIDIA-Nemotron-3-Nano-4B-BF16

#### Phase 1: Feature Flag, Shared Utilities, Stub Evaluator
- [x] Add `burn`, `tokenizers`, `safetensors` workspace deps (burn 0.20.1, tokenizers 0.22.2, safetensors 0.7.0)
- [x] Add `burn-llm` feature flag in `syara/Cargo.toml` (+ `burn-llm-gpu`)
- [x] Extract `build_prompt`/`parse_response` as shared fns in `llm_evaluator.rs`
- [x] Widen LLM feature gates to `any(feature = "llm", feature = "burn-llm")`
- [x] Create stub `burn_evaluator.rs` implementing `LLMEvaluator`
- [x] Wire into Registry in `config.rs`
- [x] Verify: `cargo test`, `cargo test --features burn-llm`, `cargo test --features llm`

#### Phase 2: Common Building Blocks in Burn
- [x] `burn_model/mod.rs` — Module declarations, re-exports
- [x] `burn_model/attention.rs` — FullAttention (GQA + QK-norm + partial RoPE) — uses built-in RmsNorm/RotaryEncoding
- [x] `burn_model/ffn.rs` — FeedForward (gate/up/down + SiLU)
- [x] Unit tests for each block (shape checks, causal mask, partial RoPE, GQA repeat)
- [x] No custom RmsNorm or RoPE needed — Burn has built-in modules

#### Phase 3: Gated DeltaNet + Qwen3.5 Model Assembly
- [x] `burn_model/deltanet.rs` — Gated DeltaNet (recurrent mode)
- [x] `burn_model/qwen3.rs` — Qwen3Config, HybridBlock, Qwen3TextModel
- [x] Tests: forward pass, hybrid dispatch verification

#### Phase 4: Weight Loading, Tokenizer, Inference Pipeline
- [x] `burn_model/loader.rs` — load_qwen3 from safetensors (config.json parsing, bf16→f32, weight mapping)
- [x] `burn_model/generate.rs` — greedy_generate (argmax loop, EOS stop)
- [x] Complete `BurnEvaluator::evaluate()` (tokenize → generate → parse, Mutex for Send+Sync)
- [x] Create test fixtures (tiny model + tokenizer)
- [x] Integration test with real Qwen3.5 model (`#[ignore]`)

#### Phase 5: Nemotron Model, GPU Backend, Documentation
- [ ] `burn_model/mamba.rs` — MambaBlock
- [ ] `burn_model/nemotron.rs` — NemotronModel with hybrid pattern dispatch
- [ ] Auto-detect model type from config.json
- [ ] GPU backend via `burn-llm-gpu` feature
- [ ] BurnEvaluatorBuilder API
- [ ] Documentation and benchmarks
- [ ] Full feature matrix verification
