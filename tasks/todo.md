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
- [ ] Set up crate ownership / team access on crates.io (manual — see `tasks/PUBLISH-GUIDE.md`)
- [ ] Publish initial version (manual — see `tasks/PUBLISH-GUIDE.md`)

### Integrated Local LLM Processing via Burn Framework

Adds local inference using [Burn](https://github.com/tracel-ai/burn) as an additional LLM backend alongside the existing HTTP-based evaluator (`engine/llm_evaluator.rs`, Phase 4). Users choose HTTP (OpenAI/Ollama) or local Burn inference via configuration.

- [ ] Research Burn's current model ecosystem — available architectures, ONNX import, tokenizer support
- [ ] Decide on target model(s) for local LLM evaluation (small transformer, distilled, etc.)
- [ ] Add `burn` dependencies behind a new feature flag (e.g., `burn-llm`)
- [ ] Implement tokenizer integration (Burn + `tokenizers` crate or built-in)
- [ ] Implement model loading and inference pipeline
- [ ] Implement as a new backend behind the existing `LLMEvaluator` trait — HTTP and Burn coexist
- [ ] Allow backend selection via config registry (e.g., `llm_backend: "burn"` vs `"http"`)
- [ ] Benchmark inference latency vs. HTTP-based evaluator
- [ ] Add tests with a small model or fixture weights
- [ ] Document feature flag usage, backend selection, and model setup
