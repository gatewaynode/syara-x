# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**syara-x** is a Rust port of [SYARA (Super YARA)](../syara-rust-port), a security library that extends the YARA rule format with semantic, ML-classifier, and LLM-based matching. It catches malicious content (prompt injection, phishing, jailbreaks) by meaning and intent, not just exact text patterns.

**Goals:**
1. Faithful port of the full `.syara` DSL and execution engine
2. Embeddable library with a clean public API and optional C FFI (`capi/`)
3. Designed to sit alongside [YARA-X](../yara-x) — parallel library, not a YARA-X module (YARA-X has no runtime module registration)
4. Feature-gated ML deps mirroring Python's optional extras

**Python reference source:** `../syara-rust-port/syara/` — read it when porting behaviour.

## Commands

```bash
cargo build                          # build all crates
cargo test                           # run all tests
cargo test -p syara-x                # library tests only
cargo test <test_name>               # single test by name
cargo test -- --nocapture            # show println! output
cargo clippy -- -D warnings          # lint (must be clean)
cargo doc --open                     # browse generated docs
```

### Real-model tests (opt-in, `#[ignore]` by default)

These load actual model weights / hit real services and take minutes to run, so they're gated behind `--ignored`:

```bash
cargo test -p syara-x --features burn-llm        -- --ignored --nocapture integration_real_model               # Qwen3.5-0.8B
cargo test -p syara-x --features burn-llm        -- --ignored --nocapture integration_real_nemotron            # Nemotron-3-Nano-4B
cargo test -p syara-x --features sbert           -- --ignored --nocapture integration_real_openai_embed        # OpenAI-compatible /v1/embeddings (LM Studio / vLLM / openai.com)
cargo test -p syara-x --features sbert           -- --ignored --nocapture integration_real_ollama_embed        # Ollama /api/embed
cargo test -p syara-x --features sbert-onnx      -- --ignored --nocapture integration_real_onnx_embed          # Local MiniLM-L6-v2 ONNX (semantic matcher)
cargo test -p syara-x --features classifier      -- --ignored --nocapture integration_real_openai_classifier   # OpenAI-compatible /v1/embeddings, classifier path
cargo test -p syara-x --features classifier-onnx -- --ignored --nocapture integration_real_onnx_classifier     # Local MiniLM-L6-v2 ONNX (classifier)
cargo test -p syara-x --features phash           -- --ignored --nocapture integration_real_image_phash         # generated PNG via image crate
cargo test -p syara-x --features phash           -- --ignored --nocapture integration_real_wav_phash           # generated PCM WAV
cargo test -p syara-x-capi                       -- --ignored --nocapture integration_real_c_link             # C compiler + staticlib round-trip (Unix; needs system cc)
```

**After every major section of work that touches LLM / embedding / inference / model-loading code, ASK the user whether to run these real-model tests before declaring the section done.** The fixture tests catch shape bugs; only the real-model tests catch weight-loading and tokenizer regressions.

## Architecture

Data flows linearly: `.syara file` → `SyaraParser` → `Vec<Rule>` → `Compiler` → `CompiledRules` → `scan(text)` / `scan_file(path)` → `Vec<Match>`

### Crate layout

| Crate | Path | Purpose |
|---|---|---|
| `syara-x` | `syara/` | Main library |
| `syara-x-capi` | `capi/` | C FFI via cbindgen (Phase 7) |

### Key modules (`syara/src/`)

| Module | Maps from Python | Purpose |
|---|---|---|
| `lib.rs` | `__init__.py` | Public API: `compile()`, `compile_str()` |
| `models.rs` | `models.py` | Pure structs: `Rule`, `StringRule`, `SimilarityRule`, `PHashRule`, `ClassifierRule`, `LLMRule`, `Match`, `MatchDetail` |
| `parser.rs` | `parser.py` | Brace-counting `.syara` DSL parser; no grammar/lexer |
| `compiler.rs` | `compiler.py` | Validation (duplicate ids, undefined refs, wildcard-safe) |
| `compiled_rules.rs` | `compiled_rules.py` | Cost-ordered execution engine with LLM short-circuit |
| `condition.rs` | inline in `compiled_rules.py` | Recursive-descent boolean AST + evaluator (replaces Python `eval()`) |
| `cache.rs` | `cache.py` | SHA256-keyed `TextCache`, cleared after each scan |
| `config.rs` | `config.py` | Trait-object component registry (replaces `importlib`) |
| `error.rs` | — | `SyaraError` enum |
| `engine/cleaner.rs` | `engine/cleaner.py` | `TextCleaner` trait + Default (NFKC)/NoOp/Aggressive |
| `engine/chunker.rs` | `engine/chunker.py` | `Chunker` trait + 5 implementations |
| `engine/string_matcher.rs` | `engine/string_matcher.py` | `regex` crate + nocase/wide/dotall/fullword |

### Feature flags

| Flag | Enables | Phase |
|---|---|---|
| `sbert` | `engine/semantic_matcher.rs` — cosine similarity | 2 |
| `classifier` | `engine/classifier.rs` — ML classifiers (implies sbert) | 3 |
| `llm` | `engine/llm_evaluator.rs` — OpenAI/Ollama HTTP | 4 |
| `phash` | `engine/phash_matcher.rs` — image dHash | 5 |
| `all` | All of the above | — |

### Non-obvious design decisions

- **No YARA-X dependency in Phase 1** — string/regex matching uses the `regex` crate directly. YARA-X can be wired in later for string patterns if desired.
- **Condition evaluator** replaces Python's `eval()` with a typed AST (`condition.rs`). Grammar: `or → and → not → primary`. Wildcards (`$prefix*`) are resolved against the declared identifier map at scan time.
- **`is_identifier_needed()`** — before calling an expensive LLM evaluator, the engine optimistically assumes the LLM matches and checks if the condition would be true. If not, the call is skipped.
- **Wide modifier** — literal strings are interleaved with `\x00` bytes (UTF-16LE); regex patterns are matched against a null-stripped version of the text.
- **Compiler wildcard validation** — uses `\$\w+` regex on the condition string but skips matches immediately followed by `*` to avoid flagging `$prefix*` as an undefined identifier (Rust `regex` crate has no lookaheads).

### Porting phases

Authoritative phase numbering lives in `tasks/todo.md` — this is just a quick feature-area reference.

- Core (no ML) ✅ — parser, condition AST, string matching, cleaners, chunkers, registry, cache, public API
- `sbert` — semantic similarity (ONNX or HTTP embedding endpoint)
- `classifier` — ML classifiers
- `llm` ✅ — OpenAI / Ollama HTTP evaluators
- `burn-llm` / `burn-llm-gpu` ✅ — local LLM inference (Qwen3 + Nemotron, CPU + GPU backends)
- `phash` — perceptual image hashing (`image` crate + dHash)
- `capi` — C FFI via `cbindgen`

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Run `cargo test` and `cargo clippy -- -D warnings` before declaring done
- Ask yourself: "Would a staff engineer approve this?"
- Write tests that provide real demonstration of working code — no mock tests, no always-true tests
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "Is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over engineer
- Challenge your work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how
- Bugs always need to be confirmed fixed with existing unit tests or new tests created to capture the success or failure of the bug fix.

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Development Guidelines

- **Small and Modular**: Keep individual files under 500 lines; use thoughtful composition.
- **Follow the UNIX philosophy**:
    - "Make it easy to write, test, and run programs."
    - "Economy and elegance of design due to size constraints."
    - "Self-supporting system: avoid dependencies when possible."
- **Porting discipline**: Before implementing a feature, read the corresponding Python source in `../syara-rust-port/syara/`. Preserve semantics exactly unless there is a documented reason not to.

## Security

- **Security First**: Always consider the security implications of code decisions and strongly bias towards secure code.
- **Never Use Latest Dependencies**: Try to keep to N - 1, and never use packages that are less than 30 days old.
- **Pin Dependencies**: Always pin versions in `Cargo.toml`; commit `Cargo.lock`.
- **Thoroughly Review Everything**: Run security reviews, style reviews, architecture reviews and run tests regularly.

## MCP Tools

- **tilth**: Smarter code reading for Agents
