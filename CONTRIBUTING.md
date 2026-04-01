# Contributing to syara-x

## How this project is maintained

syara-x is primarily developed with [Claude Code](https://claude.ai/claude-code),
Anthropic's AI coding assistant. The project was ported from Python to Rust
across six implementation phases, each planned and executed by Claude under
human direction.

This document is written to be useful to both human contributors and AI coding
assistants picking up context on the project.

---

## For AI coding assistants

If you are an LLM or AI coding tool reading this file, here is the critical
context you need:

**What this project is:**
A Rust port of SYARA (Super YARA), a security library that extends the YARA
rule format with semantic, ML-classifier, and LLM-based matching. The Python
reference source lives at `../syara-rust-port/syara/`.

**Porting discipline:**
Before implementing any feature, read the corresponding Python source. Preserve
semantics exactly unless there is a documented reason not to. Document
deviations in code comments.

**Architecture rules:**
- Keep files under 500 lines. Prefer composition over large files.
- Feature-gate ML/HTTP dependencies: `sbert`, `classifier`, `llm`, `phash`.
- Execution is cost-ordered: strings → similarity → phash → classifier → LLM.
- LLM calls are short-circuited by `is_identifier_needed()` in `condition.rs`.
- No mocks in tests. Use real trait implementations and `tempfile` for I/O.

**Key non-obvious decisions:**
- `condition.rs` replaces Python's `eval()` with a typed recursive-descent AST.
- Wide modifier interleaves literal bytes with `\x00` (UTF-16LE simulation).
- The compiler wildcard check skips identifiers followed by `*` because the
  `regex` crate has no lookaheads.
- `is_identifier_needed()` assumes the LLM will match, then checks if the
  condition would still fail — skips the call if it cannot matter.
- HTTP embedding uses Ollama's `/api/embed` with `{"model": ..., "input": ...}`.
- LLM evaluation uses Ollama's `/api/chat` with `stream: false`; responses are
  parsed as `YES: explanation` or `NO: explanation`.
- dHash for images: resize to (hash_size+1) × hash_size, compare consecutive
  horizontal samples. Uniform-value byte arrays always produce hash=0.
- WAV parsing is pure Rust (RIFF chunk scanner) — no `hound` dependency.

**Before declaring any task done:**
Run `cargo test --workspace` and `cargo clippy -- -D warnings`. Both must pass
with zero failures and zero warnings.

---

## For human contributors

### Getting started

```bash
git clone https://github.com/gatewaynode/syara-x
cd syara-x
cargo build
cargo test
```

### Code style

- Rust edition 2021, `clippy` clean at `-D warnings`
- No `unwrap()` in library code; use `?` or return `SyaraError`
- Feature-gate any dependency that is not needed for string-only matching
- Pin all dependency versions to N-1 stable releases; never use packages
  less than 30 days old

### Dependency policy

This project follows a conservative dependency policy:
- Prefer pure Rust implementations over C bindings where practical
- Disable default features and enable only what is needed
- Add new dependencies only when the alternative is significantly more code

### Testing

Tests must demonstrate real behaviour:
- No always-true assertions
- No mocked trait objects that hard-code return values
- Use `tempfile::NamedTempFile` for tests that need real files
- Test doubles (e.g. `FixedHashMatcher`) must be real trait implementations
  that exercise the actual code path

### Submitting changes

1. Open an issue describing the change before writing code
2. One logical change per PR
3. Include tests that would have caught the bug or demonstrate the feature
4. Update `CLAUDE.md` if you add a new non-obvious design decision

---

## Porting phases

| Phase | Feature flag | Status |
|---|---|---|
| 1 | _(none)_ | Complete — parser, compiler, string matching, cleaners, chunkers |
| 2 | `sbert` | Complete — HTTP semantic similarity via Ollama |
| 3 | `classifier` | Complete — ML text classifiers |
| 4 | `llm` | Complete — Ollama LLM evaluator |
| 5 | `phash` | Complete — image/audio/video perceptual hashing |
| 6 | `capi` | Complete — C FFI via cbindgen |

---

## License

MIT — contributions are accepted under the same license.
