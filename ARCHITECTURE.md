# Architecture Vision

The ideal state of syara-x — what we're building toward.

## Principles

1. **Parse once, execute many.** All parsing, validation, and compilation happens upfront. The scan hot path touches only pre-computed structures (compiled ASTs, cached regexes, pre-embedded patterns).

2. **Zero silent failures.** Every misconfiguration, malformed rule, or unsupported modifier produces a clear error at compile time. Rules never silently skip matching.

3. **Type-safe boundaries.** Rust's type system should enforce invariants. No sentinel values, no stringly-typed state. `Option<T>` for optional data, enums for variants, newtypes for constrained values.

4. **Defensive at system boundaries, trusting internally.** Validate all external input (rule files, scanned content, WAV/image data, HTTP responses). Internal code paths can assume invariants established at the boundary.

5. **Feature isolation.** Each feature flag (`sbert`, `classifier`, `llm`, `phash`, `burn-llm`) is a self-contained opt-in. No compile errors, no dead code, no behavioral changes when a flag is off.

6. **Embeddable and safe.** The C API must be memory-safe by construction — no dangling pointers, no UB, no uninitialized out-params. The Rust API should be misuse-resistant.

## Data Flow (Ideal)

```
.syara file
  |
  v
SyaraParser ──> Vec<Rule>        # Regexes compiled once via LazyLock
  |                                # Escaped quotes handled
  |                                # Line numbers tracked for errors
  v
Compiler ──> CompiledRules        # Condition AST parsed and stored
  |                                # All identifiers validated
  |                                # Wildcards resolved
  |                                # Modifiers validated (fullword, wide, etc.)
  v
scan(text) / scan_file(path)      # Pre-compiled AST evaluated (no re-parse)
  |                                # Regex cache with entry API (no clones)
  |                                # Pattern embeddings cached across scans
  |                                # HTTP clients with timeouts
  |                                # LLM prompts hardened against injection
  v
Vec<Match>                        # Option<usize> positions
                                  # Only matched rules carry tags/meta
```

## Module Design

### Parser (`parser/`)
- **Layered tokenization**: top-level brace-counter and comment-stripper in `parser/mod.rs`; per-line tokenizer in `parser/scanner.rs`. The three state machines must agree on what counts as "inside a regex / string / triple-quoted literal" — pinned by the parity test in `parser/scanner.rs::tests::parity_with_split_rules_regex_consumption`.
- **No regex-over-line tokenization**: section line parsing goes through `Scanner` (escape-aware) rather than `LazyLock<Regex>` body captures (escape-blind). See BUG-039 close-out for what the latter cost us. If you find yourself reaching for `[^X]*` over a body that may contain `\X`, stop and lex.
- **String handling**: Byte-indexed cursor; UTF-8 char-boundary checks before slicing.
- **Escapes**: `\"`, `\\`, `\n`, `\t`, `\r` in quoted strings; `\/` (and any `\X`) preserved in regex bodies; triple-quoted (`"""..."""`) bodies captured raw.
- **Error reporting**: Per-line `line` field tracked through parsing; attached to `SyaraError::ParseError`.
- **Missing sections**: Explicit error for missing `condition:`, not silent empty string.

### Compiler (`compiler.rs`)
- **Single-pass validation**: Check identifiers, resolve wildcards, validate modifiers.
- **Eager regex compile**: Each `StringRule` is validate-compiled at `compile_str` time via `StringMatcher::validate` so malformed patterns surface as `SyaraError::InvalidPattern` here, not silently at scan time (BUG-040).
- **AST storage**: Parse condition into `Expr` and store it in `CompiledRules` — never re-parsed.
- **Modifier validation**: Verify all modifiers are actually implemented before accepting them.

### Condition Evaluator (`condition.rs`)
- **Strict parsing**: Error on unconsumed tokens, unknown characters, and malformed expressions.
- **Hypothetical evaluation**: `is_identifier_needed` should evaluate without cloning the match map — use a wrapper that overrides a single key.

### Execution Engine (`compiled_rules.rs`)
- **Pre-compiled AST**: Evaluate stored `Expr`, not re-parsed strings.
- **`scan_file` completeness**: Read file content for string matching alongside phash processing.
- **Cost-ordered execution**: Preserved from current design — cheap matchers first, LLM last with short-circuit.

### String Matcher (`string_matcher.rs`)
- **All modifiers work**: `fullword` adds `\b` boundaries, `wide` is cached, `dotall` is applied.
- **No regex clones**: Use entry API or index-based access from cache.
- **Accurate positions**: Wide match positions map back to original text coordinates.

### Cleaners & Chunkers (`cleaner.rs`, `chunker.rs`)
- **Unicode parity**: `AggressiveCleaner` strips Unicode digits to match Python behavior.
- **Input validation**: `chunk_size == 0` returns an error, not a panic.
- **Streaming**: For large texts, cleaners and chunkers should avoid materializing full intermediate strings where possible.

### HTTP-backed Engines (`semantic_matcher.rs`, `classifier.rs`, `llm_evaluator.rs`)
- **Shared embedding client**: One `embed()` implementation used by both semantic matcher and classifier.
- **Timeouts**: All HTTP clients configured with connect + read timeouts (default 30s).
- **Pattern caching**: Pattern embeddings computed once and cached across scans.
- **LLM hardening**: Structured prompt with clear delimiters. Response parsing requires exact `YES`/`NO` with boundary check. Attack surface documented.

### Phash Matcher (`phash_matcher.rs`)
- **Input validation**: `hash_size` capped at 8 in constructor. WAV `chunk_len` bounded to file size or reasonable max.
- **WAV correctness**: Account for channel count in frame calculation. Handle RIFF odd-chunk padding.
- **Bounds safety**: Video hash loop uses explicit bounds checks matching audio hash pattern.

### Cache (`cache.rs`)
- **Cheap keys**: Use `(cleaner_name, text)` tuple or `(&str, &str)` as key — let HashMap's built-in hasher do the work. No SHA256 pre-hashing.
- **Zero-copy returns**: Return `&str` from cache, not cloned `String`.

### Models (`models.rs`)
- **Idiomatic types**: `Option<usize>` for positions. `FromStr` trait for `Modifier`.
- **Lazy population**: `Match::no_match` carries only rule name and `matched: false` — no cloned tags/meta.

### C API (`capi/`)
- **Safe out-params**: Initialize `*out = null` at the start of every function.
- **Explicit capacity**: Store `Vec` capacity in `SyaraMatchArray` — no reliance on `shrink_to_fit`.
- **Full match data**: Expose tags, meta, and matched_patterns to C consumers.
- **cbindgen config**: Proper `cbindgen.toml` with opaque type declarations and include guards.

## Backend Architecture (LLM)

Two LLM backends coexist behind the `LLMEvaluator` trait:

```
LLMEvaluator (trait)
  |
  +-- HttpLLMEvaluator        [feature = "llm"]
  |     OpenAI / Ollama via HTTP
  |
  +-- BurnLLMEvaluator         [feature = "burn-llm"]
        Local inference via Burn framework
```

Backend selected via `Registry` configuration. Both implement the same trait. HTTP backend for cloud/hosted models, Burn backend for air-gapped or latency-sensitive deployments.

## Test Strategy

- **Unit tests per module**: Each matcher, cleaner, chunker tested in isolation.
- **Integration tests**: Cover every modifier (`fullword`, `wide`, `dotall`, `nocase`), regex-only matching, error paths, mixed rule types, and `scan_file` with combined phash+string rules.
- **Negative tests**: Malformed syntax, duplicate identifiers, unclosed braces, undefined references.
- **Fuzz targets**: Parser and WAV/image parsers should have fuzz harnesses for untrusted input.
- **No mocks**: Real execution paths, real regex compilation, real condition evaluation.

## File Size Budget

Target: every `.rs` file under 500 lines. Current violations:
- `parser/mod.rs` — 716 lines (large in-module test suite; non-test impl is ~280 lines)
- `parser/scanner.rs` — 538 lines (large in-module test suite; non-test impl is ~310 lines)
- `condition.rs` — 456 lines (acceptable, but monitor)
- `capi/src/lib.rs` — 439 lines (acceptable)
