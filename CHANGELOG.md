# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until `1.0.0`, minor-version bumps may include breaking changes to the DSL or
public API.

## [0.4.0] — 2026-05-10

### BREAKING CHANGES

- **Rules using `\/` in regex literals or `\"` in meta values / kv
  parameter values now parse correctly.** Pre-fix, the `.syara` parser's
  per-line regex tokenizers (`REGEX_PATTERN_RE`, `META_KV_RE`,
  `KV_PARAMS_RE`) used unescape-unaware body captures (`[^/]*`,
  `[^"]*`) that truncated at the first delimiter byte regardless of
  any preceding backslash. Affected rules silently miscompiled to a
  truncated body that either failed to compile or compiled to a
  pattern that never matched real input — and the failure was hidden
  by a silent error swallow at scan time (BUG-040). After this fix,
  these rules compile to the intended body and match real content.
  **Consumers should re-baseline match expectations** — particularly
  any test fixtures that asserted `!matched` for a rule containing
  `\/` or `\"` in a delimited body. Nine patterns across three files
  in the `llm_context_shield` consumer corpus were repaired.
- **Malformed regex patterns now error at `compile_str` time, not
  silently at scan time** (BUG-040). A rule containing e.g. `/[/`
  used to compile successfully and return zero matches forever; it
  now returns `Err(SyaraError::InvalidPattern { ... })` from
  `compile_str` / `compile_file`. Public API surface unchanged
  (`scan` still returns `Vec<Match>`).
- **Unknown regex flag letters now error** rather than being
  silently dropped (e.g. `/abc/sm` previously parsed as flags `i?`
  and dropped `s` and `m`; now errors with
  `unsupported regex flag '...' (only 'i' is recognized)`).
  **Migration:** rules that relied on the silently-dropped flags
  should switch to inline flags inside the pattern body — `(?s)`
  for dotall, `(?m)` for multiline, `(?i)` for case-insensitive,
  `(?x)` for extended mode. Example: `/foo.bar/sm` becomes
  `/(?sm)foo.bar/`. Inline flags are part of the `regex` crate
  syntax and have been supported since v0.3.0; this BREAKING
  change only removes the silent-drop fallback for trailing
  flag letters.

### Fixed

- **BUG-039 — `\/` truncation in regex bodies.** Reported by the
  `llm_context_shield` consumer (Phase 13.5b, 2026-05-09): the regex
  `<\/?(system|user|assistant)[\s>]` did not match `<system>`
  despite YARA-X parity. Root cause: the per-line parser regex
  `REGEX_PATTERN_RE = (\$\w+)\s*=\s*/([^/]*)/(i?)\s*(.*)` truncated
  the body at the first `/` byte. Fixed by replacing the five
  `LazyLock<Regex>` per-line tokenizers in `parser/sections.rs` with
  a hand-rolled `Scanner` (`syara/src/parser/scanner.rs`) that
  mirrors the existing top-level state-machine idiom in
  `parser/mod.rs::split_rules`. Bug class also repaired in meta
  values and kv parameters (latent — no consumer rule had triggered
  it yet). See `tasks/05-10-2026_BUGS.md` for the full close-out.
- **BUG-040 — silent regex-compile error swallow.** The string-pattern
  scan loop in `compiled_rules.rs::execute_rule` had a `_ => {}` arm
  that absorbed both `Ok(empty)` and `Err(InvalidPattern)` from
  `StringMatcher::match_rule`. This is what hid BUG-039 end-to-end.
  Fixed by adding `StringMatcher::validate(rule)` which is called
  from `Compiler::validate_and_compile` to surface regex compile
  errors at `compile_str` time. The scan-site `_ => {}` was
  tightened to `Ok(_) => {}` + `debug_assert!` on `Err`.

### Added

- **Triple-quoted patterns (`"""..."""`) for `llm:` rules**, matching
  the Python reference parser. Triple-quoted bodies may span multiple
  lines and contain unescaped `"`, `{`, `}` — useful for prompt
  templates passed to LLM evaluators. Wired through three layers:
  `parser/scanner.rs::Scanner::consume_triple_quoted_string`,
  `parser/mod.rs::remove_comments` (TripleString mode), and
  `parser/mod.rs::split_rules` (3-byte lookahead in the brace
  counter).
- **`syara/src/parser/scanner.rs`** — hand-rolled per-line tokenizer.
  Single file (~360 lines) with byte-indexed cursor, line tracking
  for parse errors (BUG-023 spirit), and 20 unit tests including a
  parity test against `parser/mod.rs::split_rules`'s regex
  consumption. Methods: `consume_identifier`,
  `consume_quoted_string`, `consume_triple_quoted_string`,
  `consume_regex_literal`, `consume_modifier_word`,
  `consume_kv_value`.

### Changed

- **Parser internals: per-line regex tokenizers replaced with a
  hand-rolled scanner.** This is an intentional divergence from the
  Python reference (`../syara-rust-port/syara/parser.py`) which has
  the same bug class and produces silent miscompilation under the
  same inputs. CLAUDE.md's porting-discipline waiver applies — the
  divergence is documented in the scanner module doc-comment.
- **`parser/sections.rs`** restructured: deleted five
  `LazyLock<Regex>` constants; each section parser now drives a
  `Scanner`. Added `parse_quoted_section_line` shared helper for
  similarity / phash / classifier sections. The LLM section is
  parsed as a single stream (rather than line-by-line) to support
  multi-line triple-quoted bodies.

## [0.3.1] — 2026-04-24

### Fixed

- **BUG-038 — `OpenAiChatEvaluator` reasoning-mode escape hatch.** The
  request body in `OpenAiChatEvaluator::evaluate` was a fixed `json!`
  literal with no way to send `reasoning_effort` or any
  server-specific knob. Reasoning-on LMStudio loadouts (the dominant
  2026 local default) consumed the entire `max_tokens` budget inside
  `reasoning_content` and emitted empty `choices[0].message.content`,
  causing all `llm:` rules to silently fail. See
  `tasks/04-24-2026_BUGS.md` for the full close-out.

### Added

- **`OpenAiChatEvaluatorBuilder::reasoning_effort(impl Into<String>)`**
  to set the OpenAI-compatible
  `reasoning_effort: "none" | "low" | "medium" | "high"` body field.
- **`OpenAiChatEvaluatorBuilder::disable_reasoning_effort()`** to omit
  the field entirely (escape hatch for strict servers that 400 on
  unknown body keys).
- **`OpenAiChatEvaluatorBuilder::extra_body(key, serde_json::Value)`**
  forward-compatible escape hatch for server-specific knobs (`top_p`,
  `seed`, `response_format`, etc.). Inserted last, so it overrides
  explicit fields.
- **`OpenAiChatEvaluator::DEFAULT_REASONING_EFFORT`** constant
  (`"none"`).
- **`SYARA_LLM_REASONING_EFFORT` env var** read by
  `resolve_openai_env_defaults`. Empty string = disable; any other
  value = pass through to the builder. Honours `SYARA_LLM_NO_ENV=1`.

### Changed

- **Default request body now includes `reasoning_effort: "none"`.**
  Reasoning-mode servers stop thinking and emit a final answer without
  configuration; permissive servers ignore the unknown field. Strict
  servers can opt out via `disable_reasoning_effort()` or
  `SYARA_LLM_REASONING_EFFORT=""`.
- Body construction extracted from the inline `json!` literal in
  `evaluate` to a `pub(crate) fn build_request_body` for
  unit-testability.

## [0.3.0] — 2026-04-22

### Added

- **Condition DSL — `#pattern` count operators.** `#ident` evaluates to the
  number of matches for the pattern identifier as an `i64` expression, matching
  YARA's pattern-count syntax. Useful for multi-turn transcript detection and
  threshold conditions (e.g. `#user >= 2 and #assistant >= 1`).
- **Condition DSL — comparison and arithmetic operators.** `==`, `!=`, `<`,
  `<=`, `>`, `>=` (non-associative — chained comparisons like `a < b < c` are
  rejected with a parse error; use `and`). `+` and `-` binary arithmetic and
  unary `-` on integer expressions.
- **Post-parse type checking.** Malformed conditions (e.g. `$s1 + 2`) now
  surface as `SyaraError::ConditionParse("type error: ...")` at compile time
  rather than evaluating silently at scan time.
- **Regex `(?m)` inline flag parity.** Explicit test asserting Rust-regex
  inline flags (`(?m)`, `(?s)`, `(?i)`) compose with `string:` modifiers.
- **`tasks/YARA-X-PARITY-GAPS.md`.** One-shot audit of the gaps between
  SYARA-X's condition DSL and YARA-X's, grouped by impact on LLM-content rule
  authoring.

### Changed

- **`is_identifier_needed` pessimism.** When an LLM identifier appears inside
  a `#count` subtree, the short-circuit optimization is skipped
  (pessimistic-but-correct); otherwise the existing boolean-substitution logic
  applies. The LLM still runs when it cannot be proven unnecessary.
- **Compiler identifier-scan regex.** Changed from `\$\w+` to `[#$]\w+` so
  `#ident` is validated against the declared-pattern set. `#name` normalizes
  to `$name` before lookup — both sigils share one identifier namespace.
- **`ROADMAP.md`.** New "YARA-X parity gaps (condition DSL)" section
  documenting deferred items: `*` / `/` / `%` arithmetic (blocked on
  set-wildcard token collision), `@pattern[i]` offset, `!pattern[i]` length,
  `KB` / `MB` integer suffixes, chained `not not x`.

### Deferred (not implemented; tracked in `tasks/YARA-X-PARITY-GAPS.md` and `ROADMAP.md`)

- Arithmetic `*` / `/` / `%`.
- `@pattern[i]` / `!pattern[i]` subscript expressions.
- Match anchors `at N` / `in (lo..hi)`.
- `for any / all of` and `for … in (range)` iteration.
- `N of (…)` / percentage quantifiers.
- Boolean literals `true` / `false`, `defined x`.
- String infix ops (`contains`, `matches`, `startswith`, …).
- `private` / `global` rule flags, `import`, `include`.
- Hex patterns, `xor` / `base64` / `base64wide` modifiers.

## [0.2.0] — 2026-04-19

Initial tagged release. Covers phases 1–7 of the port: parser, compiler,
execution engine, semantic matcher (`sbert`, `sbert-onnx`), classifier
(`classifier`, `classifier-onnx`), LLM evaluator (`llm` — OpenAI-compatible
and native Ollama paths), perceptual-hash matcher (`phash`), and C FFI
(`capi`). Local-LLM backend (`burn-llm` / `burn-llm-gpu`) walled off pending
candle-rs migration (see `ROADMAP.md`).

[0.3.1]: https://github.com/gatewaynode/syara-x/releases/tag/v0.3.1
[0.3.0]: https://github.com/gatewaynode/syara-x/releases/tag/v0.3.0
[0.2.0]: https://github.com/gatewaynode/syara-x/releases/tag/v0.2.0
