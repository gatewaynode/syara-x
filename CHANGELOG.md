# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until `1.0.0`, minor-version bumps may include breaking changes to the DSL or
public API.

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
