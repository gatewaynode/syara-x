# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until `1.0.0`, minor-version bumps may include breaking changes to the DSL or
public API.

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

[0.3.0]: https://github.com/gatewaynode/syara-x/releases/tag/v0.3.0
[0.2.0]: https://github.com/gatewaynode/syara-x/releases/tag/v0.2.0
