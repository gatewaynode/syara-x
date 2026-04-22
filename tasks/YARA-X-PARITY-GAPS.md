# YARA-X parity gaps

**Audit date:** 2026-04-22
**Reference:** `../yara-x/parser/src/{tokenizer/tokens.rs, ast/mod.rs, parser/mod.rs}`
**Status:** Research-only. Not planned work. Pick up items from here only if the `llm_context_shield` app (primary consumer of SYARA-X) requires them.

This is a one-shot inventory of what YARA-X supports that SYARA-X does not, scored for impact on realistic LLM-content rule authoring. Avoids re-running the cross-reference research.

---

## What SYARA-X has today (2026-04-22)

**Rule header:** `rule <name> [: tag1 tag2] { … }` — no `global` / `private` flags.
**Top-level items:** `rule` only — no `import`, no `include`.
**Pattern sections:** `strings` (text literals + `/regex/[i]`), plus SYARA extensions `similarity` / `phash` / `classifier` / `llm`.
**Modifiers:** `nocase`, `wide`, `ascii`, `dotall`, `fullword`.
**Condition AST:** `$id`, `#id` (count), integer literal, unary `-`, `+` / `-`, `== != < <= > >=`, `and` / `or` / `not`, `any / all of (them | $a,$b | $prefix*)`.
**Meta:** `key = "value"` only — string values; no int / bool / float.

---

## Gaps grouped by impact

★ = likely helps rule authors. ○ = YARA-X has it, no cited SYARA demand yet.

### Group A — Cheap wins, high value for LLM-content rules ★

| Gap | YARA-X syntax | Effort | Notes |
|---|---|---|---|
| Boolean literals | `true` / `false` | Trivial | Add `Token::True` / `Token::False`, `Expr::Bool`. Enables placeholder rules (`condition: true`) and disabled rules (`x and false`). |
| String infix ops | `lhs contains "foo"`, `icontains`, `startswith`, `istartswith`, `endswith`, `iendswith`, `iequals`, `matches /re/` | Moderate | Requires `Value::Str(…)`. Useful for meta-value comparisons and later pattern-capture comparisons. Post-parse type-check gates it like counts did. |
| `defined x` | Unary `defined` | Small | Returns bool; useful for optional LLM / classifier identifiers. |
| Meta int / bool / float | `key = 42`, `key = true`, `key = 1.5` | Small | `Rule.meta: HashMap<String, String>` → `HashMap<String, MetaValue>`. Affects FFI surface. |

### Group B — Match anchoring (medium value) ★

| Gap | YARA-X syntax | Effort | Notes |
|---|---|---|---|
| `$a at N` | `$s at 100` | Moderate | `MatchDetail.start_pos: Option<usize>` exists. Grammar needs `MatchAnchor`. |
| `$a in (lo..hi)` | `$s in (0..1024)` | Moderate | Same plumbing as `at`. Introduces **ranges**, which unlock `for … in`. |
| `@pattern[i]`, `!pattern[i]` | Offset / length of i-th hit | Moderate | Already in ROADMAP as deferred. Subscript grammar + `MatchDetail.end_pos`. |
| `filesize` | `filesize < 1024` | Small | `scan(text)` has no file; `scan_file()` does. Needs a `ScanContext`. |
| Integer size suffixes | `100KB`, `5MB` | Trivial | Lexer-level. Already in ROADMAP. |

### Group C — Iteration and quantification ★

| Gap | YARA-X syntax | Effort | Notes |
|---|---|---|---|
| `for any / all of` | `for any of ($a*): ($ at 0)` | Large | Requires `$` anaphoric variable + anchors. Highest-leverage YARA feature still absent — lets rules say "for any of my patterns, require it to be at a line start". |
| `for … in (range / list)` | `for any i in (1..#s): (@s[i] < 100)` | Large | Requires ranges, offset / length expressions, bound variables. Cascades across three subsystems. |
| `N of (…)` quantifier | `2 of ($a, $b, $c)`, `50% of them` | Small-to-moderate | SYARA has only `any` / `all`. Integer / percentage quantifiers drop into `Of` naturally. |

### Group D — Rule organization ○

| Gap | YARA-X syntax | Effort | Notes |
|---|---|---|---|
| `private` / `global` rule flags | Header modifiers | Small | `private` = don't report (usable in other conditions). `global` = must match for any other rule in namespace to match. |
| `import "mod"` | `import "pe"`, `import "math"` | Very large | YARA modules are hundreds of LoC each. Scope as "never unless a specific module like `math` for entropy proves worth it". |
| `include "path.yar"` | File inclusion | Small | Pre-parse file-concat step. |
| `with x = expr : (…)` | Scoped binding | Small-moderate | Trivial desugaring once `for … in` exists. |

### Group E — Binary / byte-level pattern parity ○

Mostly irrelevant to SYARA-X's LLM-content focus.

| Gap | YARA-X syntax | Notes |
|---|---|---|
| Hex patterns | `$h = { 4D 5A ?? [2-5] (01 \| 02) }` | Large. Wildcards, ranges, alternations. Low demand for text / LLM. |
| `xor` modifier | `$s = "foo" xor(0x01-0xFF)` | Moderate. Encoded-malware strings, not LLM content. |
| `base64` / `base64wide` | `$s = "password" base64` | Moderate. Same domain. |
| `entrypoint` | PE-only | Depends on `pe` module. Skip. |

### Group F — Arithmetic / bitwise parity ○

| Gap | Syntax | Notes |
|---|---|---|
| `*` / `/` / `%` | `#a * 2`, `#a % 10` | ROADMAP-deferred. `*`-token collision with set wildcards. |
| Bitwise `& \| ^ ~ << >>` | `filesize & 0xFF` | Uncommon outside modules. |
| Hex / octal integer literals | `0xFF`, `0o77` | Trivial lexer add. |
| Float literals | `1.5`, `2.0` | Needed for `math.entropy() > 7.5`-style rules. Introduces `Value::Float`. |
| `not not x` | Chained negation | ROADMAP-deferred. |

---

## Recommended priority if demand appears

1. **Group A** — `true` / `false`, `defined`, string infix ops, richer meta values. String ops (`contains`, `matches`) are the single biggest expressiveness win.
2. **Group C `N of` quantifier** — small increment on the existing `any` / `all of` parser.
3. **Group B `at` / `in (range)`** — moderate, prereq for `for … in`.
4. **Group C `for` forms** — the big one. Do only after ranges / anchors land.
5. **Group D `private` / `global` / `include`** — cheap organizational features, opportunistic.
6. **Everything else (binary patterns, modules, bitwise, full arithmetic)** — park here, add on concrete `llm_context_shield` demand.

---

## Trigger

Revisit this file when `llm_context_shield` hits a rule it wants to write but can't express in the current SYARA-X DSL. Map the blocked rule back to one of the groups above, then plan. Don't pre-build capability without a concrete rule that needs it.
