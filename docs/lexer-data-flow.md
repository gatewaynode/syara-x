# Lexer dataflow — `.syara` rule parsing

End-to-end trace of how `.syara` source becomes `Vec<Rule>`, centered on
the new per-line scanner introduced in v0.4.0 (BUG-039 / BUG-040).
Companion doc to the BUG-039 close-out at `tasks/05-10-2026_BUGS.md`.

```mermaid
flowchart TD
    A["SyaraParser::parse_str(src)<br/>parser/mod.rs:48-52"] --> B["remove_comments(src)<br/>parser/mod.rs:69-176<br/>state machine: Normal · String · TripleString · Regex"]
    B --> C["split_rules(cleaned)<br/>parser/mod.rs:180-260<br/>brace counter w/ inline string/regex/triple consumption"]
    C --> D["parse_rule_block(block)<br/>parser/mod.rs:264-313<br/>HEADER_CAPTURE_RE → name + tags"]
    D --> E["section_content(body, name)<br/>parser/sections.rs:24-44"]
    E --> F{Section type?}
    F -- meta · strings · similarity · phash · classifier --> G["Per-line Scanner<br/>Scanner::new(line, idx+1)<br/>parser/sections.rs:67-262"]
    F -- llm --> H["Whole-stream Scanner<br/>parser/sections.rs:264-318<br/>spans newlines for triple-quote"]
    F -- condition --> I["String trim only<br/>parser/sections.rs:320-326"]
    G --> J["consume_identifier<br/>expect_byte('=')<br/>consume_quoted_string OR consume_regex_literal<br/>collect_modifiers · collect_kv_params"]
    H --> K["consume_identifier<br/>expect_byte('=')<br/>peek_str(3)=='\"\"\"' ? consume_triple_quoted : consume_quoted<br/>collect_kv_params_until_newline"]
    J --> L["Vec&lt;Rule&gt;"]
    K --> L
    I --> L
    L --> M["Compiler::validate_and_compile<br/>compiler.rs:20-106<br/>duplicate-id · StringMatcher::validate · condition::parse"]
    M --> N{Validate OK?}
    N -- Err --> X["SyaraError::InvalidPattern · DuplicateIdentifier · ConditionParse<br/>(returned to caller)"]
    N -- Ok --> O["CompiledRules"]
    O --> P["scan(text)<br/>compiled_rules.rs:85-193<br/>cost-ordered: string → similarity → phash → classifier → llm"]
    P --> Q{Match?}
    Q -- Err in string arm --> Y["debug_assert! + ignore<br/>compiled_rules.rs:124-129"]
    Q -- Ok --> R["Vec&lt;Match&gt;"]
```

## Numbered walkthrough

### 1. Entry point

**A.** `SyaraParser::parse_str(content)` (`syara/src/parser/mod.rs:48-52`)
is the single public entry. `parse_file` (mod.rs:31-39) is a thin wrapper
that reads the file then delegates here.

**B.** Two transformations are applied in order: `remove_comments` then
`split_rules`. Both run over the entire input as one pass — there is no
streaming / chunked variant.

### 2. Comment stripping

**A.** `remove_comments` (`parser/mod.rs:69-176`) walks the source as a
`Vec<char>` (UTF-8 char-indexed), running a four-state machine:

  - `Mode::Normal` — outside any literal. Recognizes `//` line comments,
    `/* */` block comments, the `"`/`"""` openers, and the `=`-preceded
    `/` that opens a regex literal.
  - `Mode::String` — inside `"..."`. Honors `\X` escapes for any X
    (one-token swallow). Closes on unescaped `"`.
  - `Mode::TripleString` — inside `"""..."""`. Body captured raw (no
    escape processing). Closes only on three consecutive `"`.
  - `Mode::Regex` — inside `/.../`. Honors `\X` escapes. Closes on
    unescaped `/`, then consumes alphabetic flag chars.

**B.** Output is a new `String` with comments stripped and literals
preserved verbatim. Line breaks within literals are kept (so downstream
line counting still works for multi-line triple-quoted bodies).

**C.** This is the **first** of three escape-aware state machines in the
parser. See "Key invariants" — they must agree byte-for-byte on what
counts as "inside a literal."

### 3. Rule splitting

**A.** `split_rules` (`parser/mod.rs:180-260`) operates byte-indexed
(`content.as_bytes()`), driven by `HEADER_RE` (mod.rs:21-23) to find
each `rule <name> { ... }` opener.

**B.** Inside each rule block, a brace counter tracks `{`/`}` to find
the matching close. Within the brace counter, the same literal
classification work happens again — string / triple-string / regex
consumption is inlined directly into the `b'"'` and `b'/'` arms
(mod.rs:204-237). Inner `{` and `}` inside any literal must NOT count
toward depth.

**C.** Triple-quote handling: the `b'"'` arm peeks three bytes ahead
(mod.rs:204-218); if they're `"""`, consume to closing `"""`,
otherwise treat as a single-quote string.

**D.** Regex handling: the `b'/'` arm walks **backwards** over inline
whitespace (mod.rs:223-228) to check that the preceding non-ws byte is
`=`. Only then is `/` treated as a regex opener. Without this guard,
arithmetic-style `/` (not currently in the DSL) and division-like
patterns inside string contents would mis-trigger regex mode.

**E.** Output: `Vec<String>`, one element per rule block, **including**
the `rule <name> { ... }` braces.

### 4. Rule block parsing

**A.** `parse_rule_block` (`parser/mod.rs:264-313`) extracts the rule
name and tags via `HEADER_CAPTURE_RE`, then slices the body between the
first `{` and last `}`.

**B.** Each section is dispatched in fixed order: `meta` → `strings` →
`similarity` → `phash` → `classifier` → `llm` → `condition`. Section
extraction goes through `section_content`
(`parser/sections.rs:24-44`), which compiles a per-section header regex
on demand (cached behind `LazyLock<Mutex<HashMap>>`).

**C.** "Patterns without a condition" rejection (mod.rs:299-309) fires
after all sections parse — BUG-021.

### 5. Per-line section parsing (meta · strings · similarity · phash · classifier)

**A.** Each parser iterates `content.lines()`, trims whitespace, and
constructs a fresh `Scanner` per line: `Scanner::new(line, idx + 1)`
(e.g. `parser/sections.rs:84` for meta).

**B.** Standard shape (`parse_quoted_section_line`,
`parser/sections.rs:329-346`):
  - `consume_identifier` — `$word` or bare `word`
  - `eat_inline_ws`
  - `expect_byte(b'=')`
  - `eat_inline_ws`
  - `consume_quoted_string` (similarity / phash / classifier require
    quoted strings only)
  - `collect_kv_params_until_newline` — `key=value` pairs

**C.** `parse_strings_section` (sections.rs:103-156) is the only per-line
parser that branches on `peek_byte()` to accept `/regex/[flags]` in
addition to `"quoted"`. The `i` flag becomes an implicit
`Modifier::NoCase`; modifier words (`nocase`, `wide`, `fullword`,
`dotall`) follow the pattern via `collect_modifiers` (sections.rs:351-372).

**D.** Tolerance: a line that doesn't match the expected shape returns
`Ok(None)` from `parse_quoted_section_line` and is silently skipped
(sections.rs:336-345). Same legacy behavior as the prior
`SECTION_LINE_RE`.

### 6. LLM section parsing (whole-stream)

**A.** `parse_llm_section` (`parser/sections.rs:264-318`) is the only
section that constructs a single `Scanner` for the whole section content
(`Scanner::new(content, 1)`, sections.rs:273), so triple-quoted bodies
can span newlines.

**B.** Loop body:
  - `skip_ws_and_newlines` (sections.rs:417-424)
  - If `peek_byte() != Some(b'$')` — bump until newline and continue
    (lenient skip, sections.rs:280-290)
  - Otherwise: identifier · `=` · pattern · kv params

**C.** Pattern dispatch (sections.rs:294-303):
  - `peek_str(3) == "\"\"\""` → `consume_triple_quoted_string` (raw body)
  - `peek_byte() == Some(b'"')` → `consume_quoted_string` (escape-processed)
  - else: `SyaraError::ParseError`

### 7. Compile-time validation

**A.** `Compiler::validate_and_compile` (`syara/src/compiler.rs:20-106`)
runs after parsing. Per-rule, per-pattern checks:
  - Duplicate-identifier detection across all pattern types
  - **Eager regex compile** via `StringMatcher::validate(r)?` for every
    `StringRule` (compiler.rs:32-34) — surfaces malformed regexes as
    `SyaraError::InvalidPattern` here, not silently at scan time.
    This is the BUG-040 fix.
  - Condition string → `Expr` AST via `condition::parse`

**B.** `StringMatcher::validate` (`syara/src/engine/string_matcher.rs:41-50`)
constructs a throwaway matcher and forces it through `compile`; if the
rule has the `Wide` modifier, also forces it through `match_wide("")`
to exercise the wide-mode regex construction path.

### 8. Scan-time execution

**A.** `CompiledRules::execute_rule` (`syara/src/compiled_rules.rs:85-193`)
runs matchers in cost-order: string → similarity → phash → classifier
→ llm.

**B.** String-match arm (compiled_rules.rs:118-131):
```rust
match string_matcher.match_rule(string_rule, text) {
    Ok(hits) if !hits.is_empty() => { /* record */ }
    Ok(_) => {}
    Err(e) => {
        debug_assert!(false, "string_matcher errored at scan time despite eager validate(): {e}");
    }
}
```
The `Err` arm is the residual safety net after BUG-040: release builds
silently ignore, debug builds panic. Eager validate at compile time is
the primary guard.

**C.** LLM short-circuit (compiled_rules.rs:167-185): before running an
LLM evaluator, `condition::is_identifier_needed` is consulted —
optimistically assume the LLM matches and check whether the condition
would be true; skip the call if not. (Pessimistic for `#count` subtrees
since v0.3.0.)

## Variations

| Variant | Where it diverges |
|---|---|
| **Strings section regex literal** | `sections.rs:131-141` — `consume_regex_literal` returns `(body, flags)`; `i` flag becomes implicit `Modifier::NoCase`; non-`i` flags rejected (BREAKING from v0.3). |
| **LLM section** | `sections.rs:264-318` — single Scanner across whole section; supports `"""..."""`; no regex literal branch (LLM patterns are quoted strings only). |
| **Triple-quote** | LLM section only. Other sections fall through to `consume_quoted_string` and would consume the empty `""` then error on the next `"`. |
| **Condition section** | `sections.rs:320-326` — bypasses Scanner entirely; whole content trimmed and stored as a string for `condition::parse` later. |
| **Multi-line patterns** | Only triple-quoted bodies in the LLM section. All other section parsers iterate `content.lines()`, so a literal `\n` byte inside a non-LLM pattern would terminate the line at the iteration layer before the Scanner sees it. |

## Error contract

| Source | Error variant | Surfaces at |
|---|---|---|
| Unterminated quoted/regex/triple-quoted literal | `SyaraError::ParseError { line, col, message }` | parse-time |
| Unknown regex flag (anything other than `i`) | `SyaraError::ParseError` | parse-time (BREAKING from v0.3) |
| Missing `=` after identifier | `SyaraError::ParseError` (LLM) / silent skip (others) | parse-time |
| Patterns present but no condition section | `SyaraError::ParseError` | parse-time (BUG-021) |
| Duplicate identifier within a rule | `SyaraError::DuplicateIdentifier` | compile-time |
| Malformed regex pattern | `SyaraError::InvalidPattern` | compile-time (BUG-040 — was silent scan-time before v0.4) |
| Condition references undeclared identifier | `SyaraError::ConditionParse` | compile-time |
| String matcher errors at scan time | `debug_assert!` (panic in debug, silent ignore in release) | scan-time |

## First-pass rough edges (v0.4.0 — flagged for follow-up polish)

Severity tags: **[design]** = architectural smell, **[bug-risk]** =
could bite under specific input, **[hygiene]** = code-quality only.

### Three state machines (test scaffolding now complete) — [design]

`scanner.rs::consume_regex_literal`, `mod.rs::remove_comments`
(`Mode::Regex`), and `mod.rs::split_rules` (the `b'/'` arm) all
classify "what's inside a regex literal." Same problem for strings (3
sites) and triple-quotes (3 sites).

Parity test coverage (added since first-pass):
- Scanner ↔ `split_rules` for regex / string / triple-quote bodies
  (mirror tests, `scanner.rs::tests::parity_with_split_rules_*`)
- Scanner ↔ `remove_comments` for regex / string / triple-quote bodies
  (behavioral end-to-end tests in `mod.rs::tests`)
- `split_rules` brace-counter parity for `}` inside string and
  triple-quote bodies (regex `{n,m}` covered by an existing test)

The remaining design issue is structural — three independently-
maintained state machines vs. one shared lexer crate-internal helper.
The parity tests pin behavioral agreement so the consolidation can
proceed safely.

### `idx + 1` is "section-relative" line number — [design]

Per-line section parsers pass `Scanner::new(line, idx + 1)`. So an error
on the 3rd meta line says "line 3" — relative to the **section body
slice**, not the source file. Real file line is `meta:` keyword line +
idx. Same as old code, but the new lexer surfaces line numbers
prominently in errors, making the misleading number more visible.

### `unescape_string` duplicates `consume_quoted_string` — [design]

`scanner.rs:295-318` is `#[cfg(test)] pub(super)` and exists for one
test in `parser/mod.rs::tests`. It duplicates `consume_quoted_string`'s
escape table verbatim. If one drifts, the existing test passes against
the divergent helper and silently masks the drift. Either delete and
rewrite the test against `Scanner::consume_quoted_string`, or add a
parity test.

### `consume_kv_value` accepts any non-ws bareword — [hygiene]

`scanner.rs:268-285`: no positive validation of bareword shape.
`cleaner=ñ$@%` parses cleanly. Compared to the old `KV_PARAMS_RE` which
had a stricter shape, this is more permissive. Probably fine — the
value is consumed downstream by `f64::parse` or treated as a string
registry key — but worth noting that "bareword" is "any non-ws,
non-`"` run."

### Empty regex body permitted (`//`) — [hygiene]

`scanner.rs:401-405` test pins `consume_regex_literal("//") → Ok((""", []))`.
Downstream, `Regex::new("")` matches the empty string at every position
— i.e. matches everywhere. No test asserts what happens after an empty
regex hits `StringMatcher::validate`. Likely benign; flag for
consideration.

### Flag-rejection is BREAKING without migration story — [bug-risk]

`scanner.rs:251-267` rejects any flag char other than `i`. Old code
silently dropped `s`, `m`, `x`, etc. `CHANGELOG.md` calls this out, but
the migration ("use inline flags" — `(?s)`, `(?m)`) is only mentioned
in the v0.3.0 entry, not v0.4.0 BREAKING. A user upgrading 0.3 → 0.4
with `/abc/sm` rules will get a hard error and have to rediscover the
inline-flag idiom.

### `parse_quoted_section_line` over-tolerant skip — [hygiene]

`sections.rs:329-346` returns `Ok(None)` for any non-`$id =` line.
Means `$$foo = "bar"`, `foo = "bar"` (missing `$`), and totally
unrelated lines all silently skip. Trade-off: tolerance vs
typo-detection. Cheap to add a "looks-like-attempted-rule" warning now
that we have a real lexer.

### Duplicated section-end keyword lists — [hygiene]

`sections.rs:73, 109, 161, 199, 232, 269` — every parser hard-codes the
next-sections list. Adding a new section means editing 6 places. Easy
win to extract a `const SECTION_ORDER: &[&str]` and slice from it.

### Brace-counter `b'/'` arm requires `=` immediately before — [bug-risk]

`mod.rs:213-220` walks back over spaces/tabs and only enters regex mode
if the preceding non-ws byte is `=`. The DSL only uses `=` so this is
fine; flagging as a "thin invariant" — adding any new operator near
`/` would silently break this.

## Key invariants

- **Three state machines must agree on literal classification.** Only
  one pair has a parity test (regex). Strings and triple-quotes are
  tested only end-to-end through `SyaraParser::parse_str`.
- **Per-line Scanner instances are isolated.** Each
  `Scanner::new(line, idx+1)` resets `pos` to 0 and `line` to a
  section-relative number. Don't reuse a Scanner across lines without
  re-thinking line tracking.
- **Eager regex validation is the only safety net for BUG-040.**
  `compiled_rules.rs:124` is a `debug_assert!` — release builds
  silently ignore scan-time string-matcher errors. If any future code
  path bypasses `Compiler::validate_and_compile` (dynamic rule
  injection, alternative compile path), regex errors become silent
  again.
- **`split_rules` brace counter assumes `=` is the only op preceding a
  regex literal.** A new infix operator near `/` would silently break
  rule splitting for any regex on its right side.
- **Triple-quote support is wired in 3 layers.** `scanner.rs` +
  `remove_comments` + `split_rules` all need to agree. No cross-layer
  parity test exists.
- **Triple-quote bodies are captured raw (no escape processing) in all
  3 layers.** Matches Python reference. If escape processing is ever
  added to one layer, all three must change.
- **Section-end detection is keyword-list based, not structural.**
  Adding a new section requires editing every other section's
  `next_sections` list. Currently 6 places.
