# Continuity Notes

Working notes for resuming after context compacts. Update proactively
before any compact. Wipe stale entries when work concludes.

---

## Active session — 2026-05-10 (lexer polish pass)

### Where we are

v0.4.0 (BUG-039 + BUG-040 + new scanner module) shipped earlier this
session and is committed. Current work is the **polish pass** against
the rough-edges list in `docs/lexer-data-flow.md` (section: "First-pass
rough edges (v0.4.0 — flagged for follow-up polish)").

Three rough-edges resolved this session, each as its own commit-sized
batch:

1. **Missing parity tests** — added scanner ↔ split_rules mirrors for
   string + triple-quote (the regex one already existed); added
   behavioral parity tests for `remove_comments` (regex / string /
   triple-quote bodies preserve comment-like sequences) and
   `split_rules` brace counter (`{` / `}` inside literals don't
   decrement depth). +8 tests. **Committed.**
2. **`ParseError` carried line but not col** — added `col: usize` to
   `SyaraError::ParseError`, with `Display` suffix that hides the 0
   sentinel (used by error sites without a Scanner in scope, e.g.
   rule-header / missing-condition errors in `parse_rule_block`).
   `Scanner::col()` derives column from `pos` via `rfind('\n')` on the
   byte slice — works uniformly for per-line and stream usage. +5
   tests. **Committed.**
3. **`eat_inline_ws` ignored `\r` (CRLF files)** — added `b == b'\r'`
   to the consume condition. Verified end-to-end CRLF parsing for
   per-line sections, modifiers / kv params, LLM stream section, and
   triple-quoted bodies (CRLF inside body preserved raw, matching
   Python "raw body" parity). +5 tests. **About to commit.**

Test counts after the CRLF batch: **136 lib + 35 integration + 1 doc**
(default features). Lib clippy clean. Three pre-existing clippy
warnings in non-touched files (string_matcher.rs:406, models.rs:203,
integration.rs:2) — confirmed unrelated via `git blame`.

### Current uncommitted state — CRLF batch only

```
modified:  docs/lexer-data-flow.md       # removed the resolved CRLF rough-edge entry
modified:  syara/src/parser/mod.rs       # 5 CRLF end-to-end tests
modified:  syara/src/parser/scanner.rs   # eat_inline_ws + 1 unit test
```

User said "I'll commit" — do NOT auto-commit.

### Likely next steps after compact

User wants to **keep polishing the lexer flow**. The dataflow doc's
rough-edges section is the work queue. Remaining items, ordered roughly
by impact:

**Bug-risk tier** (do these first):

- **`parse_llm_section` skip-non-rule-line is over-tolerant**
  (`sections.rs:280-291`). Typo `pp1 = "..."` (missing `$`) silently
  skips the whole rule. Decide whether to error or warn.
- **Brace-counter `b'/'` arm requires `=` immediately before**
  (`mod.rs:213-220`). Thin invariant — adding any new operator near
  `/` would silently break this. Could be hardened with a comment,
  an assertion, or refactored to be more robust.
- **Flag-rejection is BREAKING without migration story** — `/abc/sm`
  errors hard. CHANGELOG calls it out under v0.4.0 BREAKING but the
  migration ("use inline flags `(?s)`, `(?m)`") is only mentioned in
  the v0.3.0 entry. Add migration note to v0.4.0 BREAKING section.

**Design tier**:

- **Asymmetric line tracking across `consume_*` methods** — only
  `consume_triple_quoted_string` and `bump()` track `\n`.
  `consume_quoted_string` and `consume_regex_literal` don't. Latent
  for the LLM-stream parser if anyone embeds a literal `\n` in a
  single-quoted body. Quick fix: have the consume methods bump
  through `bump()` instead of raw `pos += 1` for the byte-by-byte
  loop. Touches 4 methods.
- **`idx + 1` is section-relative line number** — error messages
  report "line 3" meaning "line 3 of the strings: section," not
  the source-file line. Real fix needs the section parsers to track
  absolute line numbers (compute the section start line in
  `section_content`, pass it down to per-line scanner constructors).
  Bigger change — touches `section_content` signature.
- **Three state machines** — structural design issue (test
  scaffolding is now complete via parity tests). Consolidating into
  a single shared lexer crate-internal helper is the eventual end
  state. Bigger refactor; defer.
- **`unescape_string` duplicates `consume_quoted_string`'s escape
  table** — one is `#[cfg(test)]` only, used by one test in
  `parser/mod.rs::tests::test_unescape_string_sequences`. Either
  delete and rewrite the test against `Scanner::consume_quoted_string`,
  or add a parity test. Quick.

**Hygiene tier**:

- **`consume_kv_value` accepts any non-ws bareword** —
  `cleaner=ñ$@%` parses cleanly. Add positive-shape validation if
  desired.
- **Empty regex body permitted (`//`)** — downstream `Regex::new("")`
  matches everywhere. Decide whether to reject at parse or accept.
- **`parse_quoted_section_line` over-tolerant skip** — silent
  `Ok(None)` for any non-`$id =` line. Add "looks-like-attempted-rule"
  warning.
- **Duplicated section-end keyword lists** in
  `sections.rs:73, 109, 161, 199, 232, 269` — extract a `const
  SECTION_ORDER`. Easy win.

When picking next batch, propose 1-3 related items (e.g., "the
bug-risk tier" or "the quick wins") and let the user choose. Don't
attempt the entire list as one mega-PR.

### Critical gotchas if you pick this up cold

- **The polish pass is iterative — one batch per commit.** User has
  been committing between batches. Do NOT bundle multiple rough-edges
  into a single PR without explicit ask.
- **Parity tests pin the three-state-machine invariant.** Before
  refactoring any of `Scanner` / `remove_comments` / `split_rules`,
  run `cargo test -p syara-x --lib parser::scanner::tests::parity`
  and `cargo test -p syara-x --lib parser::tests::test_split_rules`
  to confirm baseline. Any state-machine change MUST keep these
  green.
- **`ParseError` shape changed this session** — `{ line, col,
  message }` not `{ line, message }`. If you ever see code that
  matches on the old 2-field shape, it's stale.
- **Pre-existing clippy warnings are NOT mine.** Run `cargo clippy
  -p syara-x --lib` to filter to only the touched-file warnings.
  Don't fix the pre-existing ones unless explicitly asked.
- **Per-line section parsers iterate `content.lines()`** which
  strips both `\n` and `\r\n` line terminators. The CRLF fix matters
  for the LLM-section stream parser (which uses a single Scanner
  across newlines) and for any `\r` that sneaks into per-line input
  via `trim()` edge cases.
- **`Scanner::col()` is byte-indexed, not char-indexed.** Safe for
  ASCII inputs (which `.syara` files are in practice). Multi-byte
  UTF-8 inside a literal would give a slightly off col number — flag
  if the user ever brings non-ASCII rule files.

### Test-count verification (sanity check on resume)

```bash
cargo test -p syara-x 2>&1 | grep "test result:"
# Expect (after the CRLF batch is committed):
#   ok. 136 passed; 0 failed; 0 ignored      (lib)
#   ok. 35 passed                            (integration)
#   ok. 1 passed                             (doc)
#   plus 3 empty test files

cargo clippy -p syara-x --lib -- -D warnings  # should be clean
```

If lib counts diverge, something else has changed since I last
checked. Each polish batch typically adds ~5 tests; if the count is
off by more than that, look at the most recent commits.

### Reference docs

- **`docs/lexer-data-flow.md`** — single source of truth for the
  rough-edges work queue. Update entries (or remove them) as items
  ship.
- **`tasks/05-10-2026_BUGS.md`** — close-out archive for BUG-039 +
  BUG-040 (the original v0.4.0 work). Don't touch unless reopening.
- **`/Users/john/.claude/plans/shimmering-beaming-hennessy.md`** —
  v0.4.0 plan. Historical only.

---

## How to use this file going forward

- **Update before any compact.** Capture session-specific state
  that would be hard to reconstruct from `git status` + the dataflow
  doc alone: what was just committed, what's queued, design choices
  that informed the current direction.
- **Wipe entries when work concludes.** When the polish pass wraps
  (or transitions to the next major effort), this whole entry goes.
  Stale continuity entries are worse than none.
- **Mirror, don't duplicate.** Generalizable lessons go in
  `tasks/lessons.md`. Rough-edge tracking lives in
  `docs/lexer-data-flow.md`. This file is for in-flight session
  context only.
