# Continuity Notes

Working notes for resuming after context compacts. Update proactively
before any compact. Wipe stale entries when work concludes.

---

## Active session — 2026-05-10

### What we just finished

**v0.4.0 implementation (BUG-039 + BUG-040 fix)** — all 6 phases of plan
`/Users/john/.claude/plans/shimmering-beaming-hennessy.md` complete.
Code lives in 8 modified files (uncommitted). Tests green: 154 default,
271 with `--all-features`, clippy clean, `cargo publish --dry-run -p
syara-x --allow-dirty` passes.

### Current state — UNCOMMITTED

`git status` shows 8 modified files + 2 new files + several new test
fixtures. Nothing is staged. **The user has NOT asked for a commit
yet.** Do not commit unless asked.

```
modified:  Cargo.toml                                    # 0.3.1 → 0.4.0
modified:  CHANGELOG.md                                  # [0.4.0] entry
modified:  ARCHITECTURE.md                               # parser section
modified:  syara/src/compiled_rules.rs                   # silent swallow → debug_assert
modified:  syara/src/compiler.rs                         # StringMatcher::validate hook
modified:  syara/src/engine/string_matcher.rs            # new validate()
modified:  syara/src/parser/mod.rs                       # triple-quote + 5 tests
modified:  syara/src/parser/sections.rs                  # rewritten — scanner-driven
modified:  syara/tests/integration.rs                    # BUG-039 + BUG-040 + corpus
modified:  tasks/BUGS.md                                 # cleared open list
modified:  tasks/lessons.md                              # [^X]* lesson
new file:  syara/src/parser/scanner.rs                   # the new lexer
new file:  tasks/05-10-2026_BUGS.md                      # close-out archive
new file:  tasks/CONTINUITY.md                           # this file
```

### Likely next steps after compact

1. **User reviews diff and asks for a commit.** Use a single
   focused commit (per project convention) referencing BUG-039 +
   BUG-040 + v0.4.0. Co-author tag with the model id (see CLAUDE.md
   git protocol). Don't auto-commit.
2. **Possibly tag + publish v0.4.0 to crates.io.** Sequence: commit
   first → `git tag v0.4.0` → `cargo publish -p syara-x` →
   `cargo publish -p syara-x-capi`. Requires user authorization for
   the publish step (it's a one-way action).
3. **Downstream consumer follow-up** — `llm_context_shield` has a
   workaround in its Phase 13.5b SESSION_PROTOCOL_COMBINED test
   fixture (Cyrillic homoglyph priming) that can be dropped once
   v0.4.0 is consumed. Tracked in `llm_context_shield/tasks/RULE_FIXES.md`.
   Don't touch that repo from here without explicit ask.

### Critical gotchas if you have to pick this up cold

- **The scanner is `pub(crate)`**, not `pub`. Don't accidentally
  expose it. Same for `parser` module (`pub(crate) mod parser` in
  `syara/src/lib.rs`).
- **Triple-quote support is LLM-section only.** Other sections
  reject `"""` (Python parity). If the user later asks "why doesn't
  triple-quote work in `strings:`?", explain the parity choice.
  Wiring it elsewhere requires updating the brace counter and
  comment stripper as well.
- **`compiled_rules.rs:113-128` now has `debug_assert!(false, ...)`
  in the Err arm.** Release builds silently ignore the error. If
  any future test ever fails with "scanner errored at scan time
  despite eager validate()", a regex evades the eager path —
  investigate the new evasion route, don't relax the assert.
- **The cross-engine YARA-X parity test the plan agent suggested
  was NOT added** to syara-x's test suite. The 4 consumer-corpus
  shape tests in `tests/integration.rs::test_bug039_corpus_*`
  cover the same patterns positively, which is what matters for
  this fix. If user later asks about full YARA-X parity, that's a
  separate bigger effort.
- **The Python reference parser has the SAME bug class** and was
  intentionally diverged from. Documented in `scanner.rs` module
  doc-comment per CLAUDE.md porting-discipline waiver. Don't "fix"
  the divergence back during a future porting pass.

### Out-of-scope follow-ups (file as new bugs if/when needed)

From the plan's "Out-of-scope follow-ups" section:

- **Consolidate the three state machines** (scanner, brace counter,
  comment stripper) onto a single shared lexer crate-internal
  helper. The parity unit test pins their behavioral agreement;
  deduplication is a future cleanup. Not urgent.
- **Triple-quote in non-LLM sections.** Python doesn't support it
  there either; user opted for parity. If a future user wants a
  multi-line quoted string in `strings:` or `meta:`, this is the
  ticket.
- **YARA-X cross-engine parity test corpus** — see plan agent's
  P-A recommendation. Cheap if done from `llm_context_shield` side
  rather than vendoring rules into syara-x.

### Test-count verification (sanity check on resume)

Run these to confirm nothing has regressed since the work was
completed:

```bash
cargo test -p syara-x 2>&1 | grep "test result:"
# Expect:
#   ok. 118 passed; 0 failed; 0 ignored      (lib)
#   ok. 35 passed; 0 failed; 0 ignored       (integration)
#   ok. 1 passed; 0 failed; 0 ignored        (doc)
#   plus 3 empty test files

cargo test -p syara-x --all-features 2>&1 | grep "test result:"
# Expect:
#   ok. 226 passed; 0 failed; 6 ignored      (lib)
#   ok. 35 passed                            (integration)
#   plus 5 feature-gated test files
```

If counts diverge, something else has changed since I last checked.

### Plan file

The original approved plan is at
`/Users/john/.claude/plans/shimmering-beaming-hennessy.md` — it
captures all six phases, user decisions (v0.4.0, BUG-040 in same
PR, triple-quote implemented), and verification commands. Read it
first if context is unclear.

---

## How to use this file going forward

- **Update before any compact.** Capture session-specific state
  that would be hard to reconstruct from `git status` + the plan
  file alone: design choices made mid-session, gotchas, what was
  intentionally out of scope, what the user is likely to ask next.
- **Keep entries dated.** When work concludes (commit lands,
  feature ships, ticket closes), wipe the entry. This file is not
  an archive — `tasks/05-10-2026_BUGS.md` and the dated bug files
  serve that role. Stale continuity entries are worse than none.
- **Mirror, don't duplicate.** If something belongs in
  `tasks/lessons.md` (a generalizable rule), put it there and link
  to it. This file is for in-flight context only.
