# Lessons Learned

Patterns and corrections captured during development to prevent repeated mistakes.

---

## Rust `regex` crate has no lookahead support

**Rule:** Never use `(?!...)` or `(?=...)` in patterns passed to the `regex` crate.  
**Why:** The crate explicitly does not support look-around assertions (by design, for performance).  
**How to apply:** Use post-match checks on the surrounding string slice instead (e.g., check `haystack[m.end()..].starts_with('*')`).  
*Discovered:* compiler.rs wildcard validation, 2026-04-01.

---

## Python `DefaultCleaner` uses NFKC normalization — don't skip it

**Rule:** When porting a cleaner, check for `unicodedata.normalize` in the Python source before writing the Rust equivalent.  
**Why:** NFKC normalization collapses ligatures, compatibility forms, and fullwidth chars. Omitting it causes test failures on non-ASCII input.  
**How to apply:** Use the `unicode-normalization` crate (`text.nfkc().collect::<String>()`).  
*Discovered:* engine/cleaner.rs, 2026-04-01.

---

## Regex `[^X]*` over a body that contains `\X` is silent truncation

**Rule:** Whenever a regex needs `[^X]*` over a body that contains escapes of `X` (e.g., `[^/]*` over a regex literal whose body can contain `\/`, or `[^"]*` over a string literal whose body can contain `\"`), replace the regex with a hand-rolled scanner.  
**Why:** Regex character classes have no syntax for "any character that isn't `X`, *unless preceded by a backslash*." The class `[^/]` truncates at the first `/` byte regardless of escape — and the failure is silent because what comes after the truncated capture often parses as something benign (modifiers, kv params), so the rule compiles to a malformed-but-valid pattern that returns zero matches forever. Bug only surfaces when input contains the escape — usually long after rules have been authored. Bit us on BUG-039 (`\/` in regex literals) and latent in two other places (`\"` in meta values and kv params).  
**How to apply:** Lex line-oriented DSL syntax with a small hand-rolled scanner that has explicit escape-handling state. The scanner in `syara/src/parser/scanner.rs` is the canonical example. If you find yourself writing a `LazyLock<Regex>` whose body uses `[^X]*`, stop and ask "can `\X` appear inside?" If yes, lex it.  
*Discovered:* `parser/sections.rs` BUG-039 + BUG-040 close-out, 2026-05-10.
