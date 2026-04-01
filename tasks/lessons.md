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
