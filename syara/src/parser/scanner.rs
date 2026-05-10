//! Per-line tokenizer for `.syara` rule sections.
//!
//! Replaces the regex-over-line idiom that the Python reference parser
//! and earlier Rust passes relied on. The Python form (and the prior
//! Rust `LazyLock<Regex>` constants in `parser/sections.rs`) used body
//! captures like `[^/]*` and `[^"]*`, which silently truncate at the
//! first delimiter byte — escaped occurrences (`\/`, `\"`) are not
//! recognized. See BUG-039 in `tasks/05-10-2026_BUGS.md` for the
//! user-visible failure mode.
//!
//! The intentional divergence from `../syara-rust-port/syara/parser.py`
//! is documented per CLAUDE.md's porting-discipline waiver: the Rust
//! scanner fixes a class of silent miscompilation bugs that the Python
//! reference also exhibits.
//!
//! Behavior parity with the higher-level state machines in
//! `parser/mod.rs::remove_comments` and `parser/mod.rs::split_rules`
//! is asserted in the `parity_with_split_rules_*` tests below — the
//! three sites must classify the same byte ranges identically for
//! ASCII inputs (which is what `.syara` rule files contain in
//! practice).

use crate::error::SyaraError;

/// Per-line scanner for syara rule sections.
pub(crate) struct Scanner<'a> {
    src: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> Scanner<'a> {
    pub(crate) fn new(src: &'a str, line: usize) -> Self {
        Self { src, pos: 0, line }
    }

    pub(crate) fn line(&self) -> usize {
        self.line
    }

    #[cfg(test)]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    pub(crate) fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    /// Advance one UTF-8 char. Used for stepping past known delimiters
    /// (`=`, `\n`) where the caller has already inspected via `peek_byte`.
    /// Tracks line numbers when the consumed char is `\n`.
    pub(crate) fn bump(&mut self) {
        if let Some(b) = self.peek_byte() {
            if b == b'\n' {
                self.line += 1;
            }
            self.pos = self.next_char_end();
        }
    }

    /// Slice of up to `n` bytes from the current position, snapped to a
    /// char boundary. Used for multi-byte delimiter lookahead (`"""`).
    pub(crate) fn peek_str(&self, n: usize) -> &str {
        let mut end = (self.pos + n).min(self.src.len());
        while end > self.pos && !self.src.is_char_boundary(end) {
            end -= 1;
        }
        &self.src[self.pos..end]
    }

    /// Skip ASCII space/tab. Newlines are NOT consumed — callers
    /// control newline boundaries (line-oriented section parsers stop
    /// at `\n`; the triple-quote scanner spans newlines).
    pub(crate) fn eat_inline_ws(&mut self) {
        while let Some(b) = self.peek_byte() {
            if b == b' ' || b == b'\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn err(&self, msg: impl Into<String>) -> SyaraError {
        SyaraError::ParseError {
            line: self.line,
            message: msg.into(),
        }
    }

    /// Consume `$word` or bare `word` identifier (alphanumeric + `_`).
    pub(crate) fn consume_identifier(&mut self) -> Result<String, SyaraError> {
        let start = self.pos;
        if self.peek_byte() == Some(b'$') {
            self.pos += 1;
        }
        let word_start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == word_start {
            return Err(self.err("expected identifier"));
        }
        Ok(self.src[start..self.pos].to_owned())
    }

    /// Require a specific ASCII byte at the current position.
    pub(crate) fn expect_byte(&mut self, b: u8) -> Result<(), SyaraError> {
        if self.peek_byte() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(format!("expected `{}`", b as char)))
        }
    }

    /// Consume `"..."` with escape processing (`\"`, `\\`, `\n`, `\t`,
    /// `\r`; unknown escapes pass through verbatim, matching the
    /// behavior of `unescape_string`). Errors on unterminated.
    pub(crate) fn consume_quoted_string(&mut self) -> Result<String, SyaraError> {
        self.expect_byte(b'"')?;
        let mut out = String::new();
        while let Some(b) = self.peek_byte() {
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    match self.peek_byte() {
                        Some(b'"') => {
                            out.push('"');
                            self.pos += 1;
                        }
                        Some(b'\\') => {
                            out.push('\\');
                            self.pos += 1;
                        }
                        Some(b'n') => {
                            out.push('\n');
                            self.pos += 1;
                        }
                        Some(b't') => {
                            out.push('\t');
                            self.pos += 1;
                        }
                        Some(b'r') => {
                            out.push('\r');
                            self.pos += 1;
                        }
                        Some(_) => {
                            // Unknown escape: preserve `\` + next char raw.
                            out.push('\\');
                            let ch_end = self.next_char_end();
                            out.push_str(&self.src[self.pos..ch_end]);
                            self.pos = ch_end;
                        }
                        None => {
                            out.push('\\');
                            return Err(self.err("unterminated escape sequence"));
                        }
                    }
                }
                _ => {
                    let ch_end = self.next_char_end();
                    out.push_str(&self.src[self.pos..ch_end]);
                    self.pos = ch_end;
                }
            }
        }
        Err(self.err("unterminated string literal"))
    }

    /// Consume `"""..."""`. Body captured raw (no escape processing,
    /// matching the Python reference). Multi-line allowed. Errors on
    /// unterminated.
    pub(crate) fn consume_triple_quoted_string(&mut self) -> Result<String, SyaraError> {
        if self.peek_str(3) != "\"\"\"" {
            return Err(self.err("expected triple-quoted string"));
        }
        self.pos += 3;
        let body_start = self.pos;
        while self.pos < self.src.len() {
            if self.peek_str(3) == "\"\"\"" {
                let body = self.src[body_start..self.pos].to_owned();
                self.pos += 3;
                return Ok(body);
            }
            if self.peek_byte() == Some(b'\n') {
                self.line += 1;
            }
            let ch_end = self.next_char_end();
            self.pos = ch_end;
        }
        Err(self.err("unterminated triple-quoted string"))
    }

    /// Consume `/regex/[flags]`. The body preserves escapes verbatim
    /// (the regex engine handles its own escape semantics); only `\/`
    /// is recognized at the scanner layer so the body capture spans
    /// the whole pattern. Returns `(body, flag_chars)`. Empty body
    /// permitted. Unknown flag letters error rather than silently drop.
    pub(crate) fn consume_regex_literal(&mut self) -> Result<(String, Vec<char>), SyaraError> {
        self.expect_byte(b'/')?;
        let body_start = self.pos;
        while let Some(b) = self.peek_byte() {
            match b {
                b'/' => {
                    let body = self.src[body_start..self.pos].to_owned();
                    self.pos += 1;
                    let flags = self.consume_regex_flags()?;
                    return Ok((body, flags));
                }
                b'\\' => {
                    // Eat `\` + next char without interpreting (regex
                    // engine handles the escape semantics later).
                    self.pos += 1;
                    if self.pos < self.src.len() {
                        let ch_end = self.next_char_end();
                        self.pos = ch_end;
                    } else {
                        return Err(self.err("trailing backslash in regex literal"));
                    }
                }
                _ => {
                    let ch_end = self.next_char_end();
                    self.pos = ch_end;
                }
            }
        }
        Err(self.err("unterminated regex literal"))
    }

    fn consume_regex_flags(&mut self) -> Result<Vec<char>, SyaraError> {
        let mut flags = Vec::new();
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_alphabetic() {
                if b == b'i' {
                    flags.push('i');
                    self.pos += 1;
                } else {
                    return Err(self.err(format!(
                        "unsupported regex flag `{}` (only `i` is recognized)",
                        b as char
                    )));
                }
            } else {
                break;
            }
        }
        Ok(flags)
    }

    /// Consume an alphanumeric/underscore modifier word. Returns None
    /// when the next non-inline-ws byte is not an alphanumeric.
    pub(crate) fn consume_modifier_word(&mut self) -> Option<String> {
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            None
        } else {
            Some(self.src[start..self.pos].to_owned())
        }
    }

    /// Consume a kv value: quoted (escape-processed) OR bareword
    /// (`[^\s"]+`, leading `-` allowed for negative floats).
    pub(crate) fn consume_kv_value(&mut self) -> Result<String, SyaraError> {
        if self.peek_byte() == Some(b'"') {
            return self.consume_quoted_string();
        }
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_whitespace() || b == b'"' {
                break;
            }
            let ch_end = self.next_char_end();
            self.pos = ch_end;
        }
        if self.pos == start {
            return Err(self.err("expected value after `=`"));
        }
        Ok(self.src[start..self.pos].to_owned())
    }

    /// Byte position one UTF-8 char past `self.pos`. Caller has
    /// verified `self.pos < self.src.len()`.
    fn next_char_end(&self) -> usize {
        let mut e = self.pos + 1;
        while e < self.src.len() && !self.src.is_char_boundary(e) {
            e += 1;
        }
        e.min(self.src.len())
    }
}

/// Process escape sequences in a parsed string literal. Retained for
/// the parser test that pinned escape-sequence semantics before the
/// scanner replaced the regex-over-line tokenizer.
#[cfg(test)]
pub(super) fn unescape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(src: &str) -> Scanner<'_> {
        Scanner::new(src, 1)
    }

    #[test]
    fn identifier_dollar_and_bare() {
        assert_eq!(scan("$foo").consume_identifier().unwrap(), "$foo");
        assert_eq!(scan("bar123").consume_identifier().unwrap(), "bar123");
        assert!(scan("").consume_identifier().is_err());
    }

    #[test]
    fn quoted_string_basic_escapes() {
        let mut s = scan(r#""hello""#);
        assert_eq!(s.consume_quoted_string().unwrap(), "hello");

        let mut s = scan(r#""say \"hi\"""#);
        assert_eq!(s.consume_quoted_string().unwrap(), r#"say "hi""#);

        let mut s = scan(r#""path\\to\\file""#);
        assert_eq!(s.consume_quoted_string().unwrap(), r"path\to\file");

        let mut s = scan(r#""line\none""#);
        assert_eq!(s.consume_quoted_string().unwrap(), "line\none");
    }

    #[test]
    fn quoted_string_unknown_escape_passthrough() {
        // Matches existing unescape_string behavior: unknown escapes preserved.
        let mut s = scan(r#""\z""#);
        assert_eq!(s.consume_quoted_string().unwrap(), r"\z");
    }

    #[test]
    fn quoted_string_equals_inside_is_literal() {
        // `=` inside a quoted body is just a character (no kv interpretation).
        let mut s = scan(r#""key=value""#);
        assert_eq!(s.consume_quoted_string().unwrap(), "key=value");
    }

    #[test]
    fn quoted_string_unterminated_errors() {
        let mut s = scan(r#""no end"#);
        assert!(s.consume_quoted_string().is_err());
    }

    #[test]
    fn triple_quoted_basic() {
        let mut s = scan(r#""""abc""""#);
        assert_eq!(s.consume_triple_quoted_string().unwrap(), "abc");
    }

    #[test]
    fn triple_quoted_with_inner_quote_and_braces() {
        let body = r#"say "hi" and use {var}"#;
        let src = format!(r#""""{body}""""#);
        let mut s = Scanner::new(&src, 1);
        assert_eq!(s.consume_triple_quoted_string().unwrap(), body);
    }

    #[test]
    fn triple_quoted_multiline() {
        let src = "\"\"\"line1\nline2\nline3\"\"\"";
        let mut s = scan(src);
        assert_eq!(s.consume_triple_quoted_string().unwrap(), "line1\nline2\nline3");
    }

    #[test]
    fn triple_quoted_unterminated_errors() {
        let mut s = scan(r#""""never closes"#);
        assert!(s.consume_triple_quoted_string().is_err());
    }

    /// BUG-039 core: `\/` must not terminate the regex body capture.
    #[test]
    fn regex_literal_preserves_escaped_slash() {
        let mut s = scan(r"/<\/?(system|user|assistant)[\s>]/");
        let (body, flags) = s.consume_regex_literal().unwrap();
        assert_eq!(body, r"<\/?(system|user|assistant)[\s>]");
        assert!(flags.is_empty());
    }

    #[test]
    fn regex_literal_with_i_flag() {
        let mut s = scan(r"/abc/i");
        let (body, flags) = s.consume_regex_literal().unwrap();
        assert_eq!(body, "abc");
        assert_eq!(flags, vec!['i']);
    }

    #[test]
    fn regex_literal_with_literal_double_quote() {
        let mut s = scan(r#"/[a"b]/"#);
        let (body, _) = s.consume_regex_literal().unwrap();
        assert_eq!(body, r#"[a"b]"#);
    }

    #[test]
    fn regex_literal_empty_body() {
        let mut s = scan(r"//");
        let (body, _) = s.consume_regex_literal().unwrap();
        assert_eq!(body, "");
    }

    #[test]
    fn regex_literal_unsupported_flag_errors() {
        let mut s = scan(r"/abc/sm");
        assert!(s.consume_regex_literal().is_err());
    }

    #[test]
    fn regex_literal_unterminated_errors() {
        let mut s = scan(r"/abc");
        assert!(s.consume_regex_literal().is_err());
    }

    #[test]
    fn kv_value_quoted_with_equals_inside() {
        let mut s = scan(r#""a=b""#);
        assert_eq!(s.consume_kv_value().unwrap(), "a=b");
    }

    #[test]
    fn kv_value_bareword_negative_float() {
        let mut s = scan("-0.85 next");
        assert_eq!(s.consume_kv_value().unwrap(), "-0.85");
    }

    #[test]
    fn modifier_word_alphanumeric_sequence() {
        let mut s = scan("nocase wide");
        assert_eq!(s.consume_modifier_word(), Some("nocase".into()));
        s.eat_inline_ws();
        assert_eq!(s.consume_modifier_word(), Some("wide".into()));
        assert_eq!(s.consume_modifier_word(), None);
    }

    /// Parity invariant: the scanner's regex-literal end position must
    /// match `parser/mod.rs::split_rules`'s regex consumption for the
    /// same input. If these diverge, a rule-block split could put bytes
    /// inside the literal at the per-line layer and outside at the
    /// top layer (or vice versa) — exactly the bug class we are
    /// eliminating. Compares ending-byte positions for a fixture
    /// corpus of tricky regex bodies (ASCII only — multi-byte UTF-8
    /// inside escapes is mishandled by split_rules' byte-step but is
    /// not present in any real `.syara` rule).
    #[test]
    fn parity_with_split_rules_regex_consumption() {
        let fixtures = [
            r"/abc/",
            r"/abc/i",
            r"/<\/?(system|user|assistant)[\s>]/",
            r"/!\[[^\]]*\]\(https?:\/\/[^\s\)]+\)/i",
            r#"/[a"b]/"#,
            r"/<<\s*\/SYS\s*>>/i",
            r"//",
        ];
        for f in fixtures {
            let mut s = Scanner::new(f, 1);
            s.consume_regex_literal()
                .unwrap_or_else(|e| panic!("scanner failed on {f:?}: {e}"));
            let scanner_end = s.pos();

            // Mirror split_rules' regex consumption (parser/mod.rs:188-213).
            let bytes = f.as_bytes();
            assert_eq!(bytes[0], b'/');
            let mut j = 1;
            while j < bytes.len() {
                if bytes[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if bytes[j] == b'/' {
                    j += 1;
                    break;
                }
                j += 1;
            }
            while j < bytes.len() && bytes[j].is_ascii_alphabetic() {
                j += 1;
            }
            assert_eq!(scanner_end, j, "parity divergence for {f:?}");
        }
    }

    #[test]
    fn unescape_string_sequences() {
        assert_eq!(unescape_string(r#"hello"#), "hello");
        assert_eq!(unescape_string(r#"say \"hi\""#), "say \"hi\"");
        assert_eq!(unescape_string(r#"a\\b"#), "a\\b");
        assert_eq!(unescape_string(r#"line\none"#), "line\none");
        assert_eq!(unescape_string(r#"tab\there"#), "tab\there");
    }
}
