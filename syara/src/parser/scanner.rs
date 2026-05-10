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

    /// 1-indexed column of the current position **within the current
    /// line**. Derived from `pos` and the position of the most recent
    /// `\n` in `src` (byte-indexed; safe for ASCII inputs, which is
    /// what `.syara` rule files contain in practice).
    ///
    /// For per-line section parsers (where `src` is a single trimmed
    /// line containing no `\n`), this is `pos + 1`. For the LLM-stream
    /// parser (where `src` spans newlines), this resets to 1 after
    /// each `\n`.
    pub(crate) fn col(&self) -> usize {
        let bound = self.pos.min(self.src.len());
        let prefix = &self.src.as_bytes()[..bound];
        match prefix.iter().rposition(|&b| b == b'\n') {
            Some(nl_pos) => bound - nl_pos,
            None => bound + 1,
        }
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

    /// Skip ASCII inline whitespace: space, tab, and carriage return.
    /// Newlines (`\n`) are NOT consumed — callers control newline
    /// boundaries (line-oriented section parsers stop at `\n`; the
    /// triple-quote scanner spans newlines).
    ///
    /// `\r` is treated as inline whitespace so CRLF source files
    /// don't leave a stray `\r` between a token and its terminating
    /// `\n`. Without this, `collect_modifiers` /
    /// `collect_kv_params_until_newline` would still terminate (via
    /// `consume_modifier_word` returning `None` on non-alphanumeric),
    /// but only by accident — making the inline-whitespace contract
    /// uniform here is the cleaner fix.
    pub(crate) fn eat_inline_ws(&mut self) {
        while let Some(b) = self.peek_byte() {
            if b == b' ' || b == b'\t' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn err(&self, msg: impl Into<String>) -> SyaraError {
        SyaraError::ParseError {
            line: self.line,
            col: self.col(),
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
                            let ch_start = self.pos;
                            self.bump();
                            out.push_str(&self.src[ch_start..self.pos]);
                        }
                        None => {
                            out.push('\\');
                            return Err(self.err("unterminated escape sequence"));
                        }
                    }
                }
                _ => {
                    let ch_start = self.pos;
                    self.bump();
                    out.push_str(&self.src[ch_start..self.pos]);
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
            self.bump();
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
                    // engine handles the escape semantics later). The
                    // `\` itself is ASCII so a raw `pos += 1` is fine;
                    // the second char is routed through `bump()` so a
                    // literal `\n` after the backslash still advances
                    // `line`.
                    self.pos += 1;
                    if self.pos < self.src.len() {
                        self.bump();
                    } else {
                        return Err(self.err("trailing backslash in regex literal"));
                    }
                }
                _ => {
                    self.bump();
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

    /// `col()` for a per-line scanner (no `\n` in src) is `pos + 1`.
    #[test]
    fn col_per_line_scanner() {
        let mut s = scan("$foo = \"bar\"");
        assert_eq!(s.col(), 1);
        s.consume_identifier().unwrap(); // consumes `$foo` (4 bytes)
        assert_eq!(s.col(), 5);
        s.eat_inline_ws();
        assert_eq!(s.col(), 6); // on `=`
    }

    /// `col()` resets to 1 after each `\n` in a multi-line source
    /// (LLM-stream scanner case).
    #[test]
    fn col_resets_after_newline() {
        let src = "abc\ndef";
        let mut s = Scanner::new(src, 1);
        assert_eq!(s.col(), 1);
        s.bump(); // a
        s.bump(); // b
        s.bump(); // c → pos=3, on `\n`
        assert_eq!(s.col(), 4); // col before consuming the newline
        s.bump(); // \n → pos=4, line=2
        assert_eq!(s.line(), 2);
        assert_eq!(s.col(), 1); // first byte of new line
        s.bump(); // d
        assert_eq!(s.col(), 2);
    }

    /// Errors raised by Scanner methods carry the column where the
    /// failure happened, surfaced via the `Display` impl on `SyaraError`.
    #[test]
    fn err_includes_col_in_display() {
        // Identifier missing after `$`: scanner is at pos 0 when
        // `consume_identifier` fails on empty-after-$.
        let mut s = scan("   nope = \"x\"");
        s.eat_inline_ws();
        // Now pos = 3, col = 4. Force an error by expecting `=` here.
        let err = s.expect_byte(b'=').unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("col 4"),
            "expected col in error message, got: {msg}"
        );
    }

    /// `eat_inline_ws` consumes `\r` so CRLF source files don't leak
    /// a stray carriage return into downstream parsers.
    #[test]
    fn eat_inline_ws_consumes_carriage_return() {
        // Mixed inline whitespace including CR before a token.
        let mut s = scan(" \t\r\rfoo");
        s.eat_inline_ws();
        assert_eq!(s.peek_byte(), Some(b'f'));

        // A CR followed by an LF: CR is eaten, LF remains so callers
        // can detect end-of-line.
        let mut s = scan("nocase\r\n");
        // Start past the modifier so we're at `\r`.
        s.pos = "nocase".len();
        s.eat_inline_ws();
        assert_eq!(s.peek_byte(), Some(b'\n'));
    }

    /// Line tracking parity: every `consume_*` method that can advance
    /// past a `\n` (quoted string, regex literal, triple-quoted string)
    /// must increment `line`. Latent for the LLM-stream parser, which
    /// holds a single Scanner across newlines — without this, an error
    /// after a multi-line literal would report the wrong source line.
    #[test]
    fn quoted_string_tracks_line_with_embedded_newline() {
        let src = "\"line1\nline2\nline3\"";
        let mut s = Scanner::new(src, 1);
        assert_eq!(s.consume_quoted_string().unwrap(), "line1\nline2\nline3");
        assert_eq!(s.line(), 3);
    }

    #[test]
    fn quoted_string_tracks_line_through_unknown_escape_newline() {
        // `\` followed by a literal newline (unknown-escape passthrough).
        let src = "\"\\\nrest\"";
        let mut s = Scanner::new(src, 1);
        assert_eq!(s.consume_quoted_string().unwrap(), "\\\nrest");
        assert_eq!(s.line(), 2);
    }

    #[test]
    fn regex_literal_tracks_line_with_embedded_newline() {
        let src = "/line1\nline2/";
        let mut s = Scanner::new(src, 1);
        let (body, _) = s.consume_regex_literal().unwrap();
        assert_eq!(body, "line1\nline2");
        assert_eq!(s.line(), 2);
    }

    #[test]
    fn regex_literal_tracks_line_through_backslash_newline() {
        // `\` followed by literal `\n` in source: the second char of the
        // escape pair is a newline. The `\` itself is ASCII (no line
        // bump), but the next-char step must increment `line`.
        let src = "/foo\\\nbar/";
        let mut s = Scanner::new(src, 1);
        let (body, _) = s.consume_regex_literal().unwrap();
        assert_eq!(body, "foo\\\nbar");
        assert_eq!(s.line(), 2);
    }

    /// End-to-end: an unterminated literal that follows a multi-line
    /// quoted body must report the post-newline line number, not the
    /// line where the first body started.
    #[test]
    fn error_after_multiline_quoted_body_reports_correct_line() {
        // First a multi-line quoted body, then an unterminated regex.
        let src = "\"a\nb\nc\" /unclosed";
        let mut s = Scanner::new(src, 1);
        s.consume_quoted_string().unwrap();
        assert_eq!(s.line(), 3);
        s.eat_inline_ws();
        let err = s.consume_regex_literal().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("line 3"),
            "expected error to report line 3, got: {msg}"
        );
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

            // Mirror split_rules' regex consumption (parser/mod.rs:249-270).
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

    /// Parity invariant for quoted strings: the scanner's quoted-string
    /// end position must match `parser/mod.rs::split_rules`'s
    /// single-quote string-consumption arm (lines 210-221). If they
    /// diverge, a `}` inside a string body could be counted by the
    /// brace counter while the scanner correctly classifies it as
    /// literal content — a rule-split-corruption bug.
    #[test]
    fn parity_with_split_rules_string_consumption() {
        let fixtures = [
            r#""hello""#,
            r#""say \"hi\"""#,
            r#""path\\to\\file""#,
            r#""contains / slash""#,
            r#""contains } closing brace""#,
            r#""contains { opening brace""#,
            r#""contains // comment chars""#,
            r#""contains /* block */ chars""#,
            r#""""#, // empty body
        ];
        for f in fixtures {
            let mut s = Scanner::new(f, 1);
            s.consume_quoted_string()
                .unwrap_or_else(|e| panic!("scanner failed on {f:?}: {e}"));
            let scanner_end = s.pos();

            // Mirror split_rules' single-quote string consumption
            // (parser/mod.rs:236-248).
            let bytes = f.as_bytes();
            assert_eq!(bytes[0], b'"');
            let mut j = 1;
            while j < bytes.len() {
                if bytes[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if bytes[j] == b'"' {
                    j += 1;
                    break;
                }
                j += 1;
            }
            assert_eq!(scanner_end, j, "parity divergence for {f:?}");
        }
    }

    /// Parity invariant for triple-quoted strings: the scanner's
    /// triple-quote end position must match `parser/mod.rs::split_rules`'s
    /// triple-quote arm (lines 195-209). Embedded `{`, `}`, `"`,
    /// `//`, and `/*` inside a triple-quoted body must be classified
    /// the same way across layers.
    #[test]
    fn parity_with_split_rules_triple_quote_consumption() {
        let fixtures = [
            "\"\"\"abc\"\"\"",
            "\"\"\"\"\"\"", // empty body
            "\"\"\"contains } brace\"\"\"",
            "\"\"\"contains { brace\"\"\"",
            "\"\"\"contains // not a comment\"\"\"",
            "\"\"\"contains /* not a block */\"\"\"",
            "\"\"\"contains \"single\" quote\"\"\"",
            "\"\"\"multi\nline\nbody\"\"\"",
        ];
        for f in fixtures {
            let mut s = Scanner::new(f, 1);
            s.consume_triple_quoted_string()
                .unwrap_or_else(|e| panic!("scanner failed on {f:?}: {e}"));
            let scanner_end = s.pos();

            // Mirror split_rules' triple-quote consumption
            // (parser/mod.rs:218-235). Body is consumed byte-by-byte
            // until the next `"""` triple is found.
            let bytes = f.as_bytes();
            let n = bytes.len();
            assert!(
                n >= 3 && &bytes[0..3] == b"\"\"\"",
                "fixture must open with triple quote: {f:?}"
            );
            let mut j = 3;
            while j < n {
                if j + 2 < n
                    && bytes[j] == b'"'
                    && bytes[j + 1] == b'"'
                    && bytes[j + 2] == b'"'
                {
                    j += 3;
                    break;
                }
                j += 1;
            }
            assert_eq!(scanner_end, j, "parity divergence for {f:?}");
        }
    }

}
