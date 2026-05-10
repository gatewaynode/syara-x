/// Parser for .syara rule files.
///
/// Uses a brace-counting approach (same as the Python implementation) that
/// correctly handles `{n,m}` regex quantifiers inside string literals and
/// regex literals without a full grammar.
mod scanner;
mod sections;

use std::path::Path;
use std::sync::LazyLock;
use regex::Regex;

use crate::error::SyaraError;
use crate::models::Rule;
use sections::*;

// ── Compiled regexes (BUG-004) ──────────────────────────────────────────────

static HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\brule\s+\w+(?:\s*:\s*[\w\s]+?)?\s*\{").unwrap()
});

static HEADER_CAPTURE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)rule\s+(\w+)(?:\s*:\s*([\w\s]+?))?\s*\{").unwrap()
});

// ── Public API ──────────────────────────────────────────────────────────────

pub struct SyaraParser;

impl SyaraParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a `.syara` file.
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<Rule>, SyaraError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SyaraError::FileNotFound(path.as_ref().display().to_string())
            } else {
                SyaraError::Io(e)
            }
        })?;
        self.parse_str(&content)
    }

    /// Parse rules from a string.
    pub fn parse_str(&self, content: &str) -> Result<Vec<Rule>, SyaraError> {
        let cleaned = remove_comments(content);
        let blocks = split_rules(&cleaned);
        blocks.iter().map(|b| parse_rule_block(b)).collect()
    }
}

impl Default for SyaraParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Comment removal ───────────────────────────────────────────────────────────

fn remove_comments(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;

    #[derive(PartialEq)]
    enum Mode {
        Normal,
        String,
        TripleString,
        Regex,
    }
    let mut mode = Mode::Normal;

    while i < n {
        let c = chars[i];
        match mode {
            Mode::Normal => {
                if c == '"' && i + 2 < n && chars[i + 1] == '"' && chars[i + 2] == '"' {
                    out.push_str("\"\"\"");
                    mode = Mode::TripleString;
                    i += 3;
                } else if c == '"' {
                    out.push(c);
                    mode = Mode::String;
                    i += 1;
                } else if c == '/' && i + 1 < n && chars[i + 1] == '/' {
                    i += 2;
                    while i < n && chars[i] != '\n' {
                        i += 1;
                    }
                } else if c == '/' && i + 1 < n && chars[i + 1] == '*' {
                    i += 2;
                    while i + 1 < n && !(chars[i] == '*' && chars[i + 1] == '/') {
                        i += 1;
                    }
                    i = (i + 2).min(n);
                } else if c == '/' {
                    let mut k = i as i64 - 1;
                    while k >= 0
                        && (chars[k as usize] == ' ' || chars[k as usize] == '\t')
                    {
                        k -= 1;
                    }
                    if k >= 0 && chars[k as usize] == '=' {
                        out.push(c);
                        mode = Mode::Regex;
                        i += 1;
                    } else {
                        out.push(c);
                        i += 1;
                    }
                } else {
                    out.push(c);
                    i += 1;
                }
            }
            Mode::String => {
                out.push(c);
                if c == '\\' && i + 1 < n {
                    out.push(chars[i + 1]);
                    i += 2;
                } else if c == '"' {
                    mode = Mode::Normal;
                    i += 1;
                } else {
                    i += 1;
                }
            }
            Mode::TripleString => {
                // Body captured raw (no escape processing — matches the
                // Python reference). Closes only on `"""`.
                if c == '"' && i + 2 < n && chars[i + 1] == '"' && chars[i + 2] == '"' {
                    out.push_str("\"\"\"");
                    mode = Mode::Normal;
                    i += 3;
                } else {
                    out.push(c);
                    i += 1;
                }
            }
            Mode::Regex => {
                out.push(c);
                if c == '\\' && i + 1 < n {
                    out.push(chars[i + 1]);
                    i += 2;
                } else if c == '/' {
                    i += 1;
                    while i < n && chars[i].is_alphabetic() {
                        out.push(chars[i]);
                        i += 1;
                    }
                    mode = Mode::Normal;
                } else {
                    i += 1;
                }
            }
        }
    }

    out
}

// ── Brace-counting rule splitter (BUG-005: byte-offset based) ───────────────

fn split_rules(content: &str) -> Vec<String> {
    let bytes = content.as_bytes();
    let n = bytes.len();
    let mut blocks = Vec::new();
    let mut offset = 0;

    while offset < n {
        let slice = &content[offset..];
        let m = match HEADER_RE.find(slice) {
            Some(m) => m,
            None => break,
        };

        let rule_start = offset + m.start();
        let mut j = offset + m.end();
        let mut depth: i32 = 1;

        while j < n && depth > 0 {
            match bytes[j] {
                b'\\' => {
                    j += 2;
                }
                b'"' => {
                    // Triple-quoted: consume `"""..."""` as a single
                    // token; inner single `"` and braces are content.
                    if j + 2 < n && bytes[j + 1] == b'"' && bytes[j + 2] == b'"' {
                        j += 3;
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
                        continue;
                    }
                    j += 1;
                    while j < n {
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
                }
                b'/' => {
                    let mut k = j;
                    while k > rule_start
                        && (bytes[k - 1] == b' ' || bytes[k - 1] == b'\t')
                    {
                        k -= 1;
                    }
                    if k > rule_start && bytes[k - 1] == b'=' {
                        j += 1;
                        while j < n {
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
                        while j < n && bytes[j].is_ascii_alphabetic() {
                            j += 1;
                        }
                    } else {
                        j += 1;
                    }
                }
                b'{' => {
                    depth += 1;
                    j += 1;
                }
                b'}' => {
                    depth -= 1;
                    j += 1;
                }
                _ => {
                    j += 1;
                }
            }
        }

        blocks.push(content[rule_start..j].to_string());
        offset = j;
    }

    blocks
}

// ── Rule block parser ─────────────────────────────────────────────────────────

fn parse_rule_block(block: &str) -> Result<Rule, SyaraError> {
    let hm = HEADER_CAPTURE_RE
        .captures(block)
        .ok_or_else(|| SyaraError::ParseError {
            line: 0,
            col: 0,
            message: format!("invalid rule header: {}", &block[..block.len().min(80)]),
        })?;

    let name = hm.get(1).unwrap().as_str().to_owned();
    let tags: Vec<String> = hm
        .get(2)
        .map(|m| {
            m.as_str()
                .split_whitespace()
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    let body_start = block.find('{').map(|p| p + 1).unwrap_or(0);
    let body_end = block.rfind('}').unwrap_or(block.len());
    let body = &block[body_start..body_end];

    let meta = parse_meta_section(body);
    let strings = parse_strings_section(body)?;
    let similarity = parse_similarity_section(body)?;
    let phash = parse_phash_section(body)?;
    let classifier = parse_classifier_section(body)?;
    let llm = parse_llm_section(body)?;
    let condition = parse_condition_section(body);

    // BUG-021: patterns without a condition is an error
    let has_patterns = !strings.is_empty()
        || !similarity.is_empty()
        || !phash.is_empty()
        || !classifier.is_empty()
        || !llm.is_empty();

    if has_patterns && condition.is_empty() {
        return Err(SyaraError::ParseError {
            line: 0,
            col: 0,
            message: format!(
                "rule '{}' has patterns but no condition section",
                name
            ),
        });
    }

    Ok(Rule {
        name,
        tags,
        meta,
        strings,
        similarity,
        phash,
        classifier,
        llm,
        condition,
        compiled_condition: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Modifier;
    use scanner::unescape_string;

    #[test]
    fn test_parse_basic_rule() {
        let src = r#"
        rule test_rule: tag1 tag2
        {
            meta:
                author = "tester"

            strings:
                $s1 = "test" nocase

            condition:
                $s1
        }
        "#;

        let parser = SyaraParser::new();
        let rules = parser.parse_str(src).unwrap();

        assert_eq!(rules.len(), 1);
        let rule = &rules[0];
        assert_eq!(rule.name, "test_rule");
        assert!(rule.tags.contains(&"tag1".to_owned()));
        assert!(rule.tags.contains(&"tag2".to_owned()));
        assert_eq!(rule.meta.get("author"), Some(&"tester".to_owned()));
        assert_eq!(rule.strings.len(), 1);
        assert_eq!(rule.strings[0].identifier, "$s1");
        assert!(rule.strings[0].modifiers.contains(&Modifier::NoCase));
    }

    #[test]
    fn test_parse_similarity() {
        let src = r#"
        rule test_sim {
            similarity:
                $s1 = "test pattern" threshold=0.85 matcher="sbert" cleaner="default_cleaning"
            condition:
                $s1
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        let sim = &rules[0].similarity[0];
        assert_eq!(sim.identifier, "$s1");
        assert!((sim.threshold - 0.85).abs() < 1e-9);
        assert_eq!(sim.matcher_name, "sbert");
    }

    #[test]
    fn test_parse_phash() {
        let src = r#"
        rule test_phash {
            phash:
                $p1 = "reference.png" threshold=0.95 hasher="imagehash"
            condition:
                $p1
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        let ph = &rules[0].phash[0];
        assert_eq!(ph.file_path, "reference.png");
        assert!((ph.threshold - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_parse_regex_with_quantifier() {
        let src = r#"
        rule test_regex {
            strings:
                $s2 = /\b(disregard|ignore)\s+(all\s+)?prior\b/i
            condition:
                $s2
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].strings[0].identifier, "$s2");
        assert!(rules[0].strings[0].is_regex);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let src = r#"
        rule rule_a { strings: $s1 = "foo" condition: $s1 }
        rule rule_b { strings: $s2 = "bar" condition: $s2 }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name, "rule_a");
        assert_eq!(rules[1].name, "rule_b");
    }

    #[test]
    fn test_comment_removal() {
        let src = r#"
        // single line comment
        rule test_comments {
            /* block comment */
            strings:
                $s1 = "hello" // inline comment
            condition:
                $s1
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].strings[0].pattern, "hello");
    }

    #[test]
    fn test_comment_stripper_respects_regex_literals() {
        let src = r#"
        rule first {
            strings:
                $s = /https?:\/\//i
            condition:
                any of them
        }

        rule second {
            strings:
                $s = "dummy"
            condition:
                any of them
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name, "first");
        assert_eq!(rules[1].name, "second");
    }

    #[test]
    fn test_comment_stripper_keeps_line_comments_outside_regex() {
        let src = r#"
        // header comment
        rule r { // trailing comment after header
            strings:
                $s = "x" // comment after string
            condition:
                $s
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].strings[0].pattern, "x");
    }

    // ── BUG-020: escaped quotes in string patterns ──────────────────────

    #[test]
    fn test_escaped_quotes_in_string_pattern() {
        let src = r#"
        rule escaped_quotes {
            strings:
                $s1 = "say \"hello\"" nocase
            condition:
                $s1
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].strings[0].pattern, "say \"hello\"");
        assert!(rules[0].strings[0].modifiers.contains(&Modifier::NoCase));
    }

    #[test]
    fn test_escaped_backslash_in_string_pattern() {
        let src = r#"
        rule escaped_backslash {
            strings:
                $s1 = "path\\to\\file"
            condition:
                $s1
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules[0].strings[0].pattern, "path\\to\\file");
    }

    // ── BUG-021: missing condition with patterns ────────────────────────

    #[test]
    fn test_missing_condition_with_patterns_is_error() {
        let src = r#"
        rule no_cond {
            strings:
                $s1 = "hello"
        }
        "#;

        let result = SyaraParser::new().parse_str(src);
        assert!(result.is_err(), "patterns without condition must error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no condition"), "error should explain: {msg}");
    }

    #[test]
    fn test_typo_condition_keyword_is_error() {
        let src = r#"
        rule typo_cond {
            strings:
                $s1 = "hello"
            conditon:
                $s1
        }
        "#;

        let result = SyaraParser::new().parse_str(src);
        assert!(result.is_err(), "misspelled 'conditon' should cause error");
    }

    #[test]
    fn test_rule_without_patterns_or_condition_ok() {
        let src = r#"
        rule meta_only {
            meta:
                author = "tester"
        }
        "#;

        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "meta_only");
    }

    // ── BUG-005: large rule file doesn't hang ───────────────────────────

    #[test]
    fn test_many_rules_parse_without_quadratic_slowdown() {
        let mut src = String::new();
        for i in 0..200 {
            src.push_str(&format!(
                "rule rule_{i} {{ strings: $s = \"pattern_{i}\" condition: $s }}\n"
            ));
        }
        let rules = SyaraParser::new().parse_str(&src).unwrap();
        assert_eq!(rules.len(), 200);
    }

    #[test]
    fn test_unescape_string_sequences() {
        assert_eq!(unescape_string(r#"hello"#), "hello");
        assert_eq!(unescape_string(r#"say \"hi\""#), "say \"hi\"");
        assert_eq!(unescape_string(r#"a\\b"#), "a\\b");
        assert_eq!(unescape_string(r#"line\none"#), "line\none");
        assert_eq!(unescape_string(r#"tab\there"#), "tab\there");
    }

    // ── Triple-quoted LLM patterns (Python parity) ─────────────────────

    #[test]
    fn test_triple_quoted_llm_pattern_with_braces_and_quotes() {
        let src = r#"
        rule llm_triple {
            llm:
                $p1 = """Detect intent: classify {input} as "harmful" or "benign"."""
            condition:
                $p1
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].llm.len(), 1);
        assert_eq!(
            rules[0].llm[0].pattern,
            r#"Detect intent: classify {input} as "harmful" or "benign"."#
        );
    }

    #[test]
    fn test_triple_quoted_llm_pattern_multiline() {
        let src = "
        rule llm_multi {
            llm:
                $p1 = \"\"\"first line
second line
third line\"\"\"
            condition:
                $p1
        }
        ";
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules[0].llm[0].pattern, "first line\nsecond line\nthird line");
    }

    /// Comment stripper must NOT interpret `//` inside a triple-quoted body.
    #[test]
    fn test_comment_stripper_preserves_triple_quote_body() {
        let src = r#"
        rule llm_with_comment_chars {
            llm:
                $p1 = """visit https://example.com or //inline"""
            condition:
                $p1
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(
            rules[0].llm[0].pattern,
            "visit https://example.com or //inline"
        );
    }

    // ── Latent escape bugs the BUG-039 fix repaired ────────────────────

    /// Pre-fix, `META_KV_RE = (\w+)\s*=\s*"([^"]*)"` truncated meta
    /// values at the first `"` byte regardless of escape. After the
    /// scanner refactor, `\"` is honored.
    #[test]
    fn test_meta_value_with_escaped_quote() {
        let src = r#"
        rule meta_escape {
            meta:
                description = "say \"hi\""
            strings:
                $s = "anything"
            condition:
                $s
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(
            rules[0].meta.get("description").map(String::as_str),
            Some(r#"say "hi""#)
        );
    }

    /// Pre-fix, `KV_PARAMS_RE`'s quoted-value branch `"([^"]*)"`
    /// truncated kv values at the first `"` byte. After the scanner
    /// refactor, `\"` is honored.
    #[test]
    fn test_kv_param_with_escaped_quote() {
        let src = r#"
        rule kv_escape {
            similarity:
                $s = "phrase" cleaner="say \"hi\"" threshold=0.5
            condition:
                $s
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules[0].similarity[0].cleaner_name, r#"say "hi""#);
    }

    /// BUG-039 diagnostic: what does the parser actually capture for a
    /// regex literal containing an escaped forward slash?
    #[test]
    fn test_bug039_regex_with_escaped_slash_capture() {
        let src = r#"
        rule r {
            strings:
                $r = /<\/?(system|user|assistant)[\s>]/
            condition:
                $r
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        let pat = &rules[0].strings[0].pattern;
        eprintln!("BUG-039 captured pattern = {pat:?}");
        // Pin the *expected* shape: the parser must preserve the full regex body.
        assert_eq!(
            pat,
            r"<\/?(system|user|assistant)[\s>]",
            "parser truncated the regex body"
        );
    }

    // ── Cross-layer parity: remove_comments must not strip comment-like ──
    // ── byte sequences that appear inside literal bodies. Mirrors the   ──
    // ── scanner ↔ split_rules parity tests (scanner.rs) for the third   ──
    // ── state machine (`Mode::String` / `Mode::Regex` / `Mode::TripleString`).
    //
    // Each test exercises remove_comments (directly) AND parse_str
    // (end-to-end) so a regression at either layer is caught.

    /// `Mode::String` parity: `//` and `/* */` inside a quoted-string
    /// body are content, not comment openers.
    #[test]
    fn test_string_body_preserves_comment_chars() {
        let src = r#"
        rule r {
            strings:
                $a = "see // not a comment and /* not a block */"
            condition:
                $a
        }
        "#;
        let cleaned = remove_comments(src);
        assert!(
            cleaned.contains(r#""see // not a comment and /* not a block */""#),
            "remove_comments stripped comment-chars from inside string body: {cleaned:?}"
        );
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(
            rules[0].strings[0].pattern,
            "see // not a comment and /* not a block */"
        );
    }

    /// `Mode::Regex` parity: `//` and `/* */` inside a regex body are
    /// content, not comment openers.
    #[test]
    fn test_regex_body_preserves_comment_chars() {
        let src = r#"
        rule r {
            strings:
                $a = /https:\/\/[^\s]+\/\*nope\*\//i
            condition:
                $a
        }
        "#;
        let cleaned = remove_comments(src);
        assert!(
            cleaned.contains(r"/https:\/\/[^\s]+\/\*nope\*\//i"),
            "remove_comments stripped comment-chars from inside regex body: {cleaned:?}"
        );
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(
            rules[0].strings[0].pattern,
            r"https:\/\/[^\s]+\/\*nope\*\/"
        );
    }

    /// `Mode::TripleString` parity for `/* */` (extends the existing
    /// `test_comment_stripper_preserves_triple_quote_body` which covers
    /// `//`).
    #[test]
    fn test_triple_quote_body_preserves_block_comment_chars() {
        let src = r#"
        rule llm_block {
            llm:
                $p1 = """body /* with */ block-comment chars"""
            condition:
                $p1
        }
        "#;
        let cleaned = remove_comments(src);
        assert!(
            cleaned.contains(r#""""body /* with */ block-comment chars""""#),
            "remove_comments stripped /*…*/ from inside triple-quoted body: {cleaned:?}"
        );
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(
            rules[0].llm[0].pattern,
            "body /* with */ block-comment chars"
        );
    }

    // ── Cross-layer parity: split_rules brace counter must NOT count   ──
    // ── `{` / `}` that appear inside a literal body. Companion to the  ──
    // ── existing `test_parse_regex_with_quantifier` (which covers the  ──
    // ── regex case via `{1,3}`).

    /// `split_rules` single-quote arm parity: `}` inside a string body
    /// must not decrement the brace counter.
    #[test]
    fn test_split_rules_treats_brace_inside_string_as_literal() {
        let src = r#"
        rule r {
            strings:
                $a = "contains } closing brace and { opening brace"
            condition:
                $a
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(
            rules[0].strings[0].pattern,
            "contains } closing brace and { opening brace"
        );
    }

    /// `split_rules` triple-quote arm parity: `{` and `}` inside a
    /// triple-quoted body must not be counted.
    #[test]
    fn test_split_rules_treats_braces_inside_triple_quote_as_literal() {
        let src = r#"
        rule r {
            llm:
                $a = """body with { and } braces inside"""
            condition:
                $a
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(
            rules[0].llm[0].pattern,
            "body with { and } braces inside"
        );
    }

    /// Parse errors include a 1-indexed column number in the Display
    /// output. Pre-fix, errors reported only line, leaving consumers
    /// to grep through the line themselves to find the offending byte.
    #[test]
    fn test_parse_error_includes_col() {
        // Unterminated quoted string: scanner errors at the position
        // it was looking for the closing `"`. The trimmed line is
        // `$s = "no end`, so the error fires at pos 12 (col 13) which
        // is one past the last byte.
        let src = r#"
        rule bad {
            strings:
                $s = "no end
            condition:
                $s
        }
        "#;
        let err = SyaraParser::new().parse_str(src).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("col "),
            "parse error must include `col N`, got: {msg}"
        );
        assert!(
            msg.contains("unterminated"),
            "expected unterminated-string error, got: {msg}"
        );
    }

    /// Parse errors WITHOUT useful column info (rule-header level
    /// failures that fire after section parsing) omit the col suffix
    /// rather than emitting `col 0`.
    #[test]
    fn test_parse_error_omits_col_when_unknown() {
        // Patterns without condition: error fires in `parse_rule_block`
        // with line=0, col=0 sentinel. Display must not show "col 0".
        let src = r#"
        rule no_cond {
            strings:
                $s1 = "hello"
        }
        "#;
        let err = SyaraParser::new().parse_str(src).unwrap_err();
        let msg = err.to_string();
        assert!(
            !msg.contains("col 0"),
            "col=0 sentinel must not appear in Display output, got: {msg}"
        );
        assert!(msg.contains("no condition"), "expected the right error: {msg}");
    }

    // ── CRLF line-ending support ────────────────────────────────────────
    //
    // `.syara` files authored on Windows use `\r\n` line terminators.
    // `eat_inline_ws` now consumes `\r` so the inline-whitespace
    // contract is uniform across line endings.

    /// Per-line section parsers (meta / strings / similarity / phash /
    /// classifier) handle CRLF line endings transparently. `lines()`
    /// already strips `\r\n` and `\n` uniformly, so the per-line
    /// Scanner sees no `\r`. This test pins the end-to-end behavior.
    #[test]
    fn test_parse_crlf_per_line_sections() {
        let src = "rule crlf_test {\r\n    meta:\r\n        author = \"win\"\r\n    strings:\r\n        $s1 = \"hello\" nocase\r\n    condition:\r\n        $s1\r\n}\r\n";
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "crlf_test");
        assert_eq!(rules[0].meta.get("author"), Some(&"win".to_owned()));
        assert_eq!(rules[0].strings[0].pattern, "hello");
        assert!(rules[0].strings[0].modifiers.contains(&Modifier::NoCase));
    }

    /// Modifier list with CRLF terminator: `nocase\r\n`. Pre-fix,
    /// `eat_inline_ws` left `\r` in place; `collect_modifiers`
    /// terminated only because `consume_modifier_word` happened to
    /// return `None` on the `\r` byte. Fix routes `\r` through
    /// `eat_inline_ws` so the contract is explicit.
    #[test]
    fn test_parse_crlf_modifiers_and_kv_params() {
        let src = "rule mods_crlf {\r\n    strings:\r\n        $s = \"x\" nocase wide\r\n    similarity:\r\n        $sim = \"phrase\" threshold=0.5 matcher=\"sbert\"\r\n    condition:\r\n        $s or $sim\r\n}\r\n";
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 1);
        assert!(rules[0].strings[0].modifiers.contains(&Modifier::NoCase));
        assert!(rules[0].strings[0].modifiers.contains(&Modifier::Wide));
        let sim = &rules[0].similarity[0];
        assert!((sim.threshold - 0.5).abs() < 1e-9);
        assert_eq!(sim.matcher_name, "sbert");
    }

    /// LLM section parses as a single Scanner stream spanning newlines.
    /// CRLF line endings between rules within the section must be
    /// consumed cleanly by `skip_ws_and_newlines` (which uses
    /// `is_ascii_whitespace`, including `\r`) and the kv-param loop
    /// (which now exits cleanly via the fixed `eat_inline_ws`).
    #[test]
    fn test_parse_crlf_llm_stream_section() {
        let src = "rule llm_crlf {\r\n    llm:\r\n        $p1 = \"first prompt\" llm=\"openai-api-compatible\"\r\n        $p2 = \"second prompt\"\r\n    condition:\r\n        $p1 or $p2\r\n}\r\n";
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules[0].llm.len(), 2);
        assert_eq!(rules[0].llm[0].pattern, "first prompt");
        assert_eq!(rules[0].llm[0].llm_name, "openai-api-compatible");
        assert_eq!(rules[0].llm[1].pattern, "second prompt");
    }

    /// Triple-quoted bodies in the LLM section preserve CRLF inside
    /// the body verbatim (matches the "raw body" Python-parity
    /// contract for triple-quoted strings — we don't normalize line
    /// endings in user-authored prompt templates).
    #[test]
    fn test_parse_crlf_inside_triple_quoted_body_is_preserved() {
        let src = "rule llm_triple_crlf {\r\n    llm:\r\n        $p1 = \"\"\"line1\r\nline2\r\nline3\"\"\"\r\n    condition:\r\n        $p1\r\n}\r\n";
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules[0].llm[0].pattern, "line1\r\nline2\r\nline3");
    }

    /// `split_rules` two-rule split with brace inside literal: the
    /// inner `}` must not split the first rule prematurely. If
    /// split_rules' string arm disagreed with the brace counter on
    /// what's "inside" a literal, this would produce one giant
    /// malformed block instead of two clean rules.
    #[test]
    fn test_split_rules_two_rules_with_brace_in_string_body() {
        let src = r#"
        rule first {
            strings:
                $a = "has } brace"
            condition:
                $a
        }
        rule second {
            strings:
                $b = "no brace"
            condition:
                $b
        }
        "#;
        let rules = SyaraParser::new().parse_str(src).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name, "first");
        assert_eq!(rules[1].name, "second");
        assert_eq!(rules[0].strings[0].pattern, "has } brace");
    }
}
