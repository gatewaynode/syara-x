/// Parser for .syara rule files.
///
/// Uses a brace-counting approach (same as the Python implementation) that
/// correctly handles `{n,m}` regex quantifiers inside string literals and
/// regex literals without a full grammar.
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
        Regex,
    }
    let mut mode = Mode::Normal;

    while i < n {
        let c = chars[i];
        match mode {
            Mode::Normal => {
                if c == '"' {
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
    use sections::unescape_string;

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
}
