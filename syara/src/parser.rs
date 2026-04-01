/// Parser for .syara rule files.
///
/// Uses a brace-counting approach (same as the Python implementation) that
/// correctly handles `{n,m}` regex quantifiers inside string literals and
/// regex literals without a full grammar.
use std::collections::HashMap;
use std::path::Path;
use regex::Regex;

use crate::error::SyaraError;
use crate::models::{
    ClassifierRule, LLMRule, Modifier, PHashRule, Rule, SimilarityRule, StringRule,
};

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
    // Remove /* ... */ block comments
    let block_re = Regex::new(r"/\*[\s\S]*?\*/").unwrap();
    let s = block_re.replace_all(input, "");

    // Remove // line comments
    let line_re = Regex::new(r"//[^\n]*").unwrap();
    line_re.replace_all(&s, "").into_owned()
}

// ── Brace-counting rule splitter ─────────────────────────────────────────────

fn split_rules(content: &str) -> Vec<String> {
    let chars: Vec<char> = content.chars().collect();
    let n = chars.len();
    let mut blocks = Vec::new();
    let mut i = 0;

    // Regex to locate the start of a rule header
    let header_re =
        Regex::new(r"\brule\s+\w+(?:\s*:\s*[\w\s]+?)?\s*\{").unwrap();

    while i < n {
        let slice: String = chars[i..].iter().collect();
        let m = match header_re.find(&slice) {
            Some(m) => m,
            None => break,
        };

        let rule_start = i + m.start();
        // j points to the character after the opening '{'
        let mut j = i + m.end();
        let mut depth: i32 = 1;

        while j < n && depth > 0 {
            match chars[j] {
                '\\' => {
                    j += 2; // skip escaped character
                }
                '"' => {
                    // String literal
                    j += 1;
                    while j < n {
                        if chars[j] == '\\' {
                            j += 2;
                            continue;
                        }
                        if chars[j] == '"' {
                            j += 1;
                            break;
                        }
                        j += 1;
                    }
                }
                '/' => {
                    // Regex literal only when immediately preceded by '=' (optional whitespace)
                    let mut k = j as i64 - 1;
                    while k >= rule_start as i64
                        && (chars[k as usize] == ' ' || chars[k as usize] == '\t')
                    {
                        k -= 1;
                    }
                    if k >= rule_start as i64 && chars[k as usize] == '=' {
                        j += 1;
                        while j < n {
                            if chars[j] == '\\' {
                                j += 2;
                                continue;
                            }
                            if chars[j] == '/' {
                                j += 1;
                                break;
                            }
                            j += 1;
                        }
                        // skip optional flags
                        while j < n && chars[j].is_alphabetic() {
                            j += 1;
                        }
                    } else {
                        j += 1;
                    }
                }
                '{' => {
                    depth += 1;
                    j += 1;
                }
                '}' => {
                    depth -= 1;
                    j += 1;
                }
                _ => {
                    j += 1;
                }
            }
        }

        blocks.push(chars[rule_start..j].iter().collect());
        i = j;
    }

    blocks
}

// ── Rule block parser ─────────────────────────────────────────────────────────

fn parse_rule_block(block: &str) -> Result<Rule, SyaraError> {
    // Rule header: rule <name> [: tags] {
    let header_re = Regex::new(r"(?s)rule\s+(\w+)(?:\s*:\s*([\w\s]+?))?\s*\{").unwrap();
    let hm = header_re
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

    // Extract body between first '{' and last '}'
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
    })
}

// ── Section parsers ───────────────────────────────────────────────────────────

fn section_content<'a>(body: &'a str, section: &str, next_sections: &[&str]) -> Option<&'a str> {
    let pattern = format!(r"(?i){}:", regex::escape(section));
    let re = Regex::new(&pattern).ok()?;
    let m = re.find(body)?;
    let start = m.end();

    // Find where the next section begins
    let mut end = body.len();
    for &ns in next_sections {
        let np = format!(r"(?i){}:", regex::escape(ns));
        if let Ok(nre) = Regex::new(&np) {
            if let Some(nm) = nre.find(&body[start..]) {
                end = end.min(start + nm.start());
            }
        }
    }

    Some(&body[start..end])
}

fn parse_meta_section(body: &str) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    let content = match section_content(
        body,
        "meta",
        &["strings", "similarity", "phash", "classifier", "llm", "condition"],
    ) {
        Some(c) => c,
        None => return meta,
    };

    let kv_re = Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).unwrap();
    for cap in kv_re.captures_iter(content) {
        meta.insert(cap[1].to_owned(), cap[2].to_owned());
    }
    meta
}

fn parse_strings_section(body: &str) -> Result<Vec<StringRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(
        body,
        "strings",
        &["similarity", "phash", "classifier", "llm", "condition"],
    ) {
        Some(c) => c,
        None => return Ok(rules),
    };

    let str_re = Regex::new(r#"(\$\w+)\s*=\s*"([^"]*)"\s*(.*)"#).unwrap();
    let re_re = Regex::new(r"(\$\w+)\s*=\s*/([^/]*)/(i?)\s*(.*)").unwrap();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(cap) = str_re.captures(line) {
            let identifier = cap[1].to_owned();
            let pattern = cap[2].to_owned();
            let mods = parse_string_modifiers(&cap[3]);
            rules.push(StringRule {
                identifier,
                pattern,
                modifiers: mods,
                is_regex: false,
            });
            continue;
        }

        if let Some(cap) = re_re.captures(line) {
            let identifier = cap[1].to_owned();
            let pattern = cap[2].to_owned();
            let mut mods = parse_string_modifiers(&cap[4]);
            if &cap[3] == "i" && !mods.contains(&Modifier::NoCase) {
                mods.push(Modifier::NoCase);
            }
            rules.push(StringRule {
                identifier,
                pattern,
                modifiers: mods,
                is_regex: true,
            });
        }
    }

    Ok(rules)
}

fn parse_string_modifiers(s: &str) -> Vec<Modifier> {
    s.split_whitespace()
        .filter_map(|tok| {
            // strip trailing punctuation
            let tok = tok.trim_end_matches(|c: char| !c.is_alphanumeric());
            Modifier::from_str(tok)
        })
        .collect()
}

fn parse_similarity_section(body: &str) -> Result<Vec<SimilarityRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(
        body,
        "similarity",
        &["phash", "classifier", "llm", "condition"],
    ) {
        Some(c) => c,
        None => return Ok(rules),
    };

    let line_re = Regex::new(r#"(\$\w+)\s*=\s*"([^"]+)"\s*(.*)"#).unwrap();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match line_re.captures(line) {
            Some(c) => c,
            None => continue,
        };

        let identifier = cap[1].to_owned();
        let pattern = cap[2].to_owned();
        let params = parse_kv_params(&cap[3]);

        let threshold: f64 = params
            .get("threshold")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.8);

        rules.push(SimilarityRule {
            identifier,
            pattern,
            threshold,
            cleaner_name: params
                .get("cleaner")
                .cloned()
                .unwrap_or_else(|| "default_cleaning".into()),
            chunker_name: params
                .get("chunker")
                .cloned()
                .unwrap_or_else(|| "no_chunking".into()),
            matcher_name: params
                .get("matcher")
                .cloned()
                .unwrap_or_else(|| "sbert".into()),
        });
    }

    Ok(rules)
}

fn parse_phash_section(body: &str) -> Result<Vec<PHashRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "phash", &["classifier", "llm", "condition"]) {
        Some(c) => c,
        None => return Ok(rules),
    };

    let line_re = Regex::new(r#"(\$\w+)\s*=\s*"([^"]+)"\s*(.*)"#).unwrap();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match line_re.captures(line) {
            Some(c) => c,
            None => continue,
        };

        let identifier = cap[1].to_owned();
        let file_path = cap[2].to_owned();
        let params = parse_kv_params(&cap[3]);

        let threshold: f64 = params
            .get("threshold")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.9);

        rules.push(PHashRule {
            identifier,
            file_path,
            threshold,
            phash_name: params
                .get("hasher")
                .cloned()
                .unwrap_or_else(|| "imagehash".into()),
        });
    }

    Ok(rules)
}

fn parse_classifier_section(body: &str) -> Result<Vec<ClassifierRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "classifier", &["llm", "condition"]) {
        Some(c) => c,
        None => return Ok(rules),
    };

    let line_re = Regex::new(r#"(\$\w+)\s*=\s*"([^"]+)"\s*(.*)"#).unwrap();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match line_re.captures(line) {
            Some(c) => c,
            None => continue,
        };

        let identifier = cap[1].to_owned();
        let pattern = cap[2].to_owned();
        let params = parse_kv_params(&cap[3]);

        let threshold: f64 = params
            .get("threshold")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.7);

        rules.push(ClassifierRule {
            identifier,
            pattern,
            threshold,
            cleaner_name: params
                .get("cleaner")
                .cloned()
                .unwrap_or_else(|| "default_cleaning".into()),
            chunker_name: params
                .get("chunker")
                .cloned()
                .unwrap_or_else(|| "no_chunking".into()),
            classifier_name: params
                .get("classifier")
                .cloned()
                .unwrap_or_else(|| "tuned-sbert".into()),
        });
    }

    Ok(rules)
}

fn parse_llm_section(body: &str) -> Result<Vec<LLMRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "llm", &["condition"]) {
        Some(c) => c,
        None => return Ok(rules),
    };

    let line_re = Regex::new(r#"(\$\w+)\s*=\s*"([^"]+)"\s*(.*)"#).unwrap();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match line_re.captures(line) {
            Some(c) => c,
            None => continue,
        };

        let identifier = cap[1].to_owned();
        let pattern = cap[2].to_owned();
        let params = parse_kv_params(&cap[3]);

        rules.push(LLMRule {
            identifier,
            pattern,
            llm_name: params
                .get("llm")
                .cloned()
                .unwrap_or_else(|| "flan-t5-large".into()),
            cleaner_name: params
                .get("cleaner")
                .cloned()
                .unwrap_or_else(|| "no_op".into()),
            chunker_name: params
                .get("chunker")
                .cloned()
                .unwrap_or_else(|| "no_chunking".into()),
        });
    }

    Ok(rules)
}

fn parse_condition_section(body: &str) -> String {
    let content = match section_content(body, "condition", &[]) {
        Some(c) => c,
        None => return String::new(),
    };
    content.trim().to_owned()
}

// ── Key-value parameter parser ────────────────────────────────────────────────

/// Parse `key=value` and `key="value"` pairs from a parameter string.
fn parse_kv_params(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    // Matches: key="value" or key=value
    let re = Regex::new(r#"(\w+)=(?:"([^"]*)"|([^\s"]+))"#).unwrap();
    for cap in re.captures_iter(s) {
        let key = cap[1].to_owned();
        let val = cap
            .get(2)
            .map(|m| m.as_str())
            .or_else(|| cap.get(3).map(|m| m.as_str()))
            .unwrap_or("")
            .to_owned();
        map.insert(key, val);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Brace-counting must not be fooled by {n,m} inside regex
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
}
