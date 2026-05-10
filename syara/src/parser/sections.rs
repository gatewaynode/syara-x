/// Section parsers for individual rule body sections (meta, strings, etc.).
///
/// Each per-line parse goes through `super::scanner::Scanner` rather than
/// a regex-over-line capture. The scanner correctly handles escape
/// sequences in delimited bodies (`\/`, `\"`, `\\`) — the bug class that
/// the prior `LazyLock<Regex>` constants exhibited (see BUG-039 in
/// `tasks/05-10-2026_BUGS.md`).
use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::error::SyaraError;
use crate::models::{
    ClassifierRule, LLMRule, Modifier, PHashRule, SimilarityRule, StringRule,
};

use super::scanner::Scanner;

// ── Section ordering ────────────────────────────────────────────────────────

/// Canonical section order within a rule block. `section_content` uses
/// the suffix of this list (everything after the requested section) as
/// the set of "next sections" that terminate the body slice. Keeping
/// the order in one place avoids the prior 6-site duplication of the
/// trailing-section list across the per-section parsers.
pub(crate) const SECTION_ORDER: &[&str] = &[
    "meta",
    "strings",
    "similarity",
    "phash",
    "classifier",
    "llm",
    "condition",
];

fn next_sections(current: &str) -> &'static [&'static str] {
    let idx = SECTION_ORDER
        .iter()
        .position(|&s| s == current)
        .expect("section name must appear in SECTION_ORDER");
    &SECTION_ORDER[idx + 1..]
}

// ── Section content extractor ───────────────────────────────────────────────

/// Locate the body of a named section (`strings:`, `meta:`, etc.) within
/// a rule block. The keyword set is fixed and small, so the dynamic
/// regex compile is cached behind a `LazyLock<Mutex<HashMap>>`.
pub(crate) fn section_content<'a>(body: &'a str, section: &str) -> Option<&'a str> {
    let header_re = section_header_regex(section)?;
    let m = header_re.find(body)?;
    let start = m.end();

    let mut end = body.len();
    for &ns in next_sections(section) {
        if let Some(nre) = section_header_regex(ns) {
            if let Some(nm) = nre.find(&body[start..]) {
                end = end.min(start + nm.start());
            }
        }
    }
    Some(&body[start..end])
}

fn section_header_regex(section: &str) -> Option<Regex> {
    static SECTION_HEADER_CACHE: LazyLock<std::sync::Mutex<HashMap<String, Regex>>> =
        LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

    let pattern = format!(r"(?i){}:", regex::escape(section));
    let mut cache = SECTION_HEADER_CACHE.lock().ok()?;
    if let Some(re) = cache.get(&pattern) {
        return Some(re.clone());
    }
    let re = Regex::new(&pattern).ok()?;
    cache.insert(pattern, re.clone());
    Some(re)
}

// ── Per-section line parsers ────────────────────────────────────────────────

pub(crate) fn parse_meta_section(body: &str) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    let content = match section_content(body, "meta") {
        Some(c) => c,
        None => return meta,
    };

    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut s = Scanner::new(line, idx + 1);
        let key = match s.consume_identifier() {
            Ok(k) => k,
            Err(_) => continue,
        };
        s.eat_inline_ws();
        if s.expect_byte(b'=').is_err() {
            continue;
        }
        s.eat_inline_ws();
        if let Ok(value) = s.consume_quoted_string() {
            meta.insert(key, value);
        }
    }
    meta
}

pub(crate) fn parse_strings_section(body: &str) -> Result<Vec<StringRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "strings") {
        Some(c) => c,
        None => return Ok(rules),
    };

    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut s = Scanner::new(line, idx + 1);
        let identifier = match s.consume_identifier() {
            Ok(id) if id.starts_with('$') => id,
            _ => continue,
        };
        s.eat_inline_ws();
        if s.expect_byte(b'=').is_err() {
            continue;
        }
        s.eat_inline_ws();

        let (pattern, is_regex, mut implicit_mods) = match s.peek_byte() {
            Some(b'"') => (s.consume_quoted_string()?, false, Vec::new()),
            Some(b'/') => {
                let (regex_body, flags) = s.consume_regex_literal()?;
                let mods = if flags.contains(&'i') {
                    vec![Modifier::NoCase]
                } else {
                    Vec::new()
                };
                (regex_body, true, mods)
            }
            _ => {
                return Err(SyaraError::ParseError {
                    line: idx + 1,
                    col: s.col(),
                    message: format!(
                        "expected `\"` or `/` after `=` in string rule, got: {}",
                        line
                    ),
                });
            }
        };

        for m in collect_modifiers(&mut s) {
            if !implicit_mods.contains(&m) {
                implicit_mods.push(m);
            }
        }

        rules.push(StringRule {
            identifier,
            pattern,
            modifiers: implicit_mods,
            is_regex,
        });
    }
    Ok(rules)
}

pub(crate) fn parse_similarity_section(body: &str) -> Result<Vec<SimilarityRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "similarity") {
        Some(c) => c,
        None => return Ok(rules),
    };

    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (identifier, pattern, params) = parse_quoted_section_line(line, idx + 1)?;
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

pub(crate) fn parse_phash_section(body: &str) -> Result<Vec<PHashRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "phash") {
        Some(c) => c,
        None => return Ok(rules),
    };

    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (identifier, file_path, params) = parse_quoted_section_line(line, idx + 1)?;
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

pub(crate) fn parse_classifier_section(body: &str) -> Result<Vec<ClassifierRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "classifier") {
        Some(c) => c,
        None => return Ok(rules),
    };

    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (identifier, pattern, params) = parse_quoted_section_line(line, idx + 1)?;
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

/// LLM section is the only place that accepts triple-quoted patterns
/// (matching the Python reference). Triple-quoted bodies may span
/// multiple lines, so the parser scans the entire section as a single
/// stream rather than line-by-line.
pub(crate) fn parse_llm_section(body: &str) -> Result<Vec<LLMRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "llm") {
        Some(c) => c,
        None => return Ok(rules),
    };

    let mut s = Scanner::new(content, 1);
    loop {
        skip_ws_and_newlines(&mut s);
        if s.at_end() {
            break;
        }
        if s.peek_byte() != Some(b'$') {
            // Strict: the comment stripper has already removed `//`
            // and `/* */` comments, so any non-blank, non-`$id`
            // content here is a typo (e.g. `pp1 = ...` missing the
            // `$`) or stray text. Silently skipping silently loses
            // the user's rule — refuse instead.
            return Err(SyaraError::ParseError {
                line: s.line(),
                col: s.col(),
                message: "expected `$` to start an llm rule identifier".into(),
            });
        }

        let line_at_start = s.line();
        let identifier = s.consume_identifier()?;
        s.eat_inline_ws();
        s.expect_byte(b'=')?;
        s.eat_inline_ws();

        let pattern = match s.peek_byte() {
            Some(b'"') if s.peek_str(3) == "\"\"\"" => s.consume_triple_quoted_string()?,
            Some(b'"') => s.consume_quoted_string()?,
            _ => {
                return Err(SyaraError::ParseError {
                    line: line_at_start,
                    col: s.col(),
                    message: "expected quoted pattern after `=` in llm rule".into(),
                });
            }
        };

        let params = collect_kv_params_until_newline(&mut s);

        rules.push(LLMRule {
            identifier,
            pattern,
            llm_name: params
                .get("llm")
                .cloned()
                .unwrap_or_else(|| "openai-api-compatible".into()),
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

pub(crate) fn parse_condition_section(body: &str) -> String {
    let content = match section_content(body, "condition") {
        Some(c) => c,
        None => return String::new(),
    };
    content.trim().to_owned()
}

// ── Shared helpers ──────────────────────────────────────────────────────────

/// `(identifier, pattern, kv params)` triple for a parsed
/// `$id = "..." key=value ...` line.
type SectionLineParts = (String, String, HashMap<String, String>);

/// Parse one of the standard `$id = "pattern" key=value ...` section
/// lines (similarity, phash, classifier — all of which require a
/// quoted string body, not a regex literal). Returns a `ParseError`
/// for any non-blank line that doesn't match the expected shape;
/// callers strip blank lines before calling. Strict by design — the
/// prior tolerant `Ok(None)` would silently swallow typos like
/// `foo = "..."` (missing `$`) and lose the user's rule.
fn parse_quoted_section_line(
    line: &str,
    line_no: usize,
) -> Result<SectionLineParts, SyaraError> {
    let mut s = Scanner::new(line, line_no);
    let identifier = s.consume_identifier()?;
    if !identifier.starts_with('$') {
        return Err(SyaraError::ParseError {
            line: line_no,
            col: 1,
            message: format!(
                "expected `$` to start rule identifier, got `{identifier}`"
            ),
        });
    }
    s.eat_inline_ws();
    s.expect_byte(b'=')?;
    s.eat_inline_ws();
    if s.peek_byte() != Some(b'"') {
        return Err(SyaraError::ParseError {
            line: line_no,
            col: s.col(),
            message: "expected quoted pattern after `=`".into(),
        });
    }
    let pattern = s.consume_quoted_string()?;
    let params = collect_kv_params_until_newline(&mut s);
    Ok((identifier, pattern, params))
}

/// Read modifier words (alphanumeric tokens) from the current scanner
/// position to end-of-input/newline. Used by `parse_strings_section`.
fn collect_modifiers(s: &mut Scanner<'_>) -> Vec<Modifier> {
    let mut out = Vec::new();
    loop {
        s.eat_inline_ws();
        match s.peek_byte() {
            None | Some(b'\n') => break,
            _ => {}
        }
        match s.consume_modifier_word() {
            Some(word) => {
                if let Some(m) = Modifier::from_str(&word) {
                    out.push(m);
                }
            }
            None => break,
        }
    }
    out
}

/// Read `key=value` and `key="value"` pairs from the current scanner
/// position to end-of-input/newline. Bareword keys without `=value`
/// are silently dropped (matches prior Rust behavior; bareword flags
/// are handled by `collect_modifiers` for the strings section).
fn collect_kv_params_until_newline(s: &mut Scanner<'_>) -> HashMap<String, String> {
    let mut params = HashMap::new();
    loop {
        s.eat_inline_ws();
        match s.peek_byte() {
            None | Some(b'\n') => break,
            _ => {}
        }
        let key = match s.consume_modifier_word() {
            Some(k) => k,
            None => {
                // Unrecognized lead char — bump until next ws to avoid
                // an infinite loop on malformed input.
                while let Some(b) = s.peek_byte() {
                    if b.is_ascii_whitespace() {
                        break;
                    }
                    s.bump();
                }
                continue;
            }
        };
        if s.peek_byte() == Some(b'=') {
            s.bump();
            if let Ok(value) = s.consume_kv_value() {
                params.insert(key, value);
            }
        }
        // Bareword key (no `=`): drop silently.
    }
    params
}

fn skip_ws_and_newlines(s: &mut Scanner<'_>) {
    while let Some(b) = s.peek_byte() {
        if b.is_ascii_whitespace() {
            s.bump();
        } else {
            break;
        }
    }
}
