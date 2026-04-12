/// Section parsers for individual rule body sections (meta, strings, etc.).
use std::collections::HashMap;
use std::sync::LazyLock;
use regex::Regex;

use crate::error::SyaraError;
use crate::models::{
    ClassifierRule, LLMRule, Modifier, PHashRule, SimilarityRule, StringRule,
};

// ── Compiled regexes (BUG-004) ──────────────────────────────────────────────

static META_KV_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).unwrap()
});

/// BUG-020: supports escaped quotes via `(?:[^"\\]|\\.)*`
static STRING_PATTERN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\$\w+)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(.*)"#).unwrap()
});

static REGEX_PATTERN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\$\w+)\s*=\s*/([^/]*)/(i?)\s*(.*)").unwrap()
});

/// Shared pattern for similarity, phash, classifier, llm section lines.
static SECTION_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\$\w+)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(.*)"#).unwrap()
});

static KV_PARAMS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\w+)=(?:"([^"]*)"|([^\s"]+))"#).unwrap()
});

// ── Section content extractor ───────────────────────────────────────────────

pub(crate) fn section_content<'a>(
    body: &'a str,
    section: &str,
    next_sections: &[&str],
) -> Option<&'a str> {
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

// ── Individual section parsers ──────────────────────────────────────────────

pub(crate) fn parse_meta_section(body: &str) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    let content = match section_content(
        body,
        "meta",
        &["strings", "similarity", "phash", "classifier", "llm", "condition"],
    ) {
        Some(c) => c,
        None => return meta,
    };

    for cap in META_KV_RE.captures_iter(content) {
        meta.insert(cap[1].to_owned(), cap[2].to_owned());
    }
    meta
}

pub(crate) fn parse_strings_section(body: &str) -> Result<Vec<StringRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(
        body,
        "strings",
        &["similarity", "phash", "classifier", "llm", "condition"],
    ) {
        Some(c) => c,
        None => return Ok(rules),
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(cap) = STRING_PATTERN_RE.captures(line) {
            let identifier = cap[1].to_owned();
            let pattern = unescape_string(&cap[2]);
            let mods = parse_string_modifiers(&cap[3]);
            rules.push(StringRule {
                identifier,
                pattern,
                modifiers: mods,
                is_regex: false,
            });
            continue;
        }

        if let Some(cap) = REGEX_PATTERN_RE.captures(line) {
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

pub(crate) fn parse_similarity_section(body: &str) -> Result<Vec<SimilarityRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(
        body,
        "similarity",
        &["phash", "classifier", "llm", "condition"],
    ) {
        Some(c) => c,
        None => return Ok(rules),
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match SECTION_LINE_RE.captures(line) {
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

pub(crate) fn parse_phash_section(body: &str) -> Result<Vec<PHashRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "phash", &["classifier", "llm", "condition"]) {
        Some(c) => c,
        None => return Ok(rules),
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match SECTION_LINE_RE.captures(line) {
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

pub(crate) fn parse_classifier_section(body: &str) -> Result<Vec<ClassifierRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "classifier", &["llm", "condition"]) {
        Some(c) => c,
        None => return Ok(rules),
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match SECTION_LINE_RE.captures(line) {
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

pub(crate) fn parse_llm_section(body: &str) -> Result<Vec<LLMRule>, SyaraError> {
    let mut rules = Vec::new();
    let content = match section_content(body, "llm", &["condition"]) {
        Some(c) => c,
        None => return Ok(rules),
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cap = match SECTION_LINE_RE.captures(line) {
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

pub(crate) fn parse_condition_section(body: &str) -> String {
    let content = match section_content(body, "condition", &[]) {
        Some(c) => c,
        None => return String::new(),
    };
    content.trim().to_owned()
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Process escape sequences in a parsed string literal.
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

fn parse_string_modifiers(s: &str) -> Vec<Modifier> {
    s.split_whitespace()
        .filter_map(|tok| {
            let tok = tok.trim_end_matches(|c: char| !c.is_alphanumeric());
            Modifier::from_str(tok)
        })
        .collect()
}

/// Parse `key=value` and `key="value"` pairs from a parameter string.
fn parse_kv_params(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for cap in KV_PARAMS_RE.captures_iter(s) {
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
