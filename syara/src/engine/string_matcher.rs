use std::collections::HashMap;
use regex::{Regex, RegexBuilder};
use crate::models::{Modifier, MatchDetail, StringRule};
use crate::error::SyaraError;

/// Handles string and regex pattern matching with YARA modifier semantics.
/// Maintains a compiled-regex cache keyed by pattern + modifier set.
pub struct StringMatcher {
    cache: HashMap<String, Regex>,
}

impl StringMatcher {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    fn cache_key(rule: &StringRule) -> String {
        let mut mods: Vec<&str> = rule
            .modifiers
            .iter()
            .map(|m| match m {
                Modifier::NoCase => "nocase",
                Modifier::Wide => "wide",
                Modifier::Ascii => "ascii",
                Modifier::Dotall => "dotall",
                Modifier::FullWord => "fullword",
            })
            .collect();
        mods.sort_unstable();
        format!("{}:{}", rule.pattern, mods.join(","))
    }

    fn compile(&mut self, rule: &StringRule) -> Result<(), SyaraError> {
        let key = Self::cache_key(rule);
        if self.cache.contains_key(&key) {
            return Ok(());
        }

        let nocase = rule.modifiers.contains(&Modifier::NoCase);
        let dotall = rule.modifiers.contains(&Modifier::Dotall);
        let fullword = rule.modifiers.contains(&Modifier::FullWord);

        let mut pattern = if rule.is_regex {
            rule.pattern.clone()
        } else {
            regex::escape(&rule.pattern)
        };

        if fullword {
            pattern = format!(r"\b(?:{})\b", pattern);
        }

        let regex = RegexBuilder::new(&pattern)
            .case_insensitive(nocase)
            .dot_matches_new_line(dotall)
            .build()
            .map_err(|e| SyaraError::InvalidPattern {
                pattern: rule.pattern.clone(),
                reason: e.to_string(),
            })?;

        self.cache.insert(key, regex);
        Ok(())
    }

    pub fn match_rule(
        &mut self,
        rule: &StringRule,
        text: &str,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let search_wide = rule.modifiers.contains(&Modifier::Wide);
        let search_ascii = rule.modifiers.contains(&Modifier::Ascii) || !search_wide;

        let mut details = Vec::new();

        if search_ascii {
            self.compile(rule)?;
            let key = Self::cache_key(rule);
            let regex = &self.cache[&key];
            for m in regex.find_iter(text) {
                details.push(
                    MatchDetail::new(rule.identifier.clone(), m.as_str())
                        .with_position(m.start(), m.end()),
                );
            }
        }

        if search_wide {
            details.extend(self.match_wide(rule, text)?);
        }

        Ok(details)
    }

    fn match_wide(
        &mut self,
        rule: &StringRule,
        text: &str,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let nocase = rule.modifiers.contains(&Modifier::NoCase);
        let fullword = rule.modifiers.contains(&Modifier::FullWord);
        let mut details = Vec::new();

        if rule.is_regex {
            // Build position map from stripped text back to original
            let mut stripped = String::new();
            let mut pos_map: Vec<usize> = Vec::new(); // stripped byte idx -> original byte idx
            for (orig_idx, c) in text.char_indices() {
                if c != '\x00' {
                    for _ in 0..c.len_utf8() {
                        pos_map.push(orig_idx);
                    }
                    stripped.push(c);
                }
            }
            // Sentinel for end positions
            pos_map.push(text.len());

            let wide_key = format!("{}:wide_regex", Self::cache_key(rule));
            if !self.cache.contains_key(&wide_key) {
                let mut pattern = rule.pattern.clone();
                if fullword {
                    pattern = format!(r"\b(?:{})\b", pattern);
                }
                let regex = RegexBuilder::new(&pattern)
                    .case_insensitive(nocase)
                    .build()
                    .map_err(|e| SyaraError::InvalidPattern {
                        pattern: rule.pattern.clone(),
                        reason: e.to_string(),
                    })?;
                self.cache.insert(wide_key.clone(), regex);
            }
            let regex = &self.cache[&wide_key];
            for m in regex.find_iter(&stripped) {
                let orig_start = pos_map.get(m.start()).copied().unwrap_or(0);
                let orig_end = pos_map.get(m.end()).copied().unwrap_or(text.len());
                details.push(
                    MatchDetail::new(rule.identifier.clone(), m.as_str())
                        .with_position(orig_start, orig_end),
                );
            }
        } else {
            let wide_key = format!("{}:wide_literal", Self::cache_key(rule));
            if !self.cache.contains_key(&wide_key) {
                let wide: String = rule
                    .pattern
                    .chars()
                    .flat_map(|c| [c, '\x00'])
                    .collect();
                let mut pattern = regex::escape(&wide);
                if fullword {
                    pattern = format!(r"\b(?:{})\b", pattern);
                }
                let regex = RegexBuilder::new(&pattern)
                    .case_insensitive(nocase)
                    .build()
                    .map_err(|e| SyaraError::InvalidPattern {
                        pattern: rule.pattern.clone(),
                        reason: e.to_string(),
                    })?;
                self.cache.insert(wide_key.clone(), regex);
            }
            let regex = &self.cache[&wide_key];
            for m in regex.find_iter(text) {
                details.push(
                    MatchDetail::new(rule.identifier.clone(), m.as_str())
                        .with_position(m.start(), m.end()),
                );
            }
        }

        Ok(details)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for StringMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal(id: &str, pattern: &str, mods: Vec<Modifier>) -> StringRule {
        StringRule {
            identifier: id.to_string(),
            pattern: pattern.to_string(),
            modifiers: mods,
            is_regex: false,
        }
    }

    fn regex_rule(id: &str, pattern: &str, mods: Vec<Modifier>) -> StringRule {
        StringRule {
            identifier: id.to_string(),
            pattern: pattern.to_string(),
            modifiers: mods,
            is_regex: true,
        }
    }

    // ── BUG-002: fullword modifier must add word boundaries ─────────────

    #[test]
    fn test_fullword_literal_no_substring_match() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "cat", vec![Modifier::FullWord]);

        // "cat" as a standalone word
        let hits = matcher.match_rule(&rule, "the cat sat").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].matched_text, "cat");

        // "cat" as substring of "concatenate" — must NOT match
        let hits = matcher.match_rule(&rule, "concatenate").unwrap();
        assert!(hits.is_empty(), "fullword must not match substrings");
    }

    #[test]
    fn test_fullword_regex() {
        let mut matcher = StringMatcher::new();
        let rule = regex_rule("$r1", "err(or)?", vec![Modifier::FullWord]);

        let hits = matcher.match_rule(&rule, "an error occurred").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].matched_text, "error");

        // Should not match inside "erroneous"
        let hits = matcher.match_rule(&rule, "erroneous").unwrap();
        assert!(hits.is_empty(), "fullword regex must not match substrings");
    }

    #[test]
    fn test_no_fullword_matches_substring() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "cat", vec![]);

        let hits = matcher.match_rule(&rule, "concatenate").unwrap();
        assert_eq!(hits.len(), 1, "without fullword, substrings should match");
    }

    // ── BUG-026: wide regex positions map to original text ──────────────

    #[test]
    fn test_wide_regex_positions_map_to_original() {
        let mut matcher = StringMatcher::new();
        let rule = regex_rule("$r1", "he.lo", vec![Modifier::Wide]);

        // Simulate wide-encoded text: "hello" with null bytes interleaved
        let wide_text = "h\x00e\x00l\x00l\x00o\x00";
        let hits = matcher.match_rule(&rule, wide_text).unwrap();
        assert!(!hits.is_empty(), "wide regex should match null-stripped text");

        // Positions should reference the original wide text, not the stripped version
        let h = &hits[0];
        assert!(h.start_pos.is_some());
        assert!(h.end_pos.is_some());
        let start = h.start_pos.unwrap();
        let end = h.end_pos.unwrap();
        // The match spans the original wide-encoded bytes
        assert!(end > start);
        assert!(end <= wide_text.len());
    }

    #[test]
    fn test_wide_literal_match() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "AB", vec![Modifier::Wide]);

        let wide_text = "A\x00B\x00";
        let hits = matcher.match_rule(&rule, wide_text).unwrap();
        assert_eq!(hits.len(), 1);
    }

    // ── Basic modifier tests ────────────────────────────────────────────

    #[test]
    fn test_literal_match() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "hello", vec![]);
        let hits = matcher.match_rule(&rule, "say hello world").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].matched_text, "hello");
        assert_eq!(hits[0].start_pos, Some(4));
        assert_eq!(hits[0].end_pos, Some(9));
    }

    #[test]
    fn test_regex_match() {
        let mut matcher = StringMatcher::new();
        let rule = regex_rule("$r1", r"\d{3}-\d{4}", vec![]);
        let hits = matcher.match_rule(&rule, "call 555-1234 now").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].matched_text, "555-1234");
    }

    #[test]
    fn test_nocase_modifier() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "hello", vec![Modifier::NoCase]);
        let hits = matcher.match_rule(&rule, "say HELLO there").unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_dotall_modifier() {
        let mut matcher = StringMatcher::new();
        let rule = regex_rule("$r1", "start.*end", vec![Modifier::Dotall]);
        let hits = matcher.match_rule(&rule, "start\nmiddle\nend").unwrap();
        assert_eq!(hits.len(), 1, "dotall should make . match newlines");
    }

    #[test]
    fn test_dotall_without_flag_no_newline() {
        let mut matcher = StringMatcher::new();
        let rule = regex_rule("$r1", "start.*end", vec![]);
        let hits = matcher.match_rule(&rule, "start\nend").unwrap();
        assert!(hits.is_empty(), "without dotall, . should not match newlines");
    }

    #[test]
    fn test_multiple_matches() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "ab", vec![]);
        let hits = matcher.match_rule(&rule, "ab cd ab ef ab").unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn test_no_match_returns_empty() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "xyz", vec![]);
        let hits = matcher.match_rule(&rule, "nothing here").unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_special_chars_escaped_in_literal() {
        let mut matcher = StringMatcher::new();
        // Literal "a.b" should not match "axb" — the dot must be escaped
        let rule = literal("$s1", "a.b", vec![]);
        let hits = matcher.match_rule(&rule, "axb").unwrap();
        assert!(hits.is_empty());
        let hits = matcher.match_rule(&rule, "a.b").unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_cache_reuse() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "test", vec![]);
        // First call compiles and caches
        let _ = matcher.match_rule(&rule, "test").unwrap();
        assert!(!matcher.cache.is_empty());
        // Second call uses cache — same result
        let hits = matcher.match_rule(&rule, "test again test").unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_clear_cache() {
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "x", vec![]);
        let _ = matcher.match_rule(&rule, "x").unwrap();
        assert!(!matcher.cache.is_empty());
        matcher.clear_cache();
        assert!(matcher.cache.is_empty());
    }

    #[test]
    fn test_wide_and_ascii_combined() {
        // When both wide and ascii are specified, both searches run
        let mut matcher = StringMatcher::new();
        let rule = literal("$s1", "AB", vec![Modifier::Wide, Modifier::Ascii]);
        // Text contains both plain "AB" and wide "A\0B\0"
        let text = "AB and A\x00B\x00";
        let hits = matcher.match_rule(&rule, text).unwrap();
        // Should get at least the ASCII match
        assert!(hits.len() >= 1);
    }

    /// Inline `(?m)` parity: in multiline mode `^` anchors after newlines,
    /// not just at the start of input. Confirms the regex crate's flag
    /// syntax is honoured end-to-end through StringMatcher.
    #[test]
    fn test_multiline_inline_flag() {
        let mut matcher = StringMatcher::new();
        let rule = regex_rule("$s", "(?m)^foo", vec![]);

        let hits = matcher.match_rule(&rule, "bar\nfoo").unwrap();
        assert_eq!(hits.len(), 1, "(?m)^foo must match after a newline");

        let hits = matcher.match_rule(&rule, "barfoo").unwrap();
        assert!(hits.is_empty(), "^ must not match inside a line");
    }
}
