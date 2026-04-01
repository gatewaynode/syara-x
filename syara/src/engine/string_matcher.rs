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

    fn compile(&mut self, rule: &StringRule) -> Result<&Regex, SyaraError> {
        let key = Self::cache_key(rule);
        if self.cache.contains_key(&key) {
            return Ok(&self.cache[&key]);
        }

        let nocase = rule.modifiers.contains(&Modifier::NoCase);
        let dotall = rule.modifiers.contains(&Modifier::Dotall);

        let pattern = if rule.is_regex {
            rule.pattern.clone()
        } else {
            regex::escape(&rule.pattern)
        };

        let regex = RegexBuilder::new(&pattern)
            .case_insensitive(nocase)
            .dot_matches_new_line(dotall)
            .build()
            .map_err(|e| SyaraError::InvalidPattern {
                pattern: rule.pattern.clone(),
                reason: e.to_string(),
            })?;

        self.cache.insert(key.clone(), regex);
        Ok(&self.cache[&key])
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
            let regex = self.compile(rule)?;
            // Clone to avoid borrow conflict — regex is small
            let regex = regex.clone();
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
        let mut details = Vec::new();

        if rule.is_regex {
            // Best-effort: strip null bytes and apply original regex
            let stripped: String = text.chars().filter(|&c| c != '\x00').collect();
            let regex = RegexBuilder::new(&rule.pattern)
                .case_insensitive(nocase)
                .build()
                .map_err(|e| SyaraError::InvalidPattern {
                    pattern: rule.pattern.clone(),
                    reason: e.to_string(),
                })?;
            for m in regex.find_iter(&stripped) {
                details.push(
                    MatchDetail::new(rule.identifier.clone(), m.as_str())
                        .with_position(m.start(), m.end()),
                );
            }
        } else {
            // Interleave each char with a null byte (UTF-16LE representation)
            let wide: String = rule
                .pattern
                .chars()
                .flat_map(|c| [c, '\x00'])
                .collect();
            let pattern = regex::escape(&wide);
            let regex = RegexBuilder::new(&pattern)
                .case_insensitive(nocase)
                .build()
                .map_err(|e| SyaraError::InvalidPattern {
                    pattern: rule.pattern.clone(),
                    reason: e.to_string(),
                })?;
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
