use std::collections::HashMap;

use crate::condition::Expr;

/// Modifiers applicable to a string pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Modifier {
    NoCase,
    Wide,
    Ascii,
    Dotall,
    FullWord,
}

impl Modifier {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "nocase" => Some(Self::NoCase),
            "wide" => Some(Self::Wide),
            "ascii" => Some(Self::Ascii),
            "dotall" => Some(Self::Dotall),
            "fullword" => Some(Self::FullWord),
            _ => None,
        }
    }
}

/// Traditional string or regex pattern.
#[derive(Debug, Clone)]
pub struct StringRule {
    pub identifier: String,
    pub pattern: String,
    pub modifiers: Vec<Modifier>,
    pub is_regex: bool,
}

/// Semantic similarity pattern.
#[derive(Debug, Clone)]
pub struct SimilarityRule {
    pub identifier: String,
    pub pattern: String,
    pub threshold: f64,
    pub cleaner_name: String,
    pub chunker_name: String,
    pub matcher_name: String,
}

impl Default for SimilarityRule {
    fn default() -> Self {
        Self {
            identifier: String::new(),
            pattern: String::new(),
            threshold: 0.8,
            cleaner_name: "default_cleaning".into(),
            chunker_name: "no_chunking".into(),
            matcher_name: "sbert".into(),
        }
    }
}

/// Perceptual hash pattern for binary files (images, audio, video).
#[derive(Debug, Clone)]
pub struct PHashRule {
    pub identifier: String,
    pub file_path: String,
    pub threshold: f64,
    pub phash_name: String,
}

impl Default for PHashRule {
    fn default() -> Self {
        Self {
            identifier: String::new(),
            file_path: String::new(),
            threshold: 0.9,
            phash_name: "imagehash".into(),
        }
    }
}

/// ML classifier-based pattern.
#[derive(Debug, Clone)]
pub struct ClassifierRule {
    pub identifier: String,
    pub pattern: String,
    pub threshold: f64,
    pub cleaner_name: String,
    pub chunker_name: String,
    pub classifier_name: String,
}

impl Default for ClassifierRule {
    fn default() -> Self {
        Self {
            identifier: String::new(),
            pattern: String::new(),
            threshold: 0.7,
            cleaner_name: "default_cleaning".into(),
            chunker_name: "no_chunking".into(),
            classifier_name: "tuned-sbert".into(),
        }
    }
}

/// LLM-based evaluation pattern.
#[derive(Debug, Clone)]
pub struct LLMRule {
    pub identifier: String,
    pub pattern: String,
    pub llm_name: String,
    pub cleaner_name: String,
    pub chunker_name: String,
}

impl Default for LLMRule {
    fn default() -> Self {
        Self {
            identifier: String::new(),
            pattern: String::new(),
            llm_name: "openai-api-compatible".into(),
            cleaner_name: "no_op".into(),
            chunker_name: "no_chunking".into(),
        }
    }
}

/// A complete parsed rule.
#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub name: String,
    pub tags: Vec<String>,
    pub meta: HashMap<String, String>,
    pub strings: Vec<StringRule>,
    pub similarity: Vec<SimilarityRule>,
    pub phash: Vec<PHashRule>,
    pub classifier: Vec<ClassifierRule>,
    pub llm: Vec<LLMRule>,
    pub condition: String,
    /// Pre-compiled condition AST, populated by the compiler.
    pub compiled_condition: Option<Expr>,
}

/// Details of a single matched pattern within a rule.
#[derive(Debug, Clone)]
pub struct MatchDetail {
    pub identifier: String,
    pub matched_text: String,
    pub start_pos: Option<usize>,
    pub end_pos: Option<usize>,
    pub score: f64,
    pub explanation: String,
}

impl MatchDetail {
    pub fn new(identifier: impl Into<String>, matched_text: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            matched_text: matched_text.into(),
            start_pos: None,
            end_pos: None,
            score: 1.0,
            explanation: String::new(),
        }
    }

    pub fn with_position(mut self, start: usize, end: usize) -> Self {
        self.start_pos = Some(start);
        self.end_pos = Some(end);
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }
}

/// Result of evaluating a single rule against input.
#[derive(Debug, Clone)]
pub struct Match {
    pub rule_name: String,
    pub tags: Vec<String>,
    pub meta: HashMap<String, String>,
    pub matched: bool,
    pub matched_patterns: HashMap<String, Vec<MatchDetail>>,
}

impl Match {
    /// BUG-024: non-matching results carry only the rule name — no cloned
    /// tags/meta, since consumers filter on `matched` first.
    pub fn no_match(rule: &Rule) -> Self {
        Self {
            rule_name: rule.name.clone(),
            tags: Vec::new(),
            meta: HashMap::new(),
            matched: false,
            matched_patterns: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BUG-017: positions use Option<usize> instead of i64 sentinels ───

    #[test]
    fn test_match_detail_default_positions_are_none() {
        let detail = MatchDetail::new("$s1", "hello");
        assert_eq!(detail.start_pos, None);
        assert_eq!(detail.end_pos, None);
    }

    #[test]
    fn test_match_detail_with_position() {
        let detail = MatchDetail::new("$s1", "hello").with_position(10, 15);
        assert_eq!(detail.start_pos, Some(10));
        assert_eq!(detail.end_pos, Some(15));
    }

    #[test]
    fn test_match_detail_with_score() {
        let detail = MatchDetail::new("$s1", "hello").with_score(0.95);
        assert!((detail.score - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_detail_default_score_is_one() {
        let detail = MatchDetail::new("$s1", "hello");
        assert!((detail.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_no_match_has_empty_patterns() {
        let rule = Rule {
            name: "test".to_string(),
            ..Default::default()
        };
        let m = Match::no_match(&rule);
        assert!(!m.matched);
        assert!(m.matched_patterns.is_empty());
        assert_eq!(m.rule_name, "test");
    }

    // ── BUG-024: no_match uses empty vecs, not clones ─────────────────────

    #[test]
    fn test_no_match_does_not_clone_tags_or_meta() {
        let rule = Rule {
            name: "tagged".to_string(),
            tags: vec!["security".to_string(), "test".to_string()],
            meta: {
                let mut m = HashMap::new();
                m.insert("author".to_string(), "tester".to_string());
                m
            },
            ..Default::default()
        };
        let m = Match::no_match(&rule);
        assert!(!m.matched);
        assert!(m.tags.is_empty(), "non-match should have empty tags");
        assert!(m.meta.is_empty(), "non-match should have empty meta");
    }

    #[test]
    fn test_modifier_from_str() {
        assert_eq!(Modifier::from_str("nocase"), Some(Modifier::NoCase));
        assert_eq!(Modifier::from_str("wide"), Some(Modifier::Wide));
        assert_eq!(Modifier::from_str("ascii"), Some(Modifier::Ascii));
        assert_eq!(Modifier::from_str("dotall"), Some(Modifier::Dotall));
        assert_eq!(Modifier::from_str("fullword"), Some(Modifier::FullWord));
        assert_eq!(Modifier::from_str("unknown"), None);
    }

    #[test]
    fn test_match_display_matched() {
        let m = Match {
            rule_name: "test_rule".to_string(),
            tags: vec![],
            meta: HashMap::new(),
            matched: true,
            matched_patterns: {
                let mut mp = HashMap::new();
                mp.insert(
                    "$s1".to_string(),
                    vec![MatchDetail::new("$s1", "x")],
                );
                mp
            },
        };
        let display = format!("{}", m);
        assert!(display.contains("matched=true"));
        assert!(display.contains("patterns=1"));
    }

    #[test]
    fn test_match_display_not_matched() {
        let rule = Rule {
            name: "test_rule".to_string(),
            ..Default::default()
        };
        let m = Match::no_match(&rule);
        let display = format!("{}", m);
        assert!(display.contains("matched=false"));
    }
}

impl std::fmt::Display for Match {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.matched {
            write!(f, "Match(rule='{}', matched=false)", self.rule_name)
        } else {
            let count: usize = self.matched_patterns.values().map(|v| v.len()).sum();
            write!(
                f,
                "Match(rule='{}', matched=true, patterns={})",
                self.rule_name, count
            )
        }
    }
}
