use std::collections::HashMap;

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
            llm_name: "ollama".into(),
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
}

/// Details of a single matched pattern within a rule.
#[derive(Debug, Clone)]
pub struct MatchDetail {
    pub identifier: String,
    pub matched_text: String,
    pub start_pos: i64,
    pub end_pos: i64,
    pub score: f64,
    pub explanation: String,
}

impl MatchDetail {
    pub fn new(identifier: impl Into<String>, matched_text: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            matched_text: matched_text.into(),
            start_pos: -1,
            end_pos: -1,
            score: 1.0,
            explanation: String::new(),
        }
    }

    pub fn with_position(mut self, start: usize, end: usize) -> Self {
        self.start_pos = start as i64;
        self.end_pos = end as i64;
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
    pub fn no_match(rule: &Rule) -> Self {
        Self {
            rule_name: rule.name.clone(),
            tags: rule.tags.clone(),
            meta: rule.meta.clone(),
            matched: false,
            matched_patterns: HashMap::new(),
        }
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
