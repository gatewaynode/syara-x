/// Execution engine for compiled rules.
///
/// Runs matchers in cost order: strings → similarity → phash → classifier → LLM.
/// LLM calls are short-circuited via `is_identifier_needed`.
use std::collections::HashMap;
use std::path::Path;

use crate::cache::TextCache;
use crate::condition;
use crate::config::Registry;
use crate::engine::string_matcher::StringMatcher;
use crate::models::{Match, MatchDetail, Rule};

pub struct CompiledRules {
    pub(crate) rules: Vec<Rule>,
    pub(crate) registry: Registry,
}

impl CompiledRules {
    /// Number of compiled rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl CompiledRules {
    pub(crate) fn new(rules: Vec<Rule>, registry: Registry) -> Self {
        Self { rules, registry }
    }

    /// Match text against all text-based rules.
    /// PHash rules are skipped (require a file path).
    pub fn scan(&self, text: &str) -> Vec<Match> {
        let mut cache = TextCache::new();
        let mut string_matcher = StringMatcher::new();
        let mut results = Vec::with_capacity(self.rules.len());

        for rule in &self.rules {
            let m = self.execute_rule(rule, text, None, &mut cache, &mut string_matcher);
            results.push(m);
        }

        cache.clear();
        results
    }

    /// Match a file against rules that contain phash patterns.
    pub fn scan_file(&self, path: &Path) -> Vec<Match> {
        let mut cache = TextCache::new();
        let mut string_matcher = StringMatcher::new();
        let mut results = Vec::new();

        for rule in &self.rules {
            if !rule.phash.is_empty() {
                let m =
                    self.execute_rule(rule, "", Some(path), &mut cache, &mut string_matcher);
                results.push(m);
            }
        }

        cache.clear();
        results
    }

    #[allow(unused_variables)]
    fn execute_rule(
        &self,
        rule: &Rule,
        text: &str,
        file_path: Option<&Path>,
        cache: &mut TextCache,
        string_matcher: &mut StringMatcher,
    ) -> Match {
        // Initialise every declared identifier with an empty list so that
        // `all of them` and `all of ($prefix*)` correctly see failed identifiers.
        let mut pattern_matches: HashMap<String, Vec<MatchDetail>> = HashMap::new();
        for r in &rule.strings {
            pattern_matches.insert(r.identifier.clone(), vec![]);
        }
        for r in &rule.similarity {
            pattern_matches.insert(r.identifier.clone(), vec![]);
        }
        for r in &rule.phash {
            pattern_matches.insert(r.identifier.clone(), vec![]);
        }
        for r in &rule.classifier {
            pattern_matches.insert(r.identifier.clone(), vec![]);
        }
        for r in &rule.llm {
            pattern_matches.insert(r.identifier.clone(), vec![]);
        }

        // 1. String patterns (cheapest)
        for string_rule in &rule.strings {
            match string_matcher.match_rule(string_rule, text) {
                Ok(hits) if !hits.is_empty() => {
                    pattern_matches.insert(string_rule.identifier.clone(), hits);
                }
                _ => {}
            }
        }

        // 2. Similarity patterns (moderate cost) — requires sbert feature
        #[cfg(feature = "sbert")]
        for sim_rule in &rule.similarity {
            if let Ok(hits) = self.execute_similarity(sim_rule, text, cache) {
                if !hits.is_empty() {
                    pattern_matches.insert(sim_rule.identifier.clone(), hits);
                }
            }
        }

        // 3. PHash patterns (moderate-to-high cost)
        #[cfg(feature = "phash")]
        if let Some(fp) = file_path {
            for phash_rule in &rule.phash {
                if let Ok(hits) = self.execute_phash(phash_rule, fp) {
                    if !hits.is_empty() {
                        pattern_matches.insert(phash_rule.identifier.clone(), hits);
                    }
                }
            }
        }

        // 4. Classifier patterns (higher cost)
        #[cfg(feature = "classifier")]
        for cls_rule in &rule.classifier {
            if let Ok(hits) = self.execute_classifier(cls_rule, text, cache) {
                if !hits.is_empty() {
                    pattern_matches.insert(cls_rule.identifier.clone(), hits);
                }
            }
        }

        // 5. LLM patterns (highest cost) — short-circuit if not needed
        #[cfg(feature = "llm")]
        if let Ok(expr) = condition::parse(&rule.condition) {
            for llm_rule in &rule.llm {
                if condition::is_identifier_needed(
                    &llm_rule.identifier,
                    &expr,
                    &pattern_matches,
                ) {
                    if let Ok(hits) = self.execute_llm(llm_rule, text, cache) {
                        if !hits.is_empty() {
                            pattern_matches.insert(llm_rule.identifier.clone(), hits);
                        }
                    }
                }
            }
        }

        // Evaluate condition
        let matched = self.evaluate_condition(&rule.condition, &pattern_matches);

        Match {
            rule_name: rule.name.clone(),
            tags: rule.tags.clone(),
            meta: rule.meta.clone(),
            matched,
            matched_patterns: if matched { pattern_matches } else { HashMap::new() },
        }
    }

    fn evaluate_condition(
        &self,
        condition: &str,
        pattern_matches: &HashMap<String, Vec<MatchDetail>>,
    ) -> bool {
        if condition.is_empty() {
            return false;
        }
        match condition::parse(condition) {
            Ok(expr) => condition::evaluate(&expr, pattern_matches),
            Err(_) => false,
        }
    }

    // ── Feature-gated execution helpers ──────────────────────────────────────

    #[cfg(feature = "sbert")]
    fn execute_similarity(
        &self,
        rule: &crate::models::SimilarityRule,
        text: &str,
        cache: &mut TextCache,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let cleaner = self.registry.get_cleaner(&rule.cleaner_name)?;
        let chunker = self.registry.get_chunker(&rule.chunker_name)?;
        let cleaned = get_cleaned(cache, text, cleaner, &rule.cleaner_name);
        let chunks = chunker.chunk(&cleaned);
        let matcher = self.registry.get_semantic_matcher(&rule.matcher_name)?;
        matcher.match_chunks(rule, &chunks)
    }

    #[cfg(feature = "phash")]
    fn execute_phash(
        &self,
        rule: &crate::models::PHashRule,
        file_path: &Path,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let matcher = self.registry.get_phash_matcher(&rule.phash_name)?;
        matcher.match_rule(rule, file_path)
    }

    #[cfg(feature = "classifier")]
    fn execute_classifier(
        &self,
        rule: &crate::models::ClassifierRule,
        text: &str,
        cache: &mut TextCache,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let cleaner = self.registry.get_cleaner(&rule.cleaner_name)?;
        let chunker = self.registry.get_chunker(&rule.chunker_name)?;
        let cleaned = get_cleaned(cache, text, cleaner, &rule.cleaner_name);
        let chunks = chunker.chunk(&cleaned);
        let classifier = self.registry.get_classifier(&rule.classifier_name)?;
        classifier.classify_chunks(rule, &chunks)
    }

    #[cfg(feature = "llm")]
    fn execute_llm(
        &self,
        rule: &crate::models::LLMRule,
        text: &str,
        cache: &mut TextCache,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let cleaner = self.registry.get_cleaner(&rule.cleaner_name)?;
        let chunker = self.registry.get_chunker(&rule.chunker_name)?;
        let cleaned = get_cleaned(cache, text, cleaner, &rule.cleaner_name);
        let chunks = chunker.chunk(&cleaned);
        let evaluator = self.registry.get_llm_evaluator(&rule.llm_name)?;
        evaluator.evaluate_chunks(rule, &chunks)
    }

    // ── Public registration API ───────────────────────────────────────────────

    pub fn register_cleaner(
        &mut self,
        name: impl Into<String>,
        cleaner: Box<dyn crate::engine::cleaner::TextCleaner>,
    ) {
        self.registry.register_cleaner(name, cleaner);
    }

    pub fn register_chunker(
        &mut self,
        name: impl Into<String>,
        chunker: Box<dyn crate::engine::chunker::Chunker>,
    ) {
        self.registry.register_chunker(name, chunker);
    }

    #[cfg(feature = "sbert")]
    pub fn register_semantic_matcher(
        &mut self,
        name: impl Into<String>,
        matcher: Box<dyn crate::engine::semantic_matcher::SemanticMatcher>,
    ) {
        self.registry.register_semantic_matcher(name, matcher);
    }

    #[cfg(feature = "classifier")]
    pub fn register_classifier(
        &mut self,
        name: impl Into<String>,
        classifier: Box<dyn crate::engine::classifier::TextClassifier>,
    ) {
        self.registry.register_classifier(name, classifier);
    }

    #[cfg(feature = "llm")]
    pub fn register_llm_evaluator(
        &mut self,
        name: impl Into<String>,
        evaluator: Box<dyn crate::engine::llm_evaluator::LLMEvaluator>,
    ) {
        self.registry.register_llm_evaluator(name, evaluator);
    }

    #[cfg(feature = "phash")]
    pub fn register_phash_matcher(
        &mut self,
        name: impl Into<String>,
        matcher: Box<dyn crate::engine::phash_matcher::PHashMatcher>,
    ) {
        self.registry.register_phash_matcher(name, matcher);
    }
}

#[allow(dead_code)]
/// Get cleaned text from cache, or clean and store.
fn get_cleaned(
    cache: &mut TextCache,
    text: &str,
    cleaner: &dyn crate::engine::cleaner::TextCleaner,
    cleaner_name: &str,
) -> String {
    if let Some(cached) = cache.get(text, cleaner_name) {
        return cached.to_owned();
    }
    let cleaned = cleaner.clean(text);
    cache.insert(text, cleaner_name, cleaned.clone());
    cleaned
}
