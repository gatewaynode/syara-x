/// Component registry — replaces Python's importlib-based ConfigManager.
///
/// Maps component names to boxed trait objects. Built-in implementations are
/// registered at construction. Users may add custom components via `register_*`.
use std::collections::HashMap;
use crate::engine::cleaner::{AggressiveCleaner, DefaultCleaner, NoOpCleaner, TextCleaner};
use crate::engine::chunker::{
    Chunker, FixedSizeChunker, NoChunker, ParagraphChunker, SentenceChunker, WordChunker,
};
#[cfg(any(feature = "sbert", feature = "classifier", feature = "llm", feature = "burn-llm", feature = "phash"))]
use crate::error::SyaraError;

#[cfg(feature = "sbert")]
use crate::engine::semantic_matcher::{HttpEmbeddingMatcher, SemanticMatcher};

#[cfg(feature = "classifier")]
use crate::engine::classifier::{HttpEmbeddingClassifier, TextClassifier};

#[cfg(any(feature = "llm", feature = "burn-llm"))]
use crate::engine::llm_evaluator::LLMEvaluator;
#[cfg(feature = "llm")]
use crate::engine::llm_evaluator::OllamaEvaluator;

#[cfg(feature = "phash")]
use crate::engine::phash_matcher::{
    AudioHashMatcher, ImageHashMatcher, PHashMatcher, VideoHashMatcher,
};

pub struct Registry {
    cleaners: HashMap<String, Box<dyn TextCleaner>>,
    chunkers: HashMap<String, Box<dyn Chunker>>,
    #[cfg(feature = "sbert")]
    semantic_matchers: HashMap<String, Box<dyn SemanticMatcher>>,
    #[cfg(feature = "classifier")]
    classifiers: HashMap<String, Box<dyn TextClassifier>>,
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    llm_evaluators: HashMap<String, Box<dyn LLMEvaluator>>,
    #[cfg(feature = "phash")]
    phash_matchers: HashMap<String, Box<dyn PHashMatcher>>,
}

impl Registry {
    pub fn new() -> Self {
        let mut r = Self {
            cleaners: HashMap::new(),
            chunkers: HashMap::new(),
            #[cfg(feature = "sbert")]
            semantic_matchers: HashMap::new(),
            #[cfg(feature = "classifier")]
            classifiers: HashMap::new(),
            #[cfg(any(feature = "llm", feature = "burn-llm"))]
            llm_evaluators: HashMap::new(),
            #[cfg(feature = "phash")]
            phash_matchers: HashMap::new(),
        };
        r.register_builtins();
        r
    }

    fn register_builtins(&mut self) {
        self.cleaners
            .insert("default_cleaning".into(), Box::new(DefaultCleaner));
        self.cleaners
            .insert("no_op".into(), Box::new(NoOpCleaner));
        self.cleaners
            .insert("aggressive_cleaning".into(), Box::new(AggressiveCleaner));

        self.chunkers
            .insert("no_chunking".into(), Box::new(NoChunker));
        self.chunkers
            .insert("sentence_chunking".into(), Box::new(SentenceChunker));
        self.chunkers.insert(
            "fixed_size_chunking".into(),
            Box::new(FixedSizeChunker::new(512, 50)),
        );
        self.chunkers
            .insert("paragraph_chunking".into(), Box::new(ParagraphChunker));
        self.chunkers
            .insert("word_chunking".into(), Box::new(WordChunker::new(100)));

        #[cfg(feature = "sbert")]
        self.semantic_matchers.insert(
            "sbert".into(),
            Box::new(HttpEmbeddingMatcher::new(
                "http://localhost:11434/api/embed",
                "all-minilm",
            )),
        );

        #[cfg(feature = "classifier")]
        self.classifiers.insert(
            "tuned-sbert".into(),
            Box::new(HttpEmbeddingClassifier::new(
                "http://localhost:11434/api/embed",
                "all-minilm",
            )),
        );

        #[cfg(feature = "llm")]
        self.llm_evaluators.insert(
            "ollama".into(),
            Box::new(OllamaEvaluator::new(
                "http://localhost:11434/api/chat",
                "llama3.2",
            )),
        );

        #[cfg(feature = "phash")]
        {
            self.phash_matchers
                .insert("imagehash".into(), Box::new(ImageHashMatcher::default()));
            self.phash_matchers
                .insert("audiohash".into(), Box::new(AudioHashMatcher));
            self.phash_matchers
                .insert("videohash".into(), Box::new(VideoHashMatcher));
        }
    }

    #[cfg(any(feature = "sbert", feature = "classifier", feature = "llm", feature = "burn-llm"))]
    pub fn get_cleaner(&self, name: &str) -> Result<&dyn TextCleaner, SyaraError> {
        self.cleaners
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| SyaraError::ComponentNotFound {
                kind: "cleaner".into(),
                name: name.into(),
            })
    }

    #[cfg(any(feature = "sbert", feature = "classifier", feature = "llm", feature = "burn-llm"))]
    pub fn get_chunker(&self, name: &str) -> Result<&dyn Chunker, SyaraError> {
        self.chunkers
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| SyaraError::ComponentNotFound {
                kind: "chunker".into(),
                name: name.into(),
            })
    }

    pub fn register_cleaner(&mut self, name: impl Into<String>, cleaner: Box<dyn TextCleaner>) {
        self.cleaners.insert(name.into(), cleaner);
    }

    pub fn register_chunker(&mut self, name: impl Into<String>, chunker: Box<dyn Chunker>) {
        self.chunkers.insert(name.into(), chunker);
    }

    #[cfg(feature = "sbert")]
    pub fn get_semantic_matcher(
        &self,
        name: &str,
    ) -> Result<&dyn SemanticMatcher, SyaraError> {
        self.semantic_matchers
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| SyaraError::ComponentNotFound {
                kind: "semantic_matcher".into(),
                name: name.into(),
            })
    }

    #[cfg(feature = "sbert")]
    pub fn register_semantic_matcher(
        &mut self,
        name: impl Into<String>,
        matcher: Box<dyn SemanticMatcher>,
    ) {
        self.semantic_matchers.insert(name.into(), matcher);
    }

    #[cfg(feature = "classifier")]
    pub fn get_classifier(&self, name: &str) -> Result<&dyn TextClassifier, crate::error::SyaraError> {
        self.classifiers
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| crate::error::SyaraError::ComponentNotFound {
                kind: "classifier".into(),
                name: name.into(),
            })
    }

    #[cfg(feature = "classifier")]
    pub fn register_classifier(
        &mut self,
        name: impl Into<String>,
        classifier: Box<dyn TextClassifier>,
    ) {
        self.classifiers.insert(name.into(), classifier);
    }

    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    pub fn get_llm_evaluator(
        &self,
        name: &str,
    ) -> Result<&dyn LLMEvaluator, crate::error::SyaraError> {
        self.llm_evaluators
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| crate::error::SyaraError::ComponentNotFound {
                kind: "llm_evaluator".into(),
                name: name.into(),
            })
    }

    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    pub fn register_llm_evaluator(
        &mut self,
        name: impl Into<String>,
        evaluator: Box<dyn LLMEvaluator>,
    ) {
        self.llm_evaluators.insert(name.into(), evaluator);
    }

    #[cfg(feature = "phash")]
    pub fn get_phash_matcher(
        &self,
        name: &str,
    ) -> Result<&dyn PHashMatcher, crate::error::SyaraError> {
        self.phash_matchers
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| crate::error::SyaraError::ComponentNotFound {
                kind: "phash_matcher".into(),
                name: name.into(),
            })
    }

    #[cfg(feature = "phash")]
    pub fn register_phash_matcher(
        &mut self,
        name: impl Into<String>,
        matcher: Box<dyn PHashMatcher>,
    ) {
        self.phash_matchers.insert(name.into(), matcher);
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}
