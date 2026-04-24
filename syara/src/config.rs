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
use crate::engine::semantic_matcher::{OpenAiEmbeddingMatcher, SemanticMatcher};

#[cfg(feature = "classifier")]
use crate::engine::classifier::{OpenAiEmbeddingClassifier, TextClassifier};

#[cfg(any(feature = "llm", feature = "burn-llm"))]
use crate::engine::llm_evaluator::LLMEvaluator;
#[cfg(feature = "llm")]
use crate::engine::llm_evaluator::{OllamaEvaluator, OpenAiChatEvaluatorBuilder};

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

        // Default to an OpenAI-compatible `/v1/embeddings` endpoint — the
        // wire format served by LM Studio, vLLM, llama-server, Open WebUI,
        // and openai.com.  Users running Ollama can register
        // `OllamaEmbeddingMatcher` explicitly under the `"sbert"` name.
        #[cfg(feature = "sbert")]
        self.semantic_matchers.insert(
            "sbert".into(),
            Box::new(OpenAiEmbeddingMatcher::new(
                "http://localhost:1234/v1/embeddings",
                "text-embedding-3-small",
            )),
        );

        #[cfg(feature = "classifier")]
        self.classifiers.insert(
            "tuned-sbert".into(),
            Box::new(OpenAiEmbeddingClassifier::new(
                "http://localhost:1234/v1/embeddings",
                "text-embedding-3-small",
            )),
        );

        #[cfg(feature = "llm")]
        {
            // Default LLM evaluator: OpenAI-compatible `/v1/chat/completions`.
            // Served by LM Studio, vLLM, llama-server, Open WebUI, openai.com,
            // and Ollama's `/v1/chat/completions` shim.
            //
            // Honours `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`
            // (and `SYARA_LLM_*` equivalents) unless `SYARA_LLM_NO_ENV=1`.
            // See README for how to scope token exposure.
            let (endpoint, model, api_key, reasoning_effort) =
                resolve_openai_env_defaults();
            let mut builder = OpenAiChatEvaluatorBuilder::new()
                .endpoint(endpoint)
                .model(model);
            if let Some(key) = api_key {
                builder = builder.api_key(key);
            }
            // None → keep builder default ("none")
            // Some("") → explicit disable for strict servers
            // Some(v) → explicit value (low/medium/high/none)
            match reasoning_effort.as_deref() {
                None => {}
                Some("") => {
                    builder = builder.disable_reasoning_effort();
                }
                Some(v) => {
                    builder = builder.reasoning_effort(v.to_string());
                }
            }
            self.llm_evaluators
                .insert("openai-api-compatible".into(), Box::new(builder.build()));

            // Legacy: Ollama's native `/api/chat` endpoint.
            self.llm_evaluators.insert(
                "ollama".into(),
                Box::new(OllamaEvaluator::new(
                    "http://localhost:11434/api/chat",
                    "llama3.2",
                )),
            );
        }

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

    /// Clear any per-scan caches held by registered LLM evaluators.
    ///
    /// Called by [`crate::compiled_rules::CompiledRules::scan`] after each
    /// scan to match the lifecycle of [`crate::cache::TextCache`].
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    pub(crate) fn clear_llm_caches(&self) {
        for evaluator in self.llm_evaluators.values() {
            evaluator.clear_cache();
        }
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

/// Resolve `(endpoint, model, api_key, reasoning_effort)` for the default
/// OpenAI-compatible LLM evaluator.
///
/// Lookup order (first hit wins):
/// 1. `SYARA_LLM_ENDPOINT` / `SYARA_LLM_MODEL` / `SYARA_LLM_API_KEY` /
///    `SYARA_LLM_REASONING_EFFORT` — scoped specifically to SYARA-X so
///    users can set them without leaking into other tools that read
///    `OPENAI_*`.
/// 2. `OPENAI_BASE_URL` (root URL; `/chat/completions` appended if missing) /
///    `OPENAI_MODEL` / `OPENAI_API_KEY` — standard OpenAI SDK conventions.
/// 3. Hardcoded local fallback: `http://localhost:1234/v1/chat/completions`
///    with model `local-model` and no API key.
///
/// `reasoning_effort` semantics:
/// * `None` returned → caller should leave the builder default
///   ([`OpenAiChatEvaluator::DEFAULT_REASONING_EFFORT`]).
/// * `Some("")` returned → caller should call
///   [`OpenAiChatEvaluatorBuilder::disable_reasoning_effort`] (explicit
///   opt-out for strict servers).
/// * `Some(value)` returned → caller should call
///   [`OpenAiChatEvaluatorBuilder::reasoning_effort`].
///
/// All env reads are skipped entirely if `SYARA_LLM_NO_ENV` is set to `1`
/// or `true`.  See README for guidance on scoping token exposure.
#[cfg(feature = "llm")]
fn resolve_openai_env_defaults() -> (String, String, Option<String>, Option<String>) {
    const FALLBACK_ENDPOINT: &str = "http://localhost:1234/v1/chat/completions";
    const FALLBACK_MODEL: &str = "local-model";

    if env_flag("SYARA_LLM_NO_ENV") {
        return (FALLBACK_ENDPOINT.into(), FALLBACK_MODEL.into(), None, None);
    }

    let endpoint = std::env::var("SYARA_LLM_ENDPOINT")
        .ok()
        .or_else(|| {
            std::env::var("OPENAI_BASE_URL").ok().map(|base| {
                if base.ends_with("/chat/completions") {
                    base
                } else {
                    format!("{}/chat/completions", base.trim_end_matches('/'))
                }
            })
        })
        .unwrap_or_else(|| FALLBACK_ENDPOINT.into());

    let model = std::env::var("SYARA_LLM_MODEL")
        .ok()
        .or_else(|| std::env::var("OPENAI_MODEL").ok())
        .unwrap_or_else(|| FALLBACK_MODEL.into());

    let api_key = std::env::var("SYARA_LLM_API_KEY")
        .ok()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    let reasoning_effort = std::env::var("SYARA_LLM_REASONING_EFFORT").ok();

    (endpoint, model, api_key, reasoning_effort)
}

#[cfg(feature = "llm")]
fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}
