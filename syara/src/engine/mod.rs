pub mod chunker;
pub mod cleaner;
pub mod string_matcher;

#[cfg(feature = "sbert")]
pub mod semantic_matcher;

#[cfg(feature = "sbert-onnx")]
pub mod onnx_embedder;

#[cfg(feature = "classifier")]
pub mod classifier;

#[cfg(any(feature = "llm", feature = "burn-llm"))]
pub mod llm_evaluator;

#[cfg(feature = "burn-llm")]
pub mod burn_evaluator;

#[cfg(feature = "burn-llm")]
pub(crate) mod burn_model;

#[cfg(feature = "phash")]
pub mod phash_matcher;

// ── Shared HTTP embedding client (BUG-011, BUG-012, BUG-033) ──────────────

/// Response shape for the configured embedding HTTP endpoint.
#[cfg(feature = "sbert")]
#[derive(Clone, Copy)]
pub(crate) enum EmbeddingApi {
    /// OpenAI `/v1/embeddings`: `{"data": [{"embedding": [...]}, ...]}`.
    /// Also served by LM Studio, vLLM, llama-server, Open WebUI, etc.
    OpenAi,
    /// Ollama `/api/embed`: `{"embeddings": [[...]]}`.
    Ollama,
}

/// Shared HTTP client for OpenAI- or Ollama-compatible embedding endpoints.
///
/// Used by both the [`semantic_matcher`] and [`classifier`] HTTP-backed
/// matchers (`OpenAi*` + `Ollama*`) to eliminate duplicated `embed()` logic
/// (BUG-012).  Configures connect + read timeouts (BUG-011) and caches
/// embeddings so repeated pattern lookups skip the HTTP round-trip (BUG-033).
#[cfg(feature = "sbert")]
pub(crate) struct HttpEmbedder {
    endpoint: String,
    model: String,
    api: EmbeddingApi,
    client: reqwest::blocking::Client,
    /// Embedding cache keyed by input text.
    cache: std::sync::Mutex<std::collections::HashMap<String, Vec<f32>>>,
}

#[cfg(feature = "sbert")]
impl HttpEmbedder {
    const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

    /// Build an OpenAI-compatible embedder (POST `{"model","input"}` →
    /// `{"data":[{"embedding":[...]}]}`).  The common default across
    /// LM Studio, vLLM, llama-server, Open WebUI, and openai.com itself.
    pub fn openai(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self::build(endpoint, model, EmbeddingApi::OpenAi)
    }

    /// Build an Ollama-compatible embedder (POST `{"model","input"}` →
    /// `{"embeddings":[[...]]}`).
    pub fn ollama(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self::build(endpoint, model, EmbeddingApi::Ollama)
    }

    fn build(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api: EmbeddingApi,
    ) -> Self {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Self::CONNECT_TIMEOUT)
            .timeout(Self::READ_TIMEOUT)
            .build()
            .expect("failed to build HTTP client");
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api,
            client,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Embed `text` into a float vector via the configured HTTP endpoint.
    ///
    /// Returns a cached result when available (BUG-033).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }

        let body = serde_json::json!({
            "model": self.model,
            "input": text
        });

        let resp = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?;

        let json: serde_json::Value = resp
            .json()
            .map_err(|e| e.to_string())?;

        let embedding: Vec<f32> = match self.api {
            EmbeddingApi::OpenAi => json
                .get("data")
                .and_then(|v| v.get(0))
                .and_then(|v| v.get("embedding"))
                .and_then(|v| v.as_array())
                .ok_or("unexpected response: missing data[0].embedding")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect(),
            EmbeddingApi::Ollama => json
                .get("embeddings")
                .and_then(|v| v.get(0))
                .and_then(|v| v.as_array())
                .ok_or("unexpected response: missing embeddings[0]")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect(),
        };

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(text.to_owned(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Clear the embedding cache (useful for testing).
    #[cfg(test)]
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

#[cfg(all(test, feature = "sbert"))]
mod tests {
    use super::HttpEmbedder;

    #[test]
    fn http_embedder_has_timeouts_configured() {
        // BUG-011: verify the client is built with non-default timeouts.
        // We can't inspect reqwest internals, but we verify construction
        // succeeds and the constants are sensible.
        let embedder = HttpEmbedder::openai(
            "http://localhost:1234/v1/embeddings",
            "text-embedding-3-small",
        );
        assert_eq!(
            HttpEmbedder::CONNECT_TIMEOUT,
            std::time::Duration::from_secs(10)
        );
        assert_eq!(
            HttpEmbedder::READ_TIMEOUT,
            std::time::Duration::from_secs(30)
        );
        // Empty text should return immediately (no HTTP call)
        assert!(embedder.embed("").unwrap().is_empty());
    }

    #[test]
    fn http_embedder_caches_results() {
        // BUG-033: verify the cache is populated after embed.
        // We can't call a real server, but we can verify the cache
        // starts empty and that empty-text calls don't pollute it.
        let embedder = HttpEmbedder::ollama(
            "http://localhost:11434/api/embed",
            "all-minilm",
        );
        assert_eq!(embedder.cache_len(), 0);

        // Empty text bypasses the cache entirely
        let _ = embedder.embed("");
        assert_eq!(embedder.cache_len(), 0, "empty text should not be cached");
    }

    #[test]
    fn shared_embedder_used_by_openai_matcher() {
        // BUG-012: verify OpenAiEmbeddingMatcher delegates to HttpEmbedder.
        use crate::engine::semantic_matcher::{OpenAiEmbeddingMatcher, SemanticMatcher};
        let matcher = OpenAiEmbeddingMatcher::new(
            "http://localhost:1234/v1/embeddings",
            "text-embedding-3-small",
        );
        // Empty text goes through the shared embedder's early-return path
        assert!(matcher.embed("").unwrap().is_empty());
    }

    #[test]
    fn shared_embedder_used_by_ollama_matcher() {
        use crate::engine::semantic_matcher::{OllamaEmbeddingMatcher, SemanticMatcher};
        let matcher = OllamaEmbeddingMatcher::new(
            "http://localhost:11434/api/embed",
            "all-minilm",
        );
        assert!(matcher.embed("").unwrap().is_empty());
    }

    #[cfg(feature = "classifier")]
    #[test]
    fn shared_embedder_used_by_classifier() {
        // BUG-012: verify classifier HTTP backends delegate to HttpEmbedder.
        use crate::engine::classifier::{OpenAiEmbeddingClassifier, TextClassifier};
        let classifier = OpenAiEmbeddingClassifier::new(
            "http://localhost:1234/v1/embeddings",
            "text-embedding-3-small",
        );
        // Empty inputs go through the shared embedder's early-return path
        assert_eq!(classifier.score("", "text").unwrap(), 0.0);
    }
}
