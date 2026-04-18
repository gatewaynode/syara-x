//! ML classifier-based matching via HTTP or local ONNX embeddings.
//!
//! [`TextClassifier`] abstracts over classification backends.  Three built-in
//! implementations:
//! - [`OpenAiEmbeddingClassifier`] — OpenAI-compatible `/v1/embeddings`,
//!   registry default for `"tuned-sbert"` (works with LM Studio, vLLM,
//!   llama-server, openai.com).
//! - [`OllamaEmbeddingClassifier`] — Ollama `/api/embed` variant, preserved.
//! - [`OnnxEmbeddingClassifier`] (`classifier-onnx` feature) — wraps the local
//!   [`OnnxEmbeddingMatcher`](crate::engine::onnx_embedder::OnnxEmbeddingMatcher)
//!   and runs MiniLM-L6-v2 (or any compatible sentence-transformer ONNX export)
//!   in-process. Recommended for offline / deterministic deployments.
//!
//! All three compute cosine similarity between the rule pattern and each input
//! chunk; threshold checks happen in the default
//! [`classify_chunks`](TextClassifier::classify_chunks).
//!
//! Because `classifier` implies `sbert`, [`cosine_similarity`] is borrowed
//! from the sibling module rather than duplicated.

use crate::engine::semantic_matcher::cosine_similarity;
use crate::error::SyaraError;
use crate::models::{ClassifierRule, MatchDetail};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Text classifier.
///
/// Implementations embed text and return a confidence score relative to a
/// rule pattern. The default [`classify_chunks`] applies the rule's threshold.
pub trait TextClassifier: Send + Sync {
    /// Compute a confidence score (0.0–1.0) for `input_text` against
    /// `rule_pattern`. Higher scores indicate stronger semantic match.
    fn score(&self, rule_pattern: &str, input_text: &str) -> Result<f64, SyaraError>;

    /// Apply classification to pre-chunked text.
    ///
    /// Scores each chunk against `rule.pattern`; returns [`MatchDetail`] for
    /// every chunk whose score is `>= rule.threshold`.
    fn classify_chunks(
        &self,
        rule: &ClassifierRule,
        chunks: &[String],
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        if chunks.is_empty() || rule.pattern.is_empty() {
            return Ok(vec![]);
        }

        let mut matches = Vec::new();
        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }
            let confidence = self.score(&rule.pattern, chunk)?;
            if confidence >= rule.threshold {
                let mut detail =
                    MatchDetail::new(rule.identifier.clone(), chunk.clone())
                        .with_score(confidence);
                detail.explanation = format!("Classifier confidence: {confidence:.3}");
                matches.push(detail);
            }
        }
        Ok(matches)
    }
}

// ── HTTP implementations ──────────────────────────────────────────────────────

fn http_score(
    embedder: &super::HttpEmbedder,
    rule_pattern: &str,
    input_text: &str,
) -> Result<f64, SyaraError> {
    if rule_pattern.is_empty() || input_text.is_empty() {
        return Ok(0.0);
    }
    let pattern_emb = embedder
        .embed(rule_pattern)
        .map_err(SyaraError::ClassifierError)?;
    let input_emb = embedder
        .embed(input_text)
        .map_err(SyaraError::ClassifierError)?;
    Ok(f64::from(cosine_similarity(&pattern_emb, &input_emb)))
}

/// Classifier backed by an OpenAI-compatible `/v1/embeddings` HTTP endpoint.
///
/// Registry default for the `"tuned-sbert"` name.  Swap in the Ollama variant
/// via [`crate::compiled_rules::CompiledRules::register_classifier`] if your
/// embedding server speaks the Ollama wire format instead.
pub struct OpenAiEmbeddingClassifier {
    embedder: super::HttpEmbedder,
}

impl OpenAiEmbeddingClassifier {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            embedder: super::HttpEmbedder::openai(endpoint, model),
        }
    }
}

impl TextClassifier for OpenAiEmbeddingClassifier {
    fn score(&self, rule_pattern: &str, input_text: &str) -> Result<f64, SyaraError> {
        http_score(&self.embedder, rule_pattern, input_text)
    }
}

/// Classifier backed by an Ollama-compatible `/api/embed` HTTP endpoint.
pub struct OllamaEmbeddingClassifier {
    embedder: super::HttpEmbedder,
}

impl OllamaEmbeddingClassifier {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            embedder: super::HttpEmbedder::ollama(endpoint, model),
        }
    }
}

impl TextClassifier for OllamaEmbeddingClassifier {
    fn score(&self, rule_pattern: &str, input_text: &str) -> Result<f64, SyaraError> {
        http_score(&self.embedder, rule_pattern, input_text)
    }
}

// ── Local ONNX implementation ─────────────────────────────────────────────────

/// Classifier backed by a local ONNX embedding model (MiniLM-L6-v2 by default).
///
/// Composes [`OnnxEmbeddingMatcher`](crate::engine::onnx_embedder::OnnxEmbeddingMatcher)
/// rather than duplicating its session/tokenizer plumbing — both compute
/// pooled, L2-normalized embeddings, so the classifier just needs to embed
/// `(rule_pattern, input_text)` and take the cosine.
///
/// Recommended over the HTTP backends when:
/// - You need deterministic, network-free scoring (CI, air-gapped hosts).
/// - You want to avoid the per-call HTTP round-trip overhead.
/// - You already ship the ONNX model as part of the deployment.
///
/// Requires the `classifier-onnx` feature and the system `libonnxruntime`
/// shared library at runtime — see the README's "System dependencies"
/// section.
///
/// ```no_run
/// # #[cfg(feature = "classifier-onnx")]
/// # fn main() -> Result<(), syara_x::error::SyaraError> {
/// use syara_x::compile_str;
/// use syara_x::engine::classifier::OnnxEmbeddingClassifier;
///
/// let mut rules = compile_str(r#"
/// rule local_classifier {
///   classifier:
///     $c = "phishing language" threshold=0.6 classifier="tuned-sbert"
///   condition:
///     $c
/// }
/// "#)?;
/// let cls = OnnxEmbeddingClassifier::from_dir("../models/all-MiniLM-L6-v2")?;
/// rules.register_classifier("tuned-sbert", Box::new(cls));
/// # Ok(()) }
/// # #[cfg(not(feature = "classifier-onnx"))] fn main() {}
/// ```
#[cfg(feature = "classifier-onnx")]
pub struct OnnxEmbeddingClassifier {
    embedder: crate::engine::onnx_embedder::OnnxEmbeddingMatcher,
}

#[cfg(feature = "classifier-onnx")]
impl OnnxEmbeddingClassifier {
    /// Load `model.onnx` + `tokenizer.json` from a directory.
    pub fn from_dir(model_dir: impl AsRef<std::path::Path>) -> Result<Self, SyaraError> {
        Ok(Self {
            embedder: crate::engine::onnx_embedder::OnnxEmbeddingMatcher::from_dir(
                model_dir,
            )?,
        })
    }

    /// Load from explicit model + tokenizer paths.
    pub fn from_paths(
        model: impl AsRef<std::path::Path>,
        tokenizer: impl AsRef<std::path::Path>,
    ) -> Result<Self, SyaraError> {
        Ok(Self {
            embedder: crate::engine::onnx_embedder::OnnxEmbeddingMatcher::from_paths(
                model, tokenizer,
            )?,
        })
    }
}

#[cfg(feature = "classifier-onnx")]
impl TextClassifier for OnnxEmbeddingClassifier {
    fn score(&self, rule_pattern: &str, input_text: &str) -> Result<f64, SyaraError> {
        if rule_pattern.is_empty() || input_text.is_empty() {
            return Ok(0.0);
        }
        use crate::engine::semantic_matcher::SemanticMatcher;
        let pattern_emb = self.embedder.embed(rule_pattern)?;
        let input_emb = self.embedder.embed(input_text)?;
        Ok(f64::from(cosine_similarity(&pattern_emb, &input_emb)))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ClassifierRule;

    /// Test double: returns fixed cosine-ready vectors for known texts.
    struct FixedClassifier(Vec<(String, Vec<f32>)>);

    impl TextClassifier for FixedClassifier {
        fn score(&self, rule_pattern: &str, input_text: &str) -> Result<f64, SyaraError> {
            let get_emb = |text: &str| -> Vec<f32> {
                for (key, vec) in &self.0 {
                    if key == text {
                        return vec.clone();
                    }
                }
                vec![0.0; 3]
            };
            let a = get_emb(rule_pattern);
            let b = get_emb(input_text);
            Ok(f64::from(cosine_similarity(&a, &b)))
        }
    }

    #[test]
    fn classify_chunks_above_threshold() {
        // matching_chunk is identical direction to pattern → score 1.0 ≥ 0.8
        // unrelated_chunk is orthogonal → score 0.0 < 0.8
        let pattern = "malicious content pattern";
        let matching_chunk = "similar malicious content";
        let unrelated_chunk = "completely different text";

        let classifier = FixedClassifier(vec![
            (pattern.into(), vec![1.0, 0.0, 0.0]),
            (matching_chunk.into(), vec![1.0, 0.0, 0.0]),
            (unrelated_chunk.into(), vec![0.0, 1.0, 0.0]),
        ]);

        let rule = ClassifierRule {
            identifier: "$cls".into(),
            pattern: pattern.into(),
            threshold: 0.8,
            ..Default::default()
        };

        let results = classifier
            .classify_chunks(&rule, &[matching_chunk.to_string(), unrelated_chunk.to_string()])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_text, matching_chunk);
        assert_eq!(results[0].identifier, "$cls");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!(results[0].explanation.contains("Classifier confidence:"));
    }

    #[test]
    fn classify_chunks_empty_input() {
        let classifier = FixedClassifier(vec![]);
        let rule = ClassifierRule::default();
        assert!(classifier.classify_chunks(&rule, &[]).unwrap().is_empty());
    }

    #[test]
    fn classify_chunks_empty_pattern() {
        let classifier = FixedClassifier(vec![]);
        let rule = ClassifierRule {
            pattern: String::new(),
            ..Default::default()
        };
        assert!(classifier
            .classify_chunks(&rule, &["some text".to_string()])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn classify_chunks_no_match_below_threshold() {
        let classifier = FixedClassifier(vec![
            ("pat".into(), vec![1.0, 0.0, 0.0]),
            ("chunk".into(), vec![0.0, 1.0, 0.0]), // orthogonal → score 0.0
        ]);
        let rule = ClassifierRule {
            identifier: "$c".into(),
            pattern: "pat".into(),
            threshold: 0.9,
            ..Default::default()
        };
        let results = classifier
            .classify_chunks(&rule, &["chunk".to_string()])
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn score_empty_inputs_return_zero() {
        let classifier = FixedClassifier(vec![]);
        // Both empty: score should be 0.0 (early exit)
        assert_eq!(
            classifier.score("", "some text").unwrap(),
            0.0
        );
        assert_eq!(
            classifier.score("pattern", "").unwrap(),
            0.0
        );
    }
}
