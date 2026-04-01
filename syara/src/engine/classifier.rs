//! ML classifier-based matching via HTTP embedding endpoints.
//!
//! [`TextClassifier`] abstracts over classification backends. The built-in
//! [`HttpEmbeddingClassifier`] reuses the same Ollama-compatible `/api/embed`
//! endpoint as the `sbert` feature, computing cosine similarity between the
//! rule pattern and each input chunk.
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

// ── HTTP implementation ───────────────────────────────────────────────────────

/// Classifier backed by an Ollama-compatible `/api/embed` HTTP endpoint.
///
/// Embeds both the rule pattern and each input chunk, then computes cosine
/// similarity as the confidence score.  Default registration uses
/// `http://localhost:11434/api/embed` with model `all-minilm` (same endpoint
/// as the built-in `sbert` semantic matcher).
pub struct HttpEmbeddingClassifier {
    endpoint: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl HttpEmbeddingClassifier {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, SyaraError> {
        if text.is_empty() {
            return Ok(vec![]);
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
            .map_err(|e| SyaraError::ClassifierError(e.to_string()))?;

        let json: serde_json::Value = resp
            .json()
            .map_err(|e| SyaraError::ClassifierError(e.to_string()))?;

        let embedding = json
            .get("embeddings")
            .and_then(|v| v.get(0))
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SyaraError::ClassifierError(
                    "unexpected response: missing embeddings[0]".into(),
                )
            })?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
    }
}

impl TextClassifier for HttpEmbeddingClassifier {
    fn score(&self, rule_pattern: &str, input_text: &str) -> Result<f64, SyaraError> {
        if rule_pattern.is_empty() || input_text.is_empty() {
            return Ok(0.0);
        }
        let pattern_emb = self.embed(rule_pattern)?;
        let input_emb = self.embed(input_text)?;
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
