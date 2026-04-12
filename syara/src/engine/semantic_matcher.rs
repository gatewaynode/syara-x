//! Semantic similarity matching via HTTP embedding endpoints.
//!
//! The [`SemanticMatcher`] trait abstracts over embedding providers.  The
//! built-in [`HttpEmbeddingMatcher`] calls any Ollama-compatible `/api/embed`
//! endpoint (POST `{"model": "…", "input": "…"}` → `{"embeddings": [[…]]}`).
//!
//! Cosine similarity is computed in pure Rust; no ML crates are required at
//! runtime — the heavy lifting is delegated to the embedding server.

use crate::error::SyaraError;
use crate::models::{MatchDetail, SimilarityRule};

// ── Trait ────────────────────────────────────────────────────────────────────

/// Semantic similarity matcher.
///
/// Implementations embed text into a vector space and apply cosine similarity.
/// The default [`match_chunks`] implementation is provided in terms of
/// [`embed`], so custom matchers only need to implement embedding.
pub trait SemanticMatcher: Send + Sync {
    /// Embed `text` into a float vector.
    ///
    /// Empty text should return an empty slice (cosine similarity treats
    /// zero-length vectors as having zero similarity with everything).
    fn embed(&self, text: &str) -> Result<Vec<f32>, SyaraError>;

    /// Match a rule against pre-chunked text.
    ///
    /// Embeds `rule.pattern` once, then compares against each chunk.
    /// Returns [`MatchDetail`] for every chunk whose cosine similarity is
    /// `>= rule.threshold`.  Position fields are `-1` (chunk-based matching
    /// does not track byte offsets — same as the Python reference).
    fn match_chunks(
        &self,
        rule: &SimilarityRule,
        chunks: &[String],
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        if chunks.is_empty() || rule.pattern.is_empty() {
            return Ok(vec![]);
        }

        let pattern_emb = self.embed(&rule.pattern)?;
        let mut matches = Vec::new();

        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }
            let chunk_emb = self.embed(chunk)?;
            let similarity = cosine_similarity(&pattern_emb, &chunk_emb);

            if f64::from(similarity) >= rule.threshold {
                let mut detail =
                    MatchDetail::new(rule.identifier.clone(), chunk.clone())
                        .with_score(f64::from(similarity));
                detail.explanation =
                    format!("Semantic similarity: {similarity:.3}");
                matches.push(detail);
            }
        }

        Ok(matches)
    }
}

// ── Cosine similarity ────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` when either vector is empty, zero-length, or has mismatched
/// dimensions.  Result is clamped to `[-1.0, 1.0]`.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.len() != a.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ── HTTP implementation ───────────────────────────────────────────────────────

/// Calls an Ollama-compatible `/api/embed` HTTP endpoint.
///
/// Delegates to the shared [`super::HttpEmbedder`] which provides timeouts
/// (BUG-011) and embedding caching (BUG-033).
///
/// The `endpoint` and `model` can be changed at registration time via
/// [`HttpEmbeddingMatcher::new`].  Default registration uses
/// `http://localhost:11434/api/embed` with model `all-minilm`.
pub struct HttpEmbeddingMatcher {
    embedder: super::HttpEmbedder,
}

impl HttpEmbeddingMatcher {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            embedder: super::HttpEmbedder::new(endpoint, model),
        }
    }
}

impl SemanticMatcher for HttpEmbeddingMatcher {
    fn embed(&self, text: &str) -> Result<Vec<f32>, SyaraError> {
        self.embedder
            .embed(text)
            .map_err(SyaraError::SemanticError)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SimilarityRule;

    // A test-only matcher backed by fixed per-text embeddings.
    struct FixedMatcher(Vec<(String, Vec<f32>)>);

    impl SemanticMatcher for FixedMatcher {
        fn embed(&self, text: &str) -> Result<Vec<f32>, SyaraError> {
            for (key, vec) in &self.0 {
                if key == text {
                    return Ok(vec.clone());
                }
            }
            Ok(vec![0.0; 3])
        }
    }

    #[test]
    fn cosine_same_vector() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should give 1.0");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_mismatched_lengths() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn match_chunks_above_threshold() {
        // chunk0 is identical to pattern → sim 1.0 ≥ 0.8 threshold → match
        // chunk1 is orthogonal                → sim 0.0  < 0.8         → no match
        let pattern = "pattern text";
        let chunk0 = "matching chunk";
        let chunk1 = "unrelated chunk";

        let matcher = FixedMatcher(vec![
            (pattern.into(), vec![1.0, 0.0, 0.0]),
            (chunk0.into(),  vec![1.0, 0.0, 0.0]),  // same direction → sim 1.0
            (chunk1.into(),  vec![0.0, 1.0, 0.0]),  // orthogonal      → sim 0.0
        ]);

        let rule = SimilarityRule {
            identifier: "$sem".into(),
            pattern: pattern.into(),
            threshold: 0.8,
            ..Default::default()
        };

        let chunks = vec![chunk0.to_string(), chunk1.to_string()];
        let results = matcher.match_chunks(&rule, &chunks).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_text, chunk0);
        assert_eq!(results[0].identifier, "$sem");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!(results[0].explanation.contains("Semantic similarity:"));
    }

    #[test]
    fn match_chunks_empty_input() {
        let matcher = FixedMatcher(vec![]);
        let rule = SimilarityRule::default();
        assert!(matcher.match_chunks(&rule, &[]).unwrap().is_empty());
    }

    #[test]
    fn match_chunks_no_match_below_threshold() {
        let matcher = FixedMatcher(vec![
            ("pat".into(), vec![1.0, 0.0, 0.0]),
            ("chunk".into(), vec![0.0, 1.0, 0.0]), // sim 0.0
        ]);
        let rule = SimilarityRule {
            pattern: "pat".into(),
            threshold: 0.9,
            ..Default::default()
        };
        let results = matcher
            .match_chunks(&rule, &["chunk".to_string()])
            .unwrap();
        assert!(results.is_empty());
    }
}
