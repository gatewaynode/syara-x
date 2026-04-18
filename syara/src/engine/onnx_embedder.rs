//! Local ONNX Runtime [`SemanticMatcher`] backend.
//!
//! Runs sentence-transformer models (MiniLM-L6-v2 by default) in-process via
//! [`ort`] with the `load-dynamic` feature, so the system `libonnxruntime`
//! (≥1.17) must be discoverable at runtime. Weights and tokenizer live
//! outside the crate under the `../models/` convention used by Phase 5.
//!
//! ```no_run
//! use syara_x::compile_str;
//! use syara_x::engine::onnx_embedder::OnnxEmbeddingMatcher;
//!
//! let mut rules = compile_str(r#"
//! rule local_sbert {
//!   similarity:
//!     $s = "ignore previous instructions" threshold=0.7 matcher="sbert"
//!   condition:
//!     $s
//! }
//! "#).expect("compile");
//! let matcher = OnnxEmbeddingMatcher::from_dir("../models/all-MiniLM-L6-v2")
//!     .expect("load MiniLM");
//! rules.register_semantic_matcher("sbert", Box::new(matcher));
//! ```
//!
//! Expected directory layout:
//!
//! ```text
//! ../models/all-MiniLM-L6-v2/
//!   model.onnx         # ONNX export of sentence-transformers/all-MiniLM-L6-v2
//!   tokenizer.json     # WordPiece/BERT tokenizer
//! ```

use std::path::Path;
use std::sync::Mutex;

use std::borrow::Cow;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::semantic_matcher::SemanticMatcher;
use crate::error::SyaraError;

const DEFAULT_MAX_LENGTH: usize = 256;
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
const TOKEN_TYPE_IDS: &str = "token_type_ids";
const LAST_HIDDEN_STATE: &str = "last_hidden_state";

/// Local ONNX embedding matcher.
///
/// Holds the ONNX session behind a [`Mutex`] because [`Session::run`] takes
/// `&mut self` but [`SemanticMatcher`] requires `&self`.
pub struct OnnxEmbeddingMatcher {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    needs_token_type_ids: bool,
}

impl OnnxEmbeddingMatcher {
    /// Load `model.onnx` + `tokenizer.json` from a directory.
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, SyaraError> {
        let dir = model_dir.as_ref();
        Self::from_paths(dir.join("model.onnx"), dir.join("tokenizer.json"))
    }

    /// Load from explicit model + tokenizer paths.
    pub fn from_paths(
        model: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
    ) -> Result<Self, SyaraError> {
        let model_path = model.as_ref();
        let tokenizer_path = tokenizer.as_ref();

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            SyaraError::SemanticError(format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let session = Session::builder()
            .map_err(|e| {
                SyaraError::SemanticError(format!(
                    "failed to build ONNX session: {e}"
                ))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                SyaraError::SemanticError(format!(
                    "failed to load ONNX model from {}: {e}",
                    model_path.display()
                ))
            })?;

        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|outlet| outlet.name() == TOKEN_TYPE_IDS);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: DEFAULT_MAX_LENGTH,
            needs_token_type_ids,
        })
    }

    /// Override the tokenizer truncation limit (default 256).
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

impl SemanticMatcher for OnnxEmbeddingMatcher {
    fn embed(&self, text: &str) -> Result<Vec<f32>, SyaraError> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| SyaraError::SemanticError(format!("tokenize: {e}")))?;

        let ids_src = encoding.get_ids();
        let mask_src = encoding.get_attention_mask();
        let types_src = encoding.get_type_ids();

        let seq_len = ids_src.len().min(self.max_length);
        if seq_len == 0 {
            return Ok(vec![]);
        }

        let ids: Vec<i64> = ids_src[..seq_len].iter().map(|&v| v as i64).collect();
        let mask: Vec<i64> =
            mask_src[..seq_len].iter().map(|&v| v as i64).collect();

        let shape = [1_usize, seq_len];
        let ids_tensor = Tensor::from_array((shape, ids)).map_err(|e| {
            SyaraError::SemanticError(format!("input_ids tensor: {e}"))
        })?;
        let mask_tensor =
            Tensor::from_array((shape, mask.clone())).map_err(|e| {
                SyaraError::SemanticError(format!("attention_mask tensor: {e}"))
            })?;

        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            (Cow::Borrowed(INPUT_IDS), SessionInputValue::from(ids_tensor)),
            (
                Cow::Borrowed(ATTENTION_MASK),
                SessionInputValue::from(mask_tensor),
            ),
        ];

        if self.needs_token_type_ids {
            let types: Vec<i64> =
                types_src[..seq_len].iter().map(|&v| v as i64).collect();
            let types_tensor =
                Tensor::from_array((shape, types)).map_err(|e| {
                    SyaraError::SemanticError(format!(
                        "token_type_ids tensor: {e}"
                    ))
                })?;
            inputs.push((
                Cow::Borrowed(TOKEN_TYPE_IDS),
                SessionInputValue::from(types_tensor),
            ));
        }

        let mut session = self.session.lock().map_err(|_| {
            SyaraError::SemanticError("ONNX session mutex poisoned".into())
        })?;
        let outputs = session.run(inputs).map_err(|e| {
            SyaraError::SemanticError(format!("ONNX run failed: {e}"))
        })?;

        // Prefer the named `last_hidden_state`; fall back to the first output.
        let output_value = outputs
            .get(LAST_HIDDEN_STATE)
            .or_else(|| outputs.get(outputs.iter().next()?.0))
            .ok_or_else(|| {
                SyaraError::SemanticError("ONNX model produced no outputs".into())
            })?;

        let hidden = output_value.try_extract_array::<f32>().map_err(|e| {
            SyaraError::SemanticError(format!(
                "failed to extract last_hidden_state: {e}"
            ))
        })?;

        let shape = hidden.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != seq_len {
            return Err(SyaraError::SemanticError(format!(
                "unexpected output shape {:?}, expected [1, {seq_len}, H]",
                shape
            )));
        }
        let hidden_dim = shape[2];

        // Mean pool across sequence axis, weighted by attention mask.
        let mut pooled = vec![0.0_f32; hidden_dim];
        let mut mask_sum = 0.0_f32;
        for t in 0..seq_len {
            let m = mask[t] as f32;
            if m == 0.0 {
                continue;
            }
            mask_sum += m;
            for (h, out) in pooled.iter_mut().enumerate().take(hidden_dim) {
                *out += hidden[[0, t, h]] * m;
            }
        }
        if mask_sum == 0.0 {
            return Ok(vec![0.0; hidden_dim]);
        }
        for v in pooled.iter_mut() {
            *v /= mask_sum;
        }

        // L2 normalize.
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in pooled.iter_mut() {
                *v /= norm;
            }
        }

        Ok(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_load_missing_file() {
        let result = OnnxEmbeddingMatcher::from_paths(
            "/nonexistent/model.onnx",
            "/nonexistent/tokenizer.json",
        );
        match result {
            Ok(_) => panic!("loading missing tokenizer must fail"),
            Err(SyaraError::SemanticError(msg)) => {
                assert!(
                    msg.contains("tokenizer"),
                    "expected tokenizer error, got: {msg}"
                );
            }
            Err(other) => panic!("expected SemanticError, got {other:?}"),
        }
    }
}
