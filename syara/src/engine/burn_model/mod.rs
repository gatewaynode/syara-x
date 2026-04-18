//! Burn model building blocks for local LLM inference.
//!
//! Shared components used by both Qwen3.5 and Nemotron model implementations.
//! Uses Burn's built-in `RmsNorm` and `RotaryEncoding` directly — only
//! attention (GQA + QK-norm) and feed-forward (SwiGLU) are custom.

use burn::prelude::*;

pub mod attention;
pub mod deltanet;
pub mod ffn;
pub mod generate;
pub mod loader;
pub mod mamba;
pub mod nemotron;
pub mod qwen3;

/// Trait for models that can produce logits from input token IDs.
///
/// Implemented by both `Qwen3TextModel` and (soon) `NemotronModel`,
/// allowing `greedy_generate` to work with any architecture.
pub trait ForwardModel<B: Backend> {
    /// Run a forward pass: `[batch, seq_len]` → `[batch, seq_len, vocab_size]`.
    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3>;
}
