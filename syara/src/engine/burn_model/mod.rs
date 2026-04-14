//! Burn model building blocks for local LLM inference.
//!
//! Shared components used by both Qwen3.5 and Nemotron model implementations.
//! Uses Burn's built-in `RmsNorm` and `RotaryEncoding` directly — only
//! attention (GQA + QK-norm) and feed-forward (SwiGLU) are custom.

pub mod attention;
pub mod ffn;
