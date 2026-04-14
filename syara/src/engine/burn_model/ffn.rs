#![allow(dead_code)] // consumed by Qwen3/Nemotron models in Phase 3+

//! SwiGLU-style feed-forward network shared by all transformer layers.
//!
//! Implements the gate/up/down projection pattern used by both Qwen3.5 and
//! Nemotron: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Configuration for the feed-forward block.
#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    /// Hidden size of the transformer (input/output dimension).
    pub d_model: usize,
    /// Intermediate (expanded) dimension.
    pub d_intermediate: usize,
}

impl FeedForwardConfig {
    /// Initialize a [`FeedForward`] module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let no_bias = |d_in, d_out| {
            LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .init(device)
        };
        FeedForward {
            gate_proj: no_bias(self.d_model, self.d_intermediate),
            up_proj: no_bias(self.d_model, self.d_intermediate),
            down_proj: no_bias(self.d_intermediate, self.d_model),
        }
    }
}

/// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
///
/// Input/output shape: `[batch, seq_len, d_model]`.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub(crate) gate_proj: Linear<B>,
    pub(crate) up_proj: Linear<B>,
    pub(crate) down_proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Forward pass. Accepts `[batch, seq_len, d_model]`, returns same shape.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn forward_preserves_shape() {
        let device = Default::default();
        let ffn = FeedForwardConfig {
            d_model: 64,
            d_intermediate: 128,
        }
        .init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn silu_gating_produces_nonzero_for_nonzero_input() {
        let device = Default::default();
        let ffn = FeedForwardConfig {
            d_model: 16,
            d_intermediate: 32,
        }
        .init::<B>(&device);

        // Random input should produce non-zero output (with overwhelming probability)
        let x = Tensor::<B, 3>::random([1, 4, 16], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let out = ffn.forward(x);
        let abs_sum: f32 = out.abs().sum().into_scalar().elem();
        assert!(abs_sum > 0.0, "SiLU gating should produce non-zero output");
    }
}
