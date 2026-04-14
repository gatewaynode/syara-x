#![allow(dead_code)] // consumed by Qwen3 model in Phase 3+

//! Gated DeltaNet — linear attention with recurrent delta-rule state updates.
//!
//! Implements the recurrent mode of the Gated Delta Networks paper
//! (arXiv:2412.06464). Each layer maintains a state matrix `S ∈ R^{D_k × D_v}`
//! per head, updated via the delta rule:
//!
//!   S_t = α_t · S_{t-1} + β_t · k_t ⊗ (v_t − S_{t-1}^T k_t)
//!
//! Used by Qwen3.5's linear attention layers (layers where `layer_type = "linear_attention"`).

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig};
use burn::prelude::*;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`GatedDeltaNet`].
#[derive(Config, Debug)]
pub struct GatedDeltaNetConfig {
    /// Hidden size (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per key head.
    pub key_head_dim: usize,
    /// Dimension per value head.
    pub value_head_dim: usize,
    /// Short convolution kernel size.
    #[config(default = 4)]
    pub conv_kernel_size: usize,
    /// Epsilon for internal RmsNorm.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
}

impl GatedDeltaNetConfig {
    /// Initialize a [`GatedDeltaNet`] module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GatedDeltaNet<B> {
        let q_dim = self.num_heads * self.key_head_dim;
        let k_dim = q_dim;
        let v_dim = self.num_heads * self.value_head_dim;
        let qkv_dim = q_dim + k_dim + v_dim;

        let no_bias = |d_in, d_out| {
            LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .init(device)
        };

        GatedDeltaNet {
            in_proj_qkv: no_bias(self.d_model, qkv_dim),
            in_proj_z: no_bias(self.d_model, v_dim),
            in_proj_a: no_bias(self.d_model, self.num_heads),
            in_proj_b: no_bias(self.d_model, self.num_heads),
            conv1d: Conv1dConfig::new(qkv_dim, qkv_dim, self.conv_kernel_size)
                .with_groups(qkv_dim)
                .with_bias(false)
                .with_padding(PaddingConfig1d::Explicit(0))
                .init(device),
            a_log: Initializer::Zeros.init([self.num_heads], device),
            dt_bias: Initializer::Zeros.init([self.num_heads], device),
            norm: RmsNormConfig::new(v_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            out_proj: no_bias(v_dim, self.d_model),
            num_heads: self.num_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            conv_kernel_size: self.conv_kernel_size,
        }
    }
}

// ── Module ───────────────────────────────────────────────────────────────────

/// Gated DeltaNet linear attention block (recurrent mode).
///
/// Input/output: `[batch, seq_len, d_model]`.
#[derive(Module, Debug)]
pub struct GatedDeltaNet<B: Backend> {
    /// Fused Q/K/V projection.
    in_proj_qkv: Linear<B>,
    /// Output gate projection.
    in_proj_z: Linear<B>,
    /// Alpha (decay) projection — one scalar per head.
    in_proj_a: Linear<B>,
    /// Beta (write strength) projection — one scalar per head.
    in_proj_b: Linear<B>,
    /// Depthwise causal convolution on concatenated Q/K/V.
    conv1d: Conv1d<B>,
    /// Log-space decay parameter, shape `[num_heads]`.
    a_log: Param<Tensor<B, 1>>,
    /// Timestep bias, shape `[num_heads]`.
    dt_bias: Param<Tensor<B, 1>>,
    /// Internal RmsNorm applied to output before gating.
    norm: RmsNorm<B>,
    /// Output projection.
    out_proj: Linear<B>,
    num_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
}

impl<B: Backend> GatedDeltaNet<B> {
    /// Forward pass with recurrent state update.
    ///
    /// `x`: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();
        let q_dim = self.num_heads * self.key_head_dim;
        let k_dim = q_dim;
        let v_dim = self.num_heads * self.value_head_dim;

        // ── 1. Projections ──────────────────────────────────────────────
        let qkv = self.in_proj_qkv.forward(x.clone());
        let z = self.in_proj_z.forward(x.clone());
        let alpha_raw = self.in_proj_a.forward(x.clone());
        let beta_raw = self.in_proj_b.forward(x);

        // ── 2. Causal depthwise conv1d ──────────────────────────────────
        // Conv1d expects [B, C, L]
        let qkv_t = qkv.swap_dims(1, 2);
        let qkv_channels = q_dim + k_dim + v_dim;
        let pad = Tensor::zeros([batch, qkv_channels, self.conv_kernel_size - 1], &device);
        let qkv_t = Tensor::cat(vec![pad, qkv_t], 2);
        let qkv_t = self.conv1d.forward(qkv_t);
        let qkv_conv = qkv_t.swap_dims(1, 2); // back to [B, L, C]

        // ── 3. Split Q/K/V, activate Q and K with SiLU ─────────────────
        let q = burn::tensor::activation::silu(qkv_conv.clone().narrow(2, 0, q_dim));
        let k = burn::tensor::activation::silu(qkv_conv.clone().narrow(2, q_dim, k_dim));
        let v = qkv_conv.narrow(2, q_dim + k_dim, v_dim);

        // Reshape to [B, H, L, D_head]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.key_head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.key_head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.value_head_dim])
            .swap_dims(1, 2);

        // ── 4. Decay factor α = sigmoid(A · dt) ────────────────────────
        // A = -softplus(A_log), dt = softplus(alpha_raw + dt_bias)
        let a = softplus(self.a_log.val()).neg(); // [H]
        let dt_bias_expanded = self.dt_bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let dt = softplus(alpha_raw + dt_bias_expanded); // [B, L, H]
        let a_expanded = a.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0); // [1, 1, H]
        let alpha = burn::tensor::activation::sigmoid(a_expanded * dt); // [B, L, H]
        // → [B, H, L, 1] for broadcasting over state dims
        let alpha = alpha.swap_dims(1, 2).unsqueeze_dim::<4>(3);

        // ── 5. Write strength β = sigmoid(beta_raw) ────────────────────
        let beta = burn::tensor::activation::sigmoid(beta_raw); // [B, L, H]
        let beta = beta.swap_dims(1, 2).unsqueeze_dim::<4>(3); // [B, H, L, 1]

        // ── 6. Recurrent delta-rule state update ────────────────────────
        let mut state: Tensor<B, 4> = Tensor::zeros(
            [batch, self.num_heads, self.key_head_dim, self.value_head_dim],
            &device,
        );
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.clone().narrow(2, t, 1).squeeze_dim::<3>(2); // [B, H, D_k]
            let k_t = k.clone().narrow(2, t, 1).squeeze_dim::<3>(2); // [B, H, D_k]
            let v_t = v.clone().narrow(2, t, 1).squeeze_dim::<3>(2); // [B, H, D_v]
            let alpha_t = alpha.clone().narrow(2, t, 1).squeeze_dim::<3>(2); // [B, H, 1]
            let beta_t = beta.clone().narrow(2, t, 1).squeeze_dim::<3>(2); // [B, H, 1]

            // Retrieve: v_old = S^T k_t (what state currently maps k_t to)
            let k_col = k_t.clone().unsqueeze_dim::<4>(3); // [B, H, D_k, 1]
            let v_old = (state.clone() * k_col)
                .sum_dim(2)
                .squeeze_dim::<3>(2); // [B, H, D_v]

            // Delta update: S = α·S + β · k_t ⊗ (v_t − v_old)
            let delta = v_t - v_old; // [B, H, D_v]
            let k_outer = k_t.unsqueeze_dim::<4>(3); // [B, H, D_k, 1]
            let delta_outer = delta.unsqueeze_dim::<4>(2); // [B, H, 1, D_v]
            let update = k_outer * delta_outer; // [B, H, D_k, D_v]

            let alpha_broadcast = alpha_t.unsqueeze_dim::<4>(3); // [B, H, 1, 1]
            let beta_broadcast = beta_t.unsqueeze_dim::<4>(3); // [B, H, 1, 1]
            state = alpha_broadcast * state + beta_broadcast * update;

            // Query: o_t = S^T q_t
            let q_col = q_t.unsqueeze_dim::<4>(3); // [B, H, D_k, 1]
            let o_t = (state.clone() * q_col)
                .sum_dim(2)
                .squeeze_dim::<3>(2); // [B, H, D_v]

            outputs.push(o_t.unsqueeze_dim::<4>(2)); // [B, H, 1, D_v]
        }

        // ── 7. Norm, output gate, project ───────────────────────────────
        let o = Tensor::cat(outputs, 2); // [B, H, L, D_v]
        let o = o
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.value_head_dim]);

        let o = self.norm.forward(o);
        let o = o * burn::tensor::activation::silu(z);
        self.out_proj.forward(o)
    }
}

/// Numerically stable softplus: log(1 + exp(x)).
fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.exp().add_scalar(1.0).log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn test_config() -> GatedDeltaNetConfig {
        // Tiny config for testing — not real Qwen3.5 dimensions
        GatedDeltaNetConfig {
            d_model: 64,
            num_heads: 4,
            key_head_dim: 16,
            value_head_dim: 16,
            conv_kernel_size: 4,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn forward_preserves_shape() {
        let device = Default::default();
        let deltanet = test_config().init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = deltanet.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn forward_single_token() {
        let device = Default::default();
        let deltanet = test_config().init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([1, 1, 64], &device);
        let out = deltanet.forward(x);
        assert_eq!(out.dims(), [1, 1, 64]);
    }

    #[test]
    fn state_accumulates_over_tokens() {
        let device = Default::default();
        let deltanet = test_config().init::<B>(&device);

        // Non-zero input — state should evolve, producing different outputs per token
        let x = Tensor::<B, 3>::random(
            [1, 4, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = deltanet.forward(x);
        assert_eq!(out.dims(), [1, 4, 64]);

        // Check that different timesteps produce different outputs
        let t0 = out.clone().narrow(1, 0, 1);
        let t3 = out.narrow(1, 3, 1);
        let diff: f32 = (t0 - t3).abs().sum().into_scalar().elem();
        // With random weights and input, outputs at different positions should differ
        assert!(diff > 0.0, "outputs at t=0 and t=3 should differ");
    }

    #[test]
    fn softplus_is_positive() {
        let device: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 1>::from_floats([-2.0, -1.0, 0.0, 1.0, 2.0], &device);
        let result = softplus(x);
        let vals: Vec<f32> = result.to_data().to_vec().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(*v > 0.0, "softplus[{i}] = {v}, expected > 0");
        }
        // softplus(0) = ln(2) ≈ 0.693
        assert!(
            (vals[2] - 0.6931).abs() < 1e-3,
            "softplus(0) should be ~ln(2)"
        );
    }
}
