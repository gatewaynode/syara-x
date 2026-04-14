#![allow(dead_code)] // consumed by Qwen3/Nemotron models in Phase 3+

//! Grouped-Query Attention with optional QK-norm and partial RoPE.
//!
//! Burn's built-in `MultiHeadAttention` doesn't support GQA or QK-norm, so
//! this is a custom implementation for Qwen3.5's full-attention layers.
//!
//! ## GQA
//! Query heads are grouped — multiple Q heads share a single K/V head. K/V
//! tensors are repeated to match the number of Q heads before the attention
//! dot product.
//!
//! ## Partial RoPE
//! Qwen3.5 uses `partial_rotary_factor = 0.25`: only the first 25% of each
//! head dimension gets rotary encoding. The rest passes through unchanged.

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`FullAttention`].
#[derive(Config, Debug)]
pub struct FullAttentionConfig {
    /// Hidden size (d_model).
    pub d_model: usize,
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA). Must divide `n_heads` evenly.
    pub n_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Maximum sequence length for RoPE precomputation.
    pub max_seq_len: usize,
    /// Whether to apply RmsNorm to Q and K projections.
    #[config(default = false)]
    pub qk_norm: bool,
    /// Fraction of head_dim that gets rotary encoding (1.0 = full RoPE).
    #[config(default = 1.0)]
    pub partial_rotary_factor: f64,
    /// RoPE theta base frequency.
    #[config(default = 10_000.0)]
    pub rope_theta: f32,
    /// Epsilon for RmsNorm (QK-norm and any internal norms).
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
}

impl FullAttentionConfig {
    /// Initialize a [`FullAttention`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FullAttention<B> {
        let no_bias = |d_in, d_out| {
            LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .init(device)
        };

        let q_proj = no_bias(self.d_model, self.n_heads * self.head_dim);
        let k_proj = no_bias(self.d_model, self.n_kv_heads * self.head_dim);
        let v_proj = no_bias(self.d_model, self.n_kv_heads * self.head_dim);
        let o_proj = no_bias(self.n_heads * self.head_dim, self.d_model);

        let q_norm = if self.qk_norm {
            Some(
                RmsNormConfig::new(self.head_dim)
                    .with_epsilon(self.rms_norm_eps)
                    .init(device),
            )
        } else {
            None
        };
        let k_norm = if self.qk_norm {
            Some(
                RmsNormConfig::new(self.head_dim)
                    .with_epsilon(self.rms_norm_eps)
                    .init(device),
            )
        } else {
            None
        };

        let rotary_dim =
            (self.head_dim as f64 * self.partial_rotary_factor).floor() as usize;
        // RotaryEncoding expects the rotary dimension (not full head_dim)
        let rope = RotaryEncodingConfig::new(self.max_seq_len, rotary_dim)
            .with_theta(self.rope_theta)
            .init(device);

        FullAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            rotary_dim,
        }
    }
}

// ── Module ───────────────────────────────────────────────────────────────────

/// Grouped-Query Attention with optional QK-norm and partial RoPE.
///
/// Input: `[batch, seq_len, d_model]` → Output: `[batch, seq_len, d_model]`.
#[derive(Module, Debug)]
pub struct FullAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
    rope: RotaryEncoding<B>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

impl<B: Backend> FullAttention<B> {
    /// Forward pass with causal masking.
    ///
    /// `x`: `[batch, seq_len, d_model]`
    /// Returns: `[batch, seq_len, d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();

        // Project Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim] then transpose to
        // [batch, n_heads, seq_len, head_dim]
        let q = q
            .reshape([batch, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Optional QK-norm (applied per-head before RoPE)
        let q = self.apply_optional_norm(&self.q_norm, q);
        let k = self.apply_optional_norm(&self.k_norm, k);

        // Apply partial RoPE
        let q = self.apply_partial_rope(q);
        let k = self.apply_partial_rope(k);

        // GQA: repeat K/V heads to match Q head count
        let kv_repeat = self.n_heads / self.n_kv_heads;
        let k = if kv_repeat > 1 {
            Self::repeat_kv(k, kv_repeat)
        } else {
            k
        };
        let v = if kv_repeat > 1 {
            Self::repeat_kv(v, kv_repeat)
        } else {
            v
        };

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.transpose()) / scale;

        // Causal mask: positions where j > i get -inf
        let mask = Self::causal_mask(seq_len, &device);
        let scores = scores + mask;

        let attn = burn::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v);

        // Reshape back: [batch, n_heads, seq_len, head_dim] → [batch, seq_len, d_model]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Apply RmsNorm if present. Input/output: `[batch, n_heads, seq_len, head_dim]`.
    fn apply_optional_norm(
        &self,
        norm: &Option<RmsNorm<B>>,
        x: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        match norm {
            Some(n) => n.forward(x),
            None => x,
        }
    }

    /// Apply RoPE to the rotary portion of the tensor, leave the rest untouched.
    ///
    /// Input: `[batch, n_heads, seq_len, head_dim]`
    fn apply_partial_rope(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.rotary_dim == 0 {
            return x;
        }
        if self.rotary_dim == self.head_dim {
            return self.rope.forward(x);
        }
        // Split into rotary and pass-through portions along the last dim
        let rotary_part = x.clone().narrow(3, 0, self.rotary_dim);
        let pass_through = x.narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim);

        let rotary_part = self.rope.forward(rotary_part);
        Tensor::cat(vec![rotary_part, pass_through], 3)
    }

    /// Repeat KV heads to match the number of Q heads.
    ///
    /// `[batch, n_kv_heads, seq_len, head_dim]` → `[batch, n_heads, seq_len, head_dim]`
    fn repeat_kv(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        let [batch, n_kv_heads, seq_len, head_dim] = x.dims();
        // [batch, n_kv_heads, 1, seq_len, head_dim]
        let x = x.unsqueeze_dim::<5>(2);
        // [batch, n_kv_heads, n_rep, seq_len, head_dim]
        let x = x.repeat_dim(2, n_rep);
        // [batch, n_kv_heads * n_rep, seq_len, head_dim]
        x.reshape([batch, n_kv_heads * n_rep, seq_len, head_dim])
    }

    /// Build a causal attention mask: `[1, 1, seq_len, seq_len]`.
    ///
    /// Upper-triangular entries (j > i) are set to -1e9 (large negative),
    /// everything else is 0. Added to attention scores before softmax.
    fn causal_mask(seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        // Row indices [seq_len, 1] and column indices [1, seq_len]
        let rows = Tensor::<B, 1, Int>::arange(0..(seq_len as i64), device)
            .reshape([seq_len, 1])
            .float();
        let cols = Tensor::<B, 1, Int>::arange(0..(seq_len as i64), device)
            .reshape([1, seq_len])
            .float();

        // mask[i][j] = 1.0 where j > i (future positions)
        let future = cols.greater(rows);

        // Fill future positions with large negative value
        let zeros = Tensor::<B, 2>::zeros([seq_len, seq_len], device);
        let mask = zeros.mask_fill(future, -1e9);

        // Broadcast dims: [1, 1, seq_len, seq_len]
        mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn test_config() -> FullAttentionConfig {
        FullAttentionConfig {
            d_model: 64,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            max_seq_len: 128,
            qk_norm: true,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn forward_preserves_shape() {
        let device = Default::default();
        let attn = test_config().init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = attn.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn forward_single_token() {
        let device = Default::default();
        let attn = test_config().init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([1, 1, 64], &device);
        let out = attn.forward(x);
        assert_eq!(out.dims(), [1, 1, 64]);
    }

    #[test]
    fn causal_mask_blocks_future() {
        let device: <B as Backend>::Device = Default::default();
        let mask = FullAttention::<B>::causal_mask(4, &device);
        assert_eq!(mask.dims(), [1, 1, 4, 4]);

        // [1, 1, 4, 4] → [4, 4] by removing the two leading singleton dims
        let data = mask.squeeze_dim::<3>(0).squeeze_dim::<2>(0);
        // Row 0: [0, -1e9, -1e9, -1e9] — token 0 can only attend to itself
        // Row 3: [0, 0, 0, 0]           — token 3 can attend to all
        let vals: Vec<f32> = data.to_data().to_vec().unwrap();
        // Diagonal and below should be 0
        assert!((vals[0]).abs() < 1e-6, "mask[0,0] should be 0");       // i=0, j=0
        assert!(vals[1] < -1e8, "mask[0,1] should be large negative");  // i=0, j=1 (future)
        assert!((vals[12]).abs() < 1e-6, "mask[3,0] should be 0");      // i=3, j=0
        assert!((vals[15]).abs() < 1e-6, "mask[3,3] should be 0");      // i=3, j=3
    }

    #[test]
    fn partial_rope_leaves_passthrough_unchanged() {
        let device = Default::default();
        // partial_rotary_factor = 0.25, head_dim = 16, so rotary_dim = 4
        let attn = test_config().init::<B>(&device);
        assert_eq!(attn.rotary_dim, 4);

        // Create input where pass-through dims (4..16) are all ones
        let x = Tensor::<B, 4>::zeros([1, 4, 2, 16], &device);
        let ones_part = Tensor::<B, 4>::ones([1, 4, 2, 12], &device);
        let x = Tensor::cat(vec![x.narrow(3, 0, 4), ones_part], 3);

        let out = attn.apply_partial_rope(x);
        // Pass-through portion (dims 4..16) should still be ones
        let pass_through = out.narrow(3, 4, 12);
        let vals: Vec<f32> = pass_through.to_data().to_vec().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "pass-through dim {i} should be 1.0, got {v}"
            );
        }
    }

    #[test]
    fn repeat_kv_expands_correctly() {
        let device: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 4>::ones([1, 2, 4, 8], &device);
        let expanded = FullAttention::<B>::repeat_kv(x, 3);
        assert_eq!(expanded.dims(), [1, 6, 4, 8]);
    }

    #[test]
    fn no_qk_norm_works() {
        let device = Default::default();
        let mut cfg = test_config();
        cfg.qk_norm = false;
        let attn = cfg.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [1, 4, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = attn.forward(x);
        assert_eq!(out.dims(), [1, 4, 64]);
    }

    #[test]
    fn full_rope_when_factor_is_one() {
        let device = Default::default();
        let mut cfg = test_config();
        cfg.partial_rotary_factor = 1.0;
        let attn = cfg.init::<B>(&device);
        assert_eq!(attn.rotary_dim, cfg.head_dim);
    }
}
