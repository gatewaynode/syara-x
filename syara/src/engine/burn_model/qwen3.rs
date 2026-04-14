#![allow(dead_code)] // consumed by BurnEvaluator in Phase 4

//! Qwen3.5 text decoder model assembled from shared building blocks.
//!
//! Architecture: 24-layer hybrid DeltaNet/attention transformer with a 3:1
//! ratio of linear attention to full attention layers (`full_attention_interval = 4`).
//!
//! Layer pattern: `[lin, lin, lin, full, lin, lin, lin, full, ...]`

use burn::nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

use super::attention::{FullAttention, FullAttentionConfig};
use super::deltanet::{GatedDeltaNet, GatedDeltaNetConfig};
use super::ffn::{FeedForward, FeedForwardConfig};

// ── Configuration ────────────────────────────────────────────────────────────

/// Qwen3.5 text model configuration, mirroring `text_config` in config.json.
#[derive(Config, Debug)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    /// Full attention heads.
    pub num_attention_heads: usize,
    /// Full attention KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Dimension per full-attention head.
    pub head_dim: usize,
    /// Number of key heads for linear (DeltaNet) layers.
    pub linear_num_key_heads: usize,
    /// Number of value heads for linear layers.
    pub linear_num_value_heads: usize,
    /// Key head dimension for linear layers.
    pub linear_key_head_dim: usize,
    /// Value head dimension for linear layers.
    pub linear_value_head_dim: usize,
    /// Short conv kernel size for DeltaNet.
    #[config(default = 4)]
    pub linear_conv_kernel_dim: usize,
    /// Every N-th layer uses full attention (rest use DeltaNet).
    #[config(default = 4)]
    pub full_attention_interval: usize,
    /// Maximum sequence length for RoPE.
    #[config(default = 4096)]
    pub max_position_embeddings: usize,
    /// RoPE theta base frequency.
    #[config(default = 10_000_000.0)]
    pub rope_theta: f32,
    /// Fraction of head_dim that gets RoPE.
    #[config(default = 0.25)]
    pub partial_rotary_factor: f64,
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
    /// Whether lm_head shares embed_tokens weights.
    #[config(default = true)]
    pub tie_word_embeddings: bool,
    /// EOS token id for generation stopping.
    #[config(default = 248044)]
    pub eos_token_id: usize,
}

impl Qwen3Config {
    /// Initialize a [`Qwen3TextModel`] from this config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3TextModel<B> {
        let embed_tokens = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);

        let layers: Vec<TransformerBlock<B>> = (0..self.num_hidden_layers)
            .map(|i| self.init_block(i, device))
            .collect();

        let final_norm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);

        Qwen3TextModel {
            embed_tokens,
            layers,
            final_norm,
            tie_word_embeddings: self.tie_word_embeddings,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
        }
    }

    /// Build a single transformer block — dispatches to DeltaNet or FullAttention.
    fn init_block<B: Backend>(&self, layer_idx: usize, device: &B::Device) -> TransformerBlock<B> {
        let is_full_attn = (layer_idx + 1).is_multiple_of(self.full_attention_interval);

        let hybrid = if is_full_attn {
            HybridBlock::Full(
                FullAttentionConfig {
                    d_model: self.hidden_size,
                    n_heads: self.num_attention_heads,
                    n_kv_heads: self.num_key_value_heads,
                    head_dim: self.head_dim,
                    max_seq_len: self.max_position_embeddings,
                    qk_norm: true,
                    partial_rotary_factor: self.partial_rotary_factor,
                    rope_theta: self.rope_theta,
                    rms_norm_eps: self.rms_norm_eps,
                }
                .init(device),
            )
        } else {
            HybridBlock::Linear(
                GatedDeltaNetConfig {
                    d_model: self.hidden_size,
                    num_heads: self.linear_num_key_heads,
                    key_head_dim: self.linear_key_head_dim,
                    value_head_dim: self.linear_value_head_dim,
                    conv_kernel_size: self.linear_conv_kernel_dim,
                    rms_norm_eps: self.rms_norm_eps,
                }
                .init(device),
            )
        };

        let norm = |size| {
            RmsNormConfig::new(size)
                .with_epsilon(self.rms_norm_eps)
                .init(device)
        };

        let ffn = FeedForwardConfig {
            d_model: self.hidden_size,
            d_intermediate: self.intermediate_size,
        }
        .init(device);

        TransformerBlock {
            input_layernorm: norm(self.hidden_size),
            hybrid,
            post_attention_layernorm: norm(self.hidden_size),
            mlp: ffn,
        }
    }
}

// ── Hybrid Block ─────────────────────────────────────────────────────────────

/// Either a DeltaNet (linear attention) or FullAttention block.
///
/// Variants are large (1400+ bytes) but few are constructed — one per layer,
/// stored in a `Vec`. Boxing isn't compatible with Burn's Module derive.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum HybridBlock<B: Backend> {
    Linear(GatedDeltaNet<B>),
    Full(FullAttention<B>),
}

impl<B: Backend> HybridBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            HybridBlock::Linear(deltanet) => deltanet.forward(x),
            HybridBlock::Full(attn) => attn.forward(x),
        }
    }
}

// ── Transformer Block ────────────────────────────────────────────────────────

/// Pre-norm transformer block: LN → Attention/DeltaNet → residual → LN → FFN → residual.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub(crate) input_layernorm: RmsNorm<B>,
    pub(crate) hybrid: HybridBlock<B>,
    pub(crate) post_attention_layernorm: RmsNorm<B>,
    pub(crate) mlp: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm + attention/deltanet + residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.hybrid.forward(h);
        let h = residual + h;

        // Pre-norm + FFN + residual
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(h);
        let h = self.mlp.forward(h);
        residual + h
    }
}

// ── Full Model ───────────────────────────────────────────────────────────────

/// Qwen3.5 text decoder: embedding → N transformer blocks → norm → logits.
#[derive(Module, Debug)]
pub struct Qwen3TextModel<B: Backend> {
    pub(crate) embed_tokens: Embedding<B>,
    pub(crate) layers: Vec<TransformerBlock<B>>,
    pub(crate) final_norm: RmsNorm<B>,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) hidden_size: usize,
    pub(crate) vocab_size: usize,
}

impl<B: Backend> Qwen3TextModel<B> {
    /// Forward pass: input token ids → logits over vocabulary.
    ///
    /// `input_ids`: `[batch, seq_len]` (Int tensor)
    /// Returns: `[batch, seq_len, vocab_size]`
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut h = self.embed_tokens.forward(input_ids);

        for layer in &self.layers {
            h = layer.forward(h);
        }

        let h = self.final_norm.forward(h);

        // lm_head: tied to embed_tokens weight (transposed multiply)
        // embed_tokens.weight: [vocab_size, hidden_size]
        // h: [batch, seq_len, hidden_size]
        // logits = h @ weight^T → [batch, seq_len, vocab_size]
        let weight = self.embed_tokens.weight.val(); // [vocab_size, hidden_size]
        let weight = weight.unsqueeze_dim::<3>(0); // [1, vocab_size, hidden_size]
        let weight = weight.transpose(); // [1, hidden_size, vocab_size]
        h.matmul(weight) // broadcasts: [batch, seq_len, vocab_size]
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<B: Backend> super::ForwardModel<B> for Qwen3TextModel<B> {
    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    /// Tiny config for fast tests — 2 layers (1 linear + 1 full), small dims.
    fn tiny_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            eos_token_id: 0,
        }
    }

    #[test]
    fn forward_produces_logits() {
        let device = Default::default();
        let model = tiny_config().init::<B>(&device);

        let input_ids = Tensor::<B, 2, Int>::zeros([1, 4], &device);
        let logits = model.forward(input_ids);
        assert_eq!(logits.dims(), [1, 4, 256]);
    }

    #[test]
    fn hybrid_dispatch_correct() {
        let cfg = tiny_config();
        // full_attention_interval = 2, so layer 0 → Linear, layer 1 → Full
        // (layer_idx + 1) % interval == 0 ⟹ layer 1 is full
        let device = Default::default();
        let model = cfg.init::<B>(&device);

        assert_eq!(model.num_layers(), 2);
        assert!(
            matches!(model.layers[0].hybrid, HybridBlock::Linear(_)),
            "layer 0 should be DeltaNet"
        );
        assert!(
            matches!(model.layers[1].hybrid, HybridBlock::Full(_)),
            "layer 1 should be FullAttention"
        );
    }

    #[test]
    fn forward_single_token() {
        let device = Default::default();
        let model = tiny_config().init::<B>(&device);

        let input_ids = Tensor::<B, 2, Int>::zeros([1, 1], &device);
        let logits = model.forward(input_ids);
        assert_eq!(logits.dims(), [1, 1, 256]);
    }

    #[test]
    fn forward_batch() {
        let device = Default::default();
        let model = tiny_config().init::<B>(&device);

        let input_ids = Tensor::<B, 2, Int>::zeros([3, 8], &device);
        let logits = model.forward(input_ids);
        assert_eq!(logits.dims(), [3, 8, 256]);
    }

    #[test]
    fn tied_weights_produce_vocab_sized_logits() {
        let device = Default::default();
        let model = tiny_config().init::<B>(&device);

        let input_ids = Tensor::<B, 2, Int>::from_data([[1, 2, 3]], &device);
        let logits = model.forward(input_ids);
        // Last dim must be vocab_size
        assert_eq!(logits.dims()[2], 256);
    }
}
