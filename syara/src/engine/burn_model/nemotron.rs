#![allow(dead_code)] // consumed by BurnEvaluator in Phase 5 sub-step 6

//! Nemotron-H text decoder model with ternary hybrid pattern.
//!
//! Architecture: 42-layer hybrid with three mixer types per the
//! `hybrid_override_pattern` string — M=Mamba2, *=Attention, -=MLP.
//! Each layer has a SINGLE mixer type (unlike Qwen3.5 where every layer
//! has both attention/DeltaNet AND a separate FFN).

use std::path::Path;

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

use super::attention::{FullAttention, FullAttentionConfig};
use super::ffn::{NemotronMlp, NemotronMlpConfig};
use super::mamba::{Mamba2Block, Mamba2Config};
use crate::error::SyaraError;

// ── Configuration ────────────────────────────────────────────────────────────

/// Nemotron-H model configuration, parsed from a flat config.json.
#[derive(Config, Debug)]
pub struct NemotronConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    pub n_groups: usize,
    pub ssm_state_size: usize,
    #[config(default = "4")]
    pub conv_kernel: usize,
    #[config(default = "true")]
    pub use_conv_bias: bool,
    #[config(default = "false")]
    pub use_bias: bool,
    #[config(default = "1e-5")]
    pub rms_norm_eps: f64,
    #[config(default = "false")]
    pub tie_word_embeddings: bool,
    #[config(default = "2")]
    pub eos_token_id: usize,
    /// Hybrid pattern string: M=Mamba, *=Attention, -=MLP.
    pub hybrid_override_pattern: String,
}

/// Mixer type for a single Nemotron layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerType {
    Mamba,
    Attention,
    Mlp,
}

impl NemotronConfig {
    /// Parse the hybrid pattern string into a vec of mixer types.
    pub fn parse_pattern(&self) -> Vec<MixerType> {
        self.hybrid_override_pattern
            .chars()
            .map(|c| match c {
                'M' => MixerType::Mamba,
                '*' => MixerType::Attention,
                '-' => MixerType::Mlp,
                _ => panic!("unknown mixer char '{c}' in hybrid_override_pattern"),
            })
            .collect()
    }

    /// Initialize a [`NemotronModel`] from this config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> NemotronModel<B> {
        let pattern = self.parse_pattern();
        assert_eq!(
            pattern.len(),
            self.num_hidden_layers,
            "pattern length {} != num_hidden_layers {}",
            pattern.len(),
            self.num_hidden_layers
        );

        let embed = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);

        let layers: Vec<NemotronBlock<B>> = pattern
            .iter()
            .map(|&mt| self.init_block(mt, device))
            .collect();

        let norm_f = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);

        let lm_head = LinearConfig::new(self.hidden_size, self.vocab_size)
            .with_bias(false)
            .init(device);

        NemotronModel {
            embeddings: embed,
            layers,
            norm_f,
            lm_head,
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }

    fn init_block<B: Backend>(&self, mt: MixerType, device: &B::Device) -> NemotronBlock<B> {
        let norm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);

        let mixer = match mt {
            MixerType::Mamba => NemotronMixer::Mamba(
                Mamba2Config {
                    d_model: self.hidden_size,
                    num_heads: self.mamba_num_heads,
                    head_dim: self.mamba_head_dim,
                    n_groups: self.n_groups,
                    ssm_state_size: self.ssm_state_size,
                    conv_kernel: self.conv_kernel,
                    use_conv_bias: self.use_conv_bias,
                    use_bias: self.use_bias,
                    rms_norm_eps: self.rms_norm_eps,
                }
                .init(device),
            ),
            MixerType::Attention => NemotronMixer::Attention(
                FullAttentionConfig {
                    d_model: self.hidden_size,
                    n_heads: self.num_attention_heads,
                    n_kv_heads: self.num_key_value_heads,
                    head_dim: self.head_dim,
                    max_seq_len: 1, // No RoPE — partial_rotary_factor=0
                    qk_norm: false,
                    partial_rotary_factor: 0.0,
                    rope_theta: 10_000.0,
                    rms_norm_eps: self.rms_norm_eps,
                }
                .init(device),
            ),
            MixerType::Mlp => NemotronMixer::Mlp(
                NemotronMlpConfig {
                    d_model: self.hidden_size,
                    d_intermediate: self.intermediate_size,
                }
                .init(device),
            ),
        };

        NemotronBlock { norm, mixer }
    }
}

// ── Mixer Enum ──────────────────────────────────────────────────────────────

/// One of three mixer types for a Nemotron layer.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum NemotronMixer<B: Backend> {
    Mamba(Mamba2Block<B>),
    Attention(FullAttention<B>),
    Mlp(NemotronMlp<B>),
}

impl<B: Backend> NemotronMixer<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            NemotronMixer::Mamba(m) => m.forward(x),
            NemotronMixer::Attention(a) => a.forward(x),
            NemotronMixer::Mlp(m) => m.forward(x),
        }
    }
}

// ── Block ───────────────────────────────────────────────────────────────────

/// Single Nemotron layer: pre-norm → mixer → residual.
#[derive(Module, Debug)]
pub struct NemotronBlock<B: Backend> {
    pub(crate) norm: RmsNorm<B>,
    pub(crate) mixer: NemotronMixer<B>,
}

impl<B: Backend> NemotronBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.norm.forward(x);
        residual + self.mixer.forward(h)
    }
}

// ── Full Model ──────────────────────────────────────────────────────────────

/// Nemotron-H text decoder: embedding → N blocks → norm → lm_head.
#[derive(Module, Debug)]
pub struct NemotronModel<B: Backend> {
    pub(crate) embeddings: Embedding<B>,
    pub(crate) layers: Vec<NemotronBlock<B>>,
    pub(crate) norm_f: RmsNorm<B>,
    pub(crate) lm_head: Linear<B>,
    pub(crate) tie_word_embeddings: bool,
}

impl<B: Backend> NemotronModel<B> {
    /// Forward pass: input token ids → logits over vocabulary.
    ///
    /// `input_ids`: `[batch, seq_len]` → `[batch, seq_len, vocab_size]`
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut h = self.embeddings.forward(input_ids);

        for layer in &self.layers {
            h = layer.forward(h);
        }

        let h = self.norm_f.forward(h);

        if self.tie_word_embeddings {
            let weight = self.embeddings.weight.val();
            let weight = weight.unsqueeze_dim::<3>(0).transpose();
            h.matmul(weight)
        } else {
            self.lm_head.forward(h)
        }
    }
}

// ── Config JSON deserialization ─────────────────────────────────────────────

/// Flat config.json structure for Nemotron models.
#[derive(serde::Deserialize)]
struct RawNemotronConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    n_groups: usize,
    ssm_state_size: usize,
    #[serde(default = "default_conv_kernel")]
    conv_kernel: usize,
    #[serde(default = "default_use_conv_bias")]
    use_conv_bias: bool,
    #[serde(default)]
    use_bias: bool,
    #[serde(default = "default_nemotron_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_nemotron_eos")]
    eos_token_id: usize,
    hybrid_override_pattern: String,
}

fn default_conv_kernel() -> usize { 4 }
fn default_use_conv_bias() -> bool { true }
fn default_nemotron_eps() -> f64 { 1e-5 }
fn default_nemotron_eos() -> usize { 2 }

/// Parse `config.json` from `model_dir` into a [`NemotronConfig`].
pub fn load_nemotron_config(model_dir: &Path) -> Result<NemotronConfig, SyaraError> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
        SyaraError::LlmError(format!("failed to read {}: {e}", config_path.display()))
    })?;
    let raw: RawNemotronConfig = serde_json::from_str(&config_str).map_err(|e| {
        SyaraError::LlmError(format!("failed to parse config.json: {e}"))
    })?;

    Ok(NemotronConfig {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        mamba_num_heads: raw.mamba_num_heads,
        mamba_head_dim: raw.mamba_head_dim,
        n_groups: raw.n_groups,
        ssm_state_size: raw.ssm_state_size,
        conv_kernel: raw.conv_kernel,
        use_conv_bias: raw.use_conv_bias,
        use_bias: raw.use_bias,
        rms_norm_eps: raw.rms_norm_eps,
        tie_word_embeddings: raw.tie_word_embeddings,
        eos_token_id: raw.eos_token_id,
        hybrid_override_pattern: raw.hybrid_override_pattern,
    })
}

impl<B: Backend> super::ForwardModel<B> for NemotronModel<B> {
    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_config() -> NemotronConfig {
        NemotronConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            mamba_num_heads: 4,
            mamba_head_dim: 8,
            n_groups: 2,
            ssm_state_size: 16,
            conv_kernel: 4,
            use_conv_bias: true,
            use_bias: false,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            eos_token_id: 0,
            hybrid_override_pattern: "M-*".to_string(),
        }
    }

    #[test]
    fn pattern_parsing() {
        let cfg = tiny_config();
        let pattern = cfg.parse_pattern();
        assert_eq!(
            pattern,
            vec![MixerType::Mamba, MixerType::Mlp, MixerType::Attention]
        );
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
    fn mixer_dispatch_correct() {
        let cfg = tiny_config();
        let device = Default::default();
        let model = cfg.init::<B>(&device);

        assert_eq!(model.layers.len(), 3);
        assert!(matches!(model.layers[0].mixer, NemotronMixer::Mamba(_)));
        assert!(matches!(model.layers[1].mixer, NemotronMixer::Mlp(_)));
        assert!(matches!(model.layers[2].mixer, NemotronMixer::Attention(_)));
    }

    #[test]
    fn forward_single_token() {
        let device = Default::default();
        let model = tiny_config().init::<B>(&device);

        let input_ids = Tensor::<B, 2, Int>::zeros([1, 1], &device);
        let logits = model.forward(input_ids);
        assert_eq!(logits.dims(), [1, 1, 256]);
    }
}
