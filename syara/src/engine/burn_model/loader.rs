//! Safetensors weight loading for Qwen3.5 models.
//!
//! Reads `config.json` for model dimensions, then loads weights from
//! `.safetensors` files into a [`Qwen3TextModel`].

use std::path::Path;

use burn::module::{Param, ParamId};
use burn::prelude::*;
use safetensors::tensor::SafeTensors;
use safetensors::Dtype;

use super::qwen3::{HybridBlock, Qwen3Config, Qwen3TextModel};
use crate::error::SyaraError;

// ── Config JSON deserialization ─────────────────────────────────────────────

/// Top-level config.json structure (only fields we need).
#[derive(serde::Deserialize)]
struct RawModelConfig {
    text_config: RawTextConfig,
}

#[derive(serde::Deserialize)]
struct RawTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    linear_conv_kernel_dim: usize,
    #[serde(default = "default_full_attn_interval")]
    full_attention_interval: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_tie")]
    tie_word_embeddings: bool,
    #[serde(default = "default_eos")]
    eos_token_id: usize,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(serde::Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
}

fn default_conv_kernel() -> usize { 4 }
fn default_full_attn_interval() -> usize { 4 }
fn default_max_pos() -> usize { 4096 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_tie() -> bool { true }
fn default_eos() -> usize { 248044 }
fn default_rope_theta() -> f32 { 10_000_000.0 }
fn default_partial_rotary() -> f64 { 0.25 }

// ── Public API ──────────────────────────────────────────────────────────────

/// Parse `config.json` from `model_dir` into a [`Qwen3Config`].
pub fn load_qwen3_config(model_dir: &Path) -> Result<Qwen3Config, SyaraError> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
        SyaraError::LlmError(format!("failed to read {}: {e}", config_path.display()))
    })?;
    let raw: RawModelConfig = serde_json::from_str(&config_str).map_err(|e| {
        SyaraError::LlmError(format!("failed to parse config.json: {e}"))
    })?;
    let tc = raw.text_config;
    let rope = tc.rope_parameters.unwrap_or(RopeParameters {
        rope_theta: default_rope_theta(),
        partial_rotary_factor: default_partial_rotary(),
    });

    Ok(Qwen3Config {
        vocab_size: tc.vocab_size,
        hidden_size: tc.hidden_size,
        intermediate_size: tc.intermediate_size,
        num_hidden_layers: tc.num_hidden_layers,
        num_attention_heads: tc.num_attention_heads,
        num_key_value_heads: tc.num_key_value_heads,
        head_dim: tc.head_dim,
        linear_num_key_heads: tc.linear_num_key_heads,
        linear_num_value_heads: tc.linear_num_value_heads,
        linear_key_head_dim: tc.linear_key_head_dim,
        linear_value_head_dim: tc.linear_value_head_dim,
        linear_conv_kernel_dim: tc.linear_conv_kernel_dim,
        full_attention_interval: tc.full_attention_interval,
        max_position_embeddings: tc.max_position_embeddings,
        rope_theta: rope.rope_theta,
        partial_rotary_factor: rope.partial_rotary_factor,
        rms_norm_eps: tc.rms_norm_eps,
        tie_word_embeddings: tc.tie_word_embeddings,
        eos_token_id: tc.eos_token_id,
    })
}

/// Load a Qwen3.5 model from safetensors files in `model_dir`.
///
/// 1. Initializes the model structure with zeros via [`Qwen3Config::init`]
/// 2. Reads the single `.safetensors` file (or first shard)
/// 3. Maps HuggingFace weight names to Burn module fields
/// 4. Skips `model.visual.*` and `mtp.*` weights
pub fn load_qwen3<B: Backend>(
    config: &Qwen3Config,
    model_dir: &Path,
    device: &B::Device,
) -> Result<Qwen3TextModel<B>, SyaraError> {
    let mut model = config.init(device);

    // Find safetensors file(s)
    let safetensors_path = find_safetensors(model_dir)?;
    let file_bytes = std::fs::read(&safetensors_path).map_err(|e| {
        SyaraError::LlmError(format!("failed to read {}: {e}", safetensors_path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&file_bytes).map_err(|e| {
        SyaraError::LlmError(format!("failed to deserialize safetensors: {e}"))
    })?;

    // Global weights
    let prefix = "model.language_model";
    assign_embedding(&tensors, &format!("{prefix}.embed_tokens.weight"), &mut model.embed_tokens, device)?;
    assign_rms_norm(&tensors, &format!("{prefix}.norm.weight"), &mut model.final_norm, device)?;

    // Per-layer weights
    for i in 0..config.num_hidden_layers {
        let lp = format!("{prefix}.layers.{i}");
        let block = &mut model.layers[i];

        assign_rms_norm(&tensors, &format!("{lp}.input_layernorm.weight"), &mut block.input_layernorm, device)?;
        assign_rms_norm(&tensors, &format!("{lp}.post_attention_layernorm.weight"), &mut block.post_attention_layernorm, device)?;

        // MLP (shared by all layer types)
        assign_linear(&tensors, &format!("{lp}.mlp.gate_proj.weight"), &mut block.mlp.gate_proj, device)?;
        assign_linear(&tensors, &format!("{lp}.mlp.up_proj.weight"), &mut block.mlp.up_proj, device)?;
        assign_linear(&tensors, &format!("{lp}.mlp.down_proj.weight"), &mut block.mlp.down_proj, device)?;

        // Hybrid block — DeltaNet or FullAttention
        match &mut block.hybrid {
            HybridBlock::Linear(dn) => {
                let dp = format!("{lp}.linear_attn");
                assign_linear(&tensors, &format!("{dp}.in_proj_qkv.weight"), &mut dn.in_proj_qkv, device)?;
                assign_linear(&tensors, &format!("{dp}.in_proj_z.weight"), &mut dn.in_proj_z, device)?;
                assign_linear(&tensors, &format!("{dp}.in_proj_a.weight"), &mut dn.in_proj_a, device)?;
                assign_linear(&tensors, &format!("{dp}.in_proj_b.weight"), &mut dn.in_proj_b, device)?;
                assign_linear(&tensors, &format!("{dp}.out_proj.weight"), &mut dn.out_proj, device)?;
                assign_param_1d(&tensors, &format!("{dp}.A_log"), &mut dn.a_log, device)?;
                assign_param_1d(&tensors, &format!("{dp}.dt_bias"), &mut dn.dt_bias, device)?;
                assign_conv1d(&tensors, &format!("{dp}.conv1d.weight"), &mut dn.conv1d, device)?;
                assign_rms_norm(&tensors, &format!("{dp}.norm.weight"), &mut dn.norm, device)?;
            }
            HybridBlock::Full(attn) => {
                let ap = format!("{lp}.self_attn");
                assign_linear(&tensors, &format!("{ap}.q_proj.weight"), &mut attn.q_proj, device)?;
                assign_linear(&tensors, &format!("{ap}.k_proj.weight"), &mut attn.k_proj, device)?;
                assign_linear(&tensors, &format!("{ap}.v_proj.weight"), &mut attn.v_proj, device)?;
                assign_linear(&tensors, &format!("{ap}.o_proj.weight"), &mut attn.o_proj, device)?;
                if let Some(ref mut qn) = attn.q_norm {
                    assign_rms_norm(&tensors, &format!("{ap}.q_norm.weight"), qn, device)?;
                }
                if let Some(ref mut kn) = attn.k_norm {
                    assign_rms_norm(&tensors, &format!("{ap}.k_norm.weight"), kn, device)?;
                }
                // RoPE cos/sin tables are computed, not loaded
            }
        }
    }

    Ok(model)
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Find the safetensors file in model_dir. Supports single-file and sharded layouts.
fn find_safetensors(model_dir: &Path) -> Result<std::path::PathBuf, SyaraError> {
    // Try single file first
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(single);
    }
    // Try sharded (single shard)
    let shard = model_dir.join("model.safetensors-00001-of-00001.safetensors");
    if shard.exists() {
        return Ok(shard);
    }
    // Scan for any safetensors file
    let entries = std::fs::read_dir(model_dir).map_err(|e| {
        SyaraError::LlmError(format!("cannot read model dir {}: {e}", model_dir.display()))
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            return Ok(path);
        }
    }
    Err(SyaraError::LlmError(format!(
        "no .safetensors file found in {}",
        model_dir.display()
    )))
}

/// Convert raw bytes to f32 based on safetensors dtype.
fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, SyaraError> {
    match view.dtype() {
        Dtype::BF16 => Ok(bf16_bytes_to_f32(view.data())),
        Dtype::F32 => Ok(f32_bytes_to_vec(view.data())),
        Dtype::F16 => Ok(f16_bytes_to_f32(view.data())),
        dt => Err(SyaraError::LlmError(format!("unsupported tensor dtype: {dt:?}"))),
    }
}

/// BF16 → f32: upper 16 bits of IEEE 754 float.
fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// F16 → f32: IEEE 754 half-precision conversion.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_to_f32(bits)
        })
        .collect()
}

/// Convert a single f16 value (as u16 bits) to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let f = f32::from_bits((127 - 14) << 23 | mant << 13) - f32::from_bits((127 - 14) << 23);
            if sign == 1 { -f } else { f }
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// F32 bytes → Vec<f32>.
fn f32_bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load a tensor from safetensors and create a Burn tensor.
fn load_tensor<B: Backend, const D: usize>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>, SyaraError> {
    let view = tensors.tensor(name).map_err(|e| {
        SyaraError::LlmError(format!("missing weight '{name}': {e}"))
    })?;
    let data = tensor_to_f32(&view)?;
    let shape: Vec<usize> = view.shape().to_vec();
    let tensor_data = burn::tensor::TensorData::new(data, shape);
    Ok(Tensor::from_data(tensor_data, device))
}

/// Assign a 2D weight tensor to a `Linear` module.
///
/// Safetensors (PyTorch convention) stores linear weights as `[out, in]`.
/// Burn stores them as `[in, out]`, so we transpose during loading.
fn assign_linear<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    linear: &mut burn::nn::Linear<B>,
    device: &B::Device,
) -> Result<(), SyaraError> {
    let t: Tensor<B, 2> = load_tensor(tensors, name, device)?;
    linear.weight = Param::initialized(ParamId::new(), t.transpose());
    Ok(())
}

/// Assign a 1D weight tensor to an `RmsNorm` module's gamma parameter.
fn assign_rms_norm<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    norm: &mut burn::nn::RmsNorm<B>,
    device: &B::Device,
) -> Result<(), SyaraError> {
    let t: Tensor<B, 1> = load_tensor(tensors, name, device)?;
    norm.gamma = Param::initialized(ParamId::new(), t);
    Ok(())
}

/// Assign a 2D weight tensor to an `Embedding` module.
fn assign_embedding<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    emb: &mut burn::nn::Embedding<B>,
    device: &B::Device,
) -> Result<(), SyaraError> {
    let t: Tensor<B, 2> = load_tensor(tensors, name, device)?;
    emb.weight = Param::initialized(ParamId::new(), t);
    Ok(())
}

/// Assign a 3D weight tensor to a `Conv1d` module.
fn assign_conv1d<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    conv: &mut burn::nn::conv::Conv1d<B>,
    device: &B::Device,
) -> Result<(), SyaraError> {
    let t: Tensor<B, 3> = load_tensor(tensors, name, device)?;
    conv.weight = Param::initialized(ParamId::new(), t);
    Ok(())
}

/// Assign a 1D tensor to a `Param<Tensor<B, 1>>` (e.g., A_log, dt_bias).
fn assign_param_1d<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    param: &mut Param<Tensor<B, 1>>,
    device: &B::Device,
) -> Result<(), SyaraError> {
    let t: Tensor<B, 1> = load_tensor(tensors, name, device)?;
    *param = Param::initialized(ParamId::new(), t);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn bf16_conversion_basic() {
        // BF16 for 1.0: sign=0, exp=01111111, mant=0000000 → 0x3F80
        let bytes = [0x80, 0x3F]; // little-endian BF16 for 1.0
        let result = bf16_bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", result[0]);
    }

    #[test]
    fn bf16_conversion_negative() {
        // BF16 for -2.0: 0xC000
        let bytes = [0x00, 0xC0];
        let result = bf16_bytes_to_f32(&bytes);
        assert!((result[0] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn bf16_conversion_zero() {
        let bytes = [0x00, 0x00];
        let result = bf16_bytes_to_f32(&bytes);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn f32_bytes_roundtrip() {
        let original = vec![1.0f32, -2.5, 0.0, 3.14];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let result = f32_bytes_to_vec(&bytes);
        assert_eq!(result, original);
    }

    #[test]
    fn load_config_from_real_model() {
        let model_dir = PathBuf::from("../models/Qwen3.5-0.8B-Base");
        if !model_dir.join("config.json").exists() {
            return; // skip if model not downloaded
        }
        let config = load_qwen3_config(&model_dir).unwrap();
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.full_attention_interval, 4);
        assert!((config.rope_theta - 10_000_000.0).abs() < 1.0);
        assert!((config.partial_rotary_factor - 0.25).abs() < 1e-6);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.eos_token_id, 248044);
    }

    #[test]
    fn load_config_from_fixture() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny-qwen");
        let config = load_qwen3_config(&fixture_dir).unwrap();
        assert_eq!(config.vocab_size, 256);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 16);
        assert_eq!(config.linear_num_key_heads, 4);
        assert_eq!(config.full_attention_interval, 2);
        assert!((config.rope_theta - 10_000.0).abs() < 1.0);
        assert_eq!(config.eos_token_id, 0);
    }

    #[test]
    fn load_model_from_fixture() {
        use burn::backend::NdArray;
        type B = NdArray<f32>;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny-qwen");
        let config = load_qwen3_config(&fixture_dir).unwrap();
        let device = Default::default();
        let model = load_qwen3::<B>(&config, &fixture_dir, &device).unwrap();

        // Verify model structure
        assert_eq!(model.num_layers(), 2);

        // Verify it can run a forward pass
        let input = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            [[1i64, 2, 3]],
            &device,
        );
        let logits = model.forward(input);
        assert_eq!(logits.dims(), [1, 3, 256]);
    }
}
