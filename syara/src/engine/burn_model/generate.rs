//! Greedy text generation for Burn-based models.
//!
//! Simple argmax decoding loop — sufficient for YES/NO classification tasks
//! where we only need a few output tokens.

use burn::prelude::*;

use super::ForwardModel;

/// Generate tokens greedily (argmax) from any model implementing `ForwardModel`.
///
/// Runs the full model on the growing sequence each step (no KV cache).
/// Stops when `eos_token_id` is produced or `max_new_tokens` is reached.
///
/// Returns the generated token IDs (excluding the input).
pub fn greedy_generate<B: Backend>(
    model: &impl ForwardModel<B>,
    input_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    device: &B::Device,
) -> Vec<u32> {
    let mut ids: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let mut generated = Vec::with_capacity(max_new_tokens);

    for _ in 0..max_new_tokens {
        let seq_len = ids.len();
        let input = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [1, seq_len]),
            device,
        );

        let logits = model.forward(input); // [1, seq_len, vocab_size]

        // Take logits at the last position
        let last_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim::<2>(1); // [1, vocab_size]
        let next_token = last_logits.argmax(1); // [1, 1]
        let next_id: i64 = next_token.into_scalar().elem();
        let next_id_u32 = next_id as u32;

        if next_id_u32 == eos_token_id {
            break;
        }

        generated.push(next_id_u32);
        ids.push(next_id);
    }

    generated
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn generate_stops_at_max_tokens() {
        let device = Default::default();
        // Tiny random model — output is gibberish but generation loop should work
        let config = super::super::qwen3::Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            eos_token_id: 31,
        };
        let model = config.init::<B>(&device);

        let input_ids = vec![0u32, 1, 2];
        let output = greedy_generate(&model, &input_ids, 5, 31, &device);

        // Should produce at most 5 tokens (may produce fewer if EOS hit)
        assert!(output.len() <= 5, "got {} tokens", output.len());
        // All generated tokens should be valid vocab indices
        for &tok in &output {
            assert!(tok < 32, "token {tok} out of vocab range");
        }
    }
}
