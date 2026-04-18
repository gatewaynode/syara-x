#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["safetensors>=0.4", "numpy>=1.24"]
# ///
"""Generate tiny test fixtures for burn_evaluator tests.

Creates two fixture sets:
  tiny-qwen/     — Qwen3.5 config matching tiny_config() in qwen3.rs tests
  tiny-nemotron/ — Nemotron-H config matching tiny_config() in nemotron.rs tests

Each contains config.json, model.safetensors, tokenizer.json.

Usage:
  uv run scripts/create_test_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

QWEN_DIR = Path("syara/tests/fixtures/tiny-qwen")
NEMOTRON_DIR = Path("syara/tests/fixtures/tiny-nemotron")

# ── Model config (matches tiny_config() in qwen3.rs) ────────────────────────

VOCAB_SIZE = 256
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 16
LINEAR_NUM_KEY_HEADS = 4
LINEAR_NUM_VALUE_HEADS = 4
LINEAR_KEY_HEAD_DIM = 16
LINEAR_VALUE_HEAD_DIM = 16
LINEAR_CONV_KERNEL_DIM = 4
FULL_ATTENTION_INTERVAL = 2
MAX_POSITION_EMBEDDINGS = 512
ROPE_THETA = 10_000.0
PARTIAL_ROTARY_FACTOR = 0.25
RMS_NORM_EPS = 1e-6
EOS_TOKEN_ID = 0  # use 0 as EOS for tiny model


def main():
    generate_qwen_fixture()
    print()
    generate_nemotron_fixture()


# ── Shared helpers ──────────────────────────────────────────────────────────

def save_tokenizer(out_dir, vocab_size=256, eos_token_id=0):
    """Save a minimal byte-level tokenizer (each byte 0..vocab_size is its own token)."""
    vocab = {chr(i) if 32 <= i < 127 else f"<0x{i:02X}>": i for i in range(vocab_size)}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": eos_token_id,
                "content": "<eos>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2) + "\n")
    print("Saved tokenizer.json")


def print_summary(tensors, out_dir):
    total_params = sum(t.size for t in tensors.values())
    total_bytes = sum(t.nbytes for t in tensors.values())
    print(f"\nFixture summary: {total_params:,} parameters, {total_bytes:,} bytes")
    print(f"Output: {out_dir}/")


# ── Qwen3.5 fixture ────────────────────────────────────────────────────────

def generate_qwen_fixture():
    print("=== Generating tiny-qwen fixture ===")
    QWEN_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    tensors = {}

    prefix = "model.language_model"

    # ── Global weights ───────────────────────────────────────────────────
    tensors[f"{prefix}.embed_tokens.weight"] = small_rand(rng, VOCAB_SIZE, HIDDEN_SIZE)
    tensors[f"{prefix}.norm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)

    # ── Per-layer weights ────────────────────────────────────────────────
    for i in range(NUM_HIDDEN_LAYERS):
        lp = f"{prefix}.layers.{i}"

        # Shared norms and MLP
        tensors[f"{lp}.input_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
        tensors[f"{lp}.post_attention_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
        tensors[f"{lp}.mlp.gate_proj.weight"] = small_rand(rng, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        tensors[f"{lp}.mlp.up_proj.weight"] = small_rand(rng, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        tensors[f"{lp}.mlp.down_proj.weight"] = small_rand(rng, HIDDEN_SIZE, INTERMEDIATE_SIZE)

        is_full_attn = (i + 1) % FULL_ATTENTION_INTERVAL == 0

        if is_full_attn:
            # FullAttention layer
            ap = f"{lp}.self_attn"
            q_dim = NUM_ATTENTION_HEADS * HEAD_DIM  # 4 * 16 = 64
            kv_dim = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 2 * 16 = 32

            tensors[f"{ap}.q_proj.weight"] = small_rand(rng, q_dim, HIDDEN_SIZE)
            tensors[f"{ap}.k_proj.weight"] = small_rand(rng, kv_dim, HIDDEN_SIZE)
            tensors[f"{ap}.v_proj.weight"] = small_rand(rng, kv_dim, HIDDEN_SIZE)
            tensors[f"{ap}.o_proj.weight"] = small_rand(rng, HIDDEN_SIZE, q_dim)
            tensors[f"{ap}.q_norm.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
            tensors[f"{ap}.k_norm.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
        else:
            # GatedDeltaNet (linear attention) layer
            dp = f"{lp}.linear_attn"
            q_dim = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM  # 4 * 16 = 64
            k_dim = q_dim
            v_dim = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM  # 4 * 16 = 64
            qkv_dim = q_dim + k_dim + v_dim  # 192

            tensors[f"{dp}.in_proj_qkv.weight"] = small_rand(rng, qkv_dim, HIDDEN_SIZE)
            tensors[f"{dp}.in_proj_z.weight"] = small_rand(rng, v_dim, HIDDEN_SIZE)
            tensors[f"{dp}.in_proj_a.weight"] = small_rand(rng, LINEAR_NUM_KEY_HEADS, HIDDEN_SIZE)
            tensors[f"{dp}.in_proj_b.weight"] = small_rand(rng, LINEAR_NUM_KEY_HEADS, HIDDEN_SIZE)
            tensors[f"{dp}.out_proj.weight"] = small_rand(rng, HIDDEN_SIZE, v_dim)
            # A_log and dt_bias have NO .weight suffix
            tensors[f"{dp}.A_log"] = rng.standard_normal(LINEAR_NUM_KEY_HEADS).astype(np.float32)
            tensors[f"{dp}.dt_bias"] = rng.standard_normal(LINEAR_NUM_KEY_HEADS).astype(np.float32)
            # conv1d: depthwise, [channels, 1, kernel_size]
            tensors[f"{dp}.conv1d.weight"] = small_rand(rng, qkv_dim, 1, LINEAR_CONV_KERNEL_DIM)
            tensors[f"{dp}.norm.weight"] = np.ones(v_dim, dtype=np.float32)

    # ── Save ─────────────────────────────────────────────────────────────
    save_file(tensors, QWEN_DIR / "model.safetensors")
    print(f"Saved model.safetensors ({len(tensors)} tensors)")

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "text_config": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "linear_num_key_heads": LINEAR_NUM_KEY_HEADS,
            "linear_num_value_heads": LINEAR_NUM_VALUE_HEADS,
            "linear_key_head_dim": LINEAR_KEY_HEAD_DIM,
            "linear_value_head_dim": LINEAR_VALUE_HEAD_DIM,
            "linear_conv_kernel_dim": LINEAR_CONV_KERNEL_DIM,
            "full_attention_interval": FULL_ATTENTION_INTERVAL,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "rms_norm_eps": RMS_NORM_EPS,
            "tie_word_embeddings": True,
            "eos_token_id": EOS_TOKEN_ID,
            "rope_parameters": {
                "rope_theta": ROPE_THETA,
                "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            },
        },
    }
    (QWEN_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Saved config.json")

    save_tokenizer(QWEN_DIR, VOCAB_SIZE, EOS_TOKEN_ID)
    print_summary(tensors, QWEN_DIR)


# ── Nemotron-H fixture ──────────────────────────────────────────────────────

# Matches tiny_config() in nemotron.rs tests: 3 layers, pattern "M-*"
NEM_VOCAB = 256
NEM_HIDDEN = 64
NEM_INTERMEDIATE = 128   # MLP intermediate_size
NEM_NUM_LAYERS = 3
NEM_ATTN_HEADS = 4
NEM_KV_HEADS = 2
NEM_HEAD_DIM = 16
NEM_MAMBA_HEADS = 4
NEM_MAMBA_HEAD_DIM = 8
NEM_N_GROUPS = 2
NEM_SSM_STATE = 16
NEM_CONV_KERNEL = 4
NEM_PATTERN = "M-*"
NEM_EOS = 0

# Derived Mamba dimensions (must match Mamba2Config calculations):
NEM_MAMBA_INTER = NEM_MAMBA_HEADS * NEM_MAMBA_HEAD_DIM    # 32
NEM_CONV_DIM = NEM_MAMBA_INTER + 2 * NEM_N_GROUPS * NEM_SSM_STATE  # 32 + 64 = 96
NEM_PROJ_SIZE = NEM_MAMBA_INTER + NEM_CONV_DIM + NEM_MAMBA_HEADS   # 32 + 96 + 4 = 132


def generate_nemotron_fixture():
    print("=== Generating tiny-nemotron fixture ===")
    NEMOTRON_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(99)
    tensors = {}

    # ── Global weights (backbone.* namespace) ────────────────────────────
    tensors["backbone.embeddings.weight"] = small_rand(rng, NEM_VOCAB, NEM_HIDDEN)
    tensors["backbone.norm_f.weight"] = np.ones(NEM_HIDDEN, dtype=np.float32)
    tensors["lm_head.weight"] = small_rand(rng, NEM_VOCAB, NEM_HIDDEN)

    # ── Per-layer weights ────────────────────────────────────────────────
    for i, mixer_char in enumerate(NEM_PATTERN):
        lp = f"backbone.layers.{i}"
        mp = f"{lp}.mixer"

        # Pre-norm (shared by all layer types)
        tensors[f"{lp}.norm.weight"] = np.ones(NEM_HIDDEN, dtype=np.float32)

        if mixer_char == "M":
            # Mamba2 mixer
            # in_proj: [d_model, projection_size] — safetensors stores [out, in]
            tensors[f"{mp}.in_proj.weight"] = small_rand(rng, NEM_PROJ_SIZE, NEM_HIDDEN)
            # conv1d: depthwise [conv_dim, 1, kernel]
            tensors[f"{mp}.conv1d.weight"] = small_rand(rng, NEM_CONV_DIM, 1, NEM_CONV_KERNEL)
            tensors[f"{mp}.conv1d.bias"] = np.zeros(NEM_CONV_DIM, dtype=np.float32)
            # 1D params (no .weight suffix)
            tensors[f"{mp}.A_log"] = rng.standard_normal(NEM_MAMBA_HEADS).astype(np.float32)
            tensors[f"{mp}.D"] = np.ones(NEM_MAMBA_HEADS, dtype=np.float32)
            tensors[f"{mp}.dt_bias"] = rng.standard_normal(NEM_MAMBA_HEADS).astype(np.float32)
            # MambaRMSNormGated weight: [intermediate_size] = [32]
            tensors[f"{mp}.norm.weight"] = np.ones(NEM_MAMBA_INTER, dtype=np.float32)
            # out_proj: [intermediate_size, d_model]
            tensors[f"{mp}.out_proj.weight"] = small_rand(rng, NEM_HIDDEN, NEM_MAMBA_INTER)

        elif mixer_char == "-":
            # NemotronMlp (relu²): up_proj + down_proj
            tensors[f"{mp}.up_proj.weight"] = small_rand(rng, NEM_INTERMEDIATE, NEM_HIDDEN)
            tensors[f"{mp}.down_proj.weight"] = small_rand(rng, NEM_HIDDEN, NEM_INTERMEDIATE)

        elif mixer_char == "*":
            # FullAttention (no QK-norm, no RoPE)
            q_dim = NEM_ATTN_HEADS * NEM_HEAD_DIM     # 64
            kv_dim = NEM_KV_HEADS * NEM_HEAD_DIM       # 32
            tensors[f"{mp}.q_proj.weight"] = small_rand(rng, q_dim, NEM_HIDDEN)
            tensors[f"{mp}.k_proj.weight"] = small_rand(rng, kv_dim, NEM_HIDDEN)
            tensors[f"{mp}.v_proj.weight"] = small_rand(rng, kv_dim, NEM_HIDDEN)
            tensors[f"{mp}.o_proj.weight"] = small_rand(rng, NEM_HIDDEN, q_dim)

    # ── Save ─────────────────────────────────────────────────────────────
    save_file(tensors, NEMOTRON_DIR / "model.safetensors")
    print(f"Saved model.safetensors ({len(tensors)} tensors)")

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": NEM_VOCAB,
        "hidden_size": NEM_HIDDEN,
        "intermediate_size": NEM_INTERMEDIATE,
        "num_hidden_layers": NEM_NUM_LAYERS,
        "num_attention_heads": NEM_ATTN_HEADS,
        "num_key_value_heads": NEM_KV_HEADS,
        "head_dim": NEM_HEAD_DIM,
        "mamba_num_heads": NEM_MAMBA_HEADS,
        "mamba_head_dim": NEM_MAMBA_HEAD_DIM,
        "n_groups": NEM_N_GROUPS,
        "ssm_state_size": NEM_SSM_STATE,
        "conv_kernel": NEM_CONV_KERNEL,
        "use_conv_bias": True,
        "use_bias": False,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "eos_token_id": NEM_EOS,
        "hybrid_override_pattern": NEM_PATTERN,
    }
    (NEMOTRON_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Saved config.json")

    save_tokenizer(NEMOTRON_DIR, NEM_VOCAB, NEM_EOS)
    print_summary(tensors, NEMOTRON_DIR)


def small_rand(rng, *shape):
    """Small random f32 tensor (scaled down to avoid numerical issues)."""
    return (rng.standard_normal(shape) * 0.02).astype(np.float32)


if __name__ == "__main__":
    main()
