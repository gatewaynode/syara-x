# Open Bugs

Outstanding, un-fixed bugs in `syara-x`. Closed bugs live in dated archives
(most recent: `tasks/04-24-2026_BUGS.md`).

Every fix must include a regression test. Dense, historical numbering —
do not reuse. Next number: **BUG-039**.

---

## On hold — blocked on backend migration (Phase 5, `burn-llm` feature)

Real-model integration tests (`cargo test --features burn-llm -- --ignored`)
surfaced these. Fixture tests pass. The `burn-llm` feature is currently
walled off behind a roadmap error; a candle or mistral.rs migration is
planned (see `ROADMAP.md`).

### BUG-036: Qwen3.5-0.8B tensor shape crash in `BurnEvaluator`
- **Feature:** `burn-llm`
- **File:** `syara/src/engine/burn_evaluator.rs` (and the Qwen3 model assembly)
- **Status:** open, on hold (pending backend migration)
- **Symptom:**
  ```
  Mul: Incompatible size at dimension '2' => '2048 != 128'
  Lhs [1, 70, 2048], Rhs [1, 1, 128]
  ```
- **Root cause:** `models/Qwen3.5-0.8B-Base/config.json` is
  **Qwen3_5ForConditionalGeneration** — a multimodal VL model whose
  features our Burn code does not implement:
  - `attn_output_gate: true` — attention output gating (not in our `FullAttention`)
  - `mrope_interleaved: true` + `mrope_section: [11,11,10]` — multi-dimensional RoPE for vision; we use plain RoPE
  - `layer_types` explicit list (replaces `full_attention_interval` derivation)
  - `head_dim: 256`, `linear_key_head_dim: 128`, `linear_num_key_heads: 16` → DeltaNet `v_dim=2048`, much larger than tiny fixture dims — exposes a broadcast-axis bug in DeltaNet recurrence
  - Architecture string `Qwen3_5ForConditionalGeneration` may not match our detector
- **Options for future fix:**
  1. Replace Burn inference with candle or mistral.rs (Metal/MLX support, KV cache, existing Qwen3/Nemotron impls). Keep Burn fixture tests as architecture validation.
  2. Find a non-VL Qwen3 model matching what we built (no output gate, plain RoPE).
  3. Implement missing features in Burn (output-gated attention, mrope, KV cache) — weeks of work to reach where candle/mistral.rs already are.

### BUG-037: Nemotron-3-Nano-4B impractical CPU inference speed
- **Feature:** `burn-llm`
- **File:** `syara/src/engine/burn_evaluator.rs` (and the Nemotron model assembly)
- **Status:** open, on hold (pending backend migration)
- **Symptom:** Model loads and runs correctly, but each generated token requires a full forward pass through 42 layers × 3136-dim on the NdArray CPU backend with no KV cache. Expected runtime: hours per generation.
- **Root cause:** NdArray CPU backend + no KV cache + 4B parameter model.
- **Options for future fix:** MLX-level Apple Silicon performance will require candle (Metal) or mistral.rs (MLX bindings) regardless — Burn's wgpu/Metal path uses generic shaders. Paths 1 and 3 from BUG-036 apply here too.

---

## How to add a new bug

1. Reserve the next number (currently BUG-038).
2. Give it one of: open / in-progress / fixed.
3. Link to file + line where reproducible.
4. When you close it, move it into the most recent dated archive file and
   add the regression test reference.
