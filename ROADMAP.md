# Roadmap

## Migrate local LLM inference to candle-rs

**Status:** Planned  
**Feature flags affected:** `burn-llm`, `burn-llm-gpu`  
**Current state:** Entry points (`BurnEvaluator::from_dir`, `BurnEvaluatorBuilder::build`) return an error pointing here. Underlying model code is preserved for reference.

### Motivation

The Burn-based local inference backend hit two blockers during real-model testing:

1. **Qwen3.5-0.8B — tensor shape crash.**  
   The HuggingFace checkpoint (`Qwen3_5ForConditionalGeneration`) is a multimodal VL model. Our Burn code does not implement output-gated attention (`attn_output_gate: true`), multi-dimensional RoPE (`mrope_interleaved: true`, `mrope_section: [11,11,10]`), or the resulting larger head dimensions. A broadcast bug in the DeltaNet recurrence is also exposed at real model dimensions.

2. **Nemotron-3-Nano-4B — impractical CPU speed.**  
   The NdArray CPU backend with no KV cache requires a full 42-layer forward pass per token over a 3136-dim embedding space. Generation is measured in hours per response.

Implementing the missing features in Burn would take weeks of work to reach a point that [candle](https://github.com/huggingface/candle) already covers, and Burn's wgpu/Metal path uses generic shaders rather than MLX/Metal Performance Shaders, so Apple Silicon performance would still lag.

### Plan

Replace `burn_evaluator.rs` and the `engine/burn_model/` subtree with a candle-based implementation:

- **Target models:** Qwen3.5 (text-only checkpoint, not VL) and Nemotron-3-Nano-4B  
- **Backend:** candle with Metal support for Apple Silicon; CUDA for Linux GPU  
- **KV cache:** required for Nemotron to be practical  
- **Tokenizer:** HuggingFace `tokenizers` crate (already a dependency)  
- **Feature flags:** reuse `burn-llm` / `burn-llm-gpu` names (rename semantics, not API surface)  
- **Public API:** `BurnEvaluator` and `BurnEvaluatorBuilder` stay; internals replace Burn with candle  
- **Fixture tests:** restore `fixture_load_and_evaluate` / `builder_cpu_default` / `fixture_load_and_evaluate_nemotron` once the wall-off guard is removed  

### Reference

- candle repo: <https://github.com/huggingface/candle>
- candle Qwen example: `candle-examples/examples/qwen/`
- Phase 5 bug notes: `tasks/todo.md` → "Phase 5 Known Bugs"
- Existing Burn model code: `syara/src/engine/burn_model/` (kept for reference)

## YARA-X parity gaps (condition DSL)

**Status:** Deferred  
**Shipped:** 2026-04-22 — `#pattern` count operators, comparisons (`== != < <= > >=`), `+`/`-` arithmetic.

Items below are known YARA-X parity gaps not yet implemented. None are blocking real rule authorship today; add as rule corpus demand appears.

- **Arithmetic `*` / `/` / `%`.** The `*` token currently serves as the set-wildcard sigil inside `any of ($prefix*)`. Adding multiplication needs parse-context disambiguation (the wildcard is always inside a `(`-delimited set, the operator always between two integer expressions, but the grammar currently does not carry that context). Workaround: rewrite `#a * 2 >= b` as `(#a + #a) >= b` or split into two comparisons.
- **`@pattern[i]` (offset of the i-th occurrence).** Returns the byte offset at which the i-th match of `pattern` starts. 1-indexed. Needs a new subscript grammar and wider `MatchDetail.start_pos` plumbing (already `Option<usize>`).
- **`!pattern[i]` (length of the i-th occurrence).** Companion to `@`. Needs `MatchDetail.end_pos - start_pos`.
- **Integer size suffixes (`KB`, `MB`, etc.).** YARA lets you write `100KB` as `102400`. Lexer-level; trivial when a rule actually wants it.
- **`not not x`.** Current grammar rejects chained `not`. YARA-X allows it; parens (`not (not x)`) work as a workaround.

## `#`-operator count ceiling for non-string rules

`#identifier` on a `similarity` / `classifier` / `llm` / `phash` rule has a ceiling of 1 — those matchers produce `vec![detail]` or `vec![]` per rule invocation, not one entry per "occurrence". `#llm1 >= 1` is still meaningful ("did the LLM fire"), but `#classifier1 >= 3` can never be true. If a matcher changes its emission cardinality (e.g. chunk-level classifier results expanded into one detail per positive chunk), revisit the documentation.
