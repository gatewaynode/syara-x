# Burn 0.20.1 API Notes for Phase 2-4 Implementation

## Key Findings

**Burn has built-in modules we can reuse directly:**
- `RmsNorm` — `burn::nn::RmsNormConfig::new(d_model).with_epsilon(eps).init(&device)`
- `RotaryEncoding` — `burn::nn::RotaryEncodingConfig::new(max_seq_len, d_model).with_theta(theta).init(&device)`
- `Linear` — `burn::nn::LinearConfig::new(d_input, d_output).with_bias(false).init(&device)`
- `Embedding` — `burn::nn::EmbeddingConfig::new(n_embedding, d_model).init(&device)`
- `Conv1d` — `burn::nn::Conv1dConfig::new(channels_in, channels_out, kernel_size).with_bias(bias).with_padding(...).init(&device)`
- `MultiHeadAttention` — exists but is standard MHA, NOT suitable for GQA with QK-norm. We need custom attention.
- `SwiGlu` — exists but uses its own linear layers. Our FFN is gate/up/down pattern, so custom is better.

## Phase 2 Revised Plan

Since RmsNorm and RotaryEncoding are built-in, Phase 2 simplifies to:
1. **`burn_model/mod.rs`** — Module declarations, re-exports
2. **`burn_model/attention.rs`** — Custom FullAttention with GQA + QK-norm (Burn's MHA doesn't support GQA or QK-norm)
3. **`burn_model/ffn.rs`** — FeedForward (gate_proj + up_proj with SiLU, down_proj)
4. **No custom RmsNorm or RoPE needed** — use `burn::nn::RmsNorm` and `burn::nn::RotaryEncoding` directly

### Partial RoPE for Qwen3.5
Qwen3.5 uses `partial_rotary_factor=0.25` (only 25% of head_dim gets rotated). Burn's RotaryEncoding applies to the full d_model. We need a wrapper that:
1. Splits Q/K into rotated portion (first 25%) and pass-through portion (remaining 75%)
2. Applies RoPE to rotated portion only
3. Concatenates them back

This is a thin wrapper, not a full custom implementation.

## Module Pattern

```rust
use burn::prelude::*;  // brings in Backend, Tensor, Module, Config, etc.

#[derive(Module, Debug)]
pub struct MyModule<B: Backend> {
    linear: burn::nn::Linear<B>,
    norm: burn::nn::RmsNorm<B>,
    // Param<Tensor<B, N>> for learnable parameters
    some_param: Param<Tensor<B, 1>>,
}
```

### Config Pattern
```rust
#[derive(Config, Debug)]
pub struct MyModuleConfig {
    pub d_model: usize,
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl MyModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModule<B> {
        MyModule {
            linear: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            norm: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.epsilon)
                .init(device),
            some_param: Initializer::Zeros.init([self.d_model], device),
        }
    }
}
```

## Tensor API Reference

### Creation
- `Tensor::<B, D>::zeros(shape, &device)` / `ones` / `full`
- `Tensor::<B, D>::from_floats(data, &device)` / `from_data(data, &device)`
- `Tensor::<B, D, Int>::arange(0..n, &device)` / `arange_step(0..n, step, &device)`
- `Tensor::random(shape, Distribution::Default, &device)`

### Shape Operations
- `.reshape(shape)` — `tensor.reshape([batch, seq, dim])`
- `.transpose()` — swaps last two dims
- `.swap_dims(d1, d2)` — swap arbitrary dims
- `.unsqueeze::<N>()` — add dim at front (increase rank to N)
- `.unsqueeze_dim::<N>(dim)` — add dim at specific position
- `.squeeze_dim::<N>(dim)` — remove dim at position (decrease rank to N)
- `.narrow(dim, start, length)` — slice along one dimension
- `.slice([ranges])` — multi-dim slicing with Range<usize>
- `.dims()` -> `[usize; D]`
- `.shape()` -> `Shape`
- `.repeat_dim(dim, times)` — repeat along dimension

### Math Operations
- `.matmul(other)` — matrix multiplication
- `.mul(other)` / `.mul_scalar(s)` — element-wise multiply
- `.div(other)` / `.div_scalar(s)` — element-wise divide
- `.add(other)` / `.add_scalar(s)` — element-wise add
- `.sub(other)` / `.sub_scalar(s)` — element-wise subtract
- `.exp()` / `.log()` / `.sqrt()` / `.recip()`
- `.neg()` — negate
- `.square()` — x^2 (Burn does NOT have general scalar-to-tensor pow)
- `.mean_dim(dim)` / `.sum_dim(dim)` / `.max_dim(dim)`
- `.clone()` — required before consuming operations (tensors are consumed)
- `.cos()` / `.sin()`

### Type Casting
- `.float()` — cast Int tensor to Float
- `.int()` — cast Float tensor to Int
- `.cast(DType::F32)` — cast to specific dtype
- `.dtype()` -> DType

### Activation Functions (free functions in `burn::tensor::activation`)
- `softmax(tensor, dim)` — along dimension
- `silu(tensor)` — SiLU/Swish activation: `x * sigmoid(x)`
- `sigmoid(tensor)`
- `relu(tensor)`

### Concatenation
- `Tensor::cat(vec![t1, t2], dim)` — concatenate along dimension

### Masking
- `.mask_fill(mask, value)` — fill where mask is true
- `.mask_where(condition, other)` — select from other where condition

### Module nn API
- `burn::nn::Linear` — `.forward(input)` for any rank
- `burn::nn::RmsNorm` — `.forward(input)` normalizes along last dim
- `burn::nn::RotaryEncoding` — `.forward(input)` or `.apply(input, start_pos)`
- `burn::nn::Embedding` — `.forward(input_ids)` where input is Int tensor
- `burn::nn::Conv1d` — `.forward(input)` where input is `[batch, channels, length]`
- `burn::nn::Initializer::Zeros.init(shape, device)` — returns `Param<Tensor>`
- `burn::nn::Initializer::Ones.init(shape, device)`

### Conv1d for Qwen3.5/Nemotron
Both models use short Conv1d (kernel=4) in their SSM blocks. Note Conv1d expects `[batch, channels, length]` format (channels-first), so we need to transpose from `[batch, length, channels]`.

## Weight Loading Strategy (Phase 4)

Burn modules have public fields. We can construct modules with `Initializer::Zeros`, then directly assign loaded weights to the `Param<Tensor>` fields. The `Param` type wraps tensors and can be created from raw tensors:

```rust
use burn::module::{Param, ParamId};
let tensor: Tensor<B, 2> = load_from_safetensors(...);
let param = Param::initialized(ParamId::new(), tensor);
module.weight = param;
```

## Important Notes
- Tensors are consumed by most operations — `.clone()` before reuse
- `#[derive(Module)]` auto-implements serialization/deserialization
- `#[derive(Config)]` generates builder pattern with `.with_*()` methods
- Backend is generic: `B: Backend` — NdArray for CPU, Wgpu for GPU
- Use `burn::prelude::*` for common imports
- The `forward` method is by convention, not trait-enforced
