//! Mamba2 SSM block for Nemotron hybrid models.
//!
//! Implements the Mamba2 selective state-space model in recurrent (sequential)
//! mode. No CUDA kernels or chunk-based scanning — just a simple loop over
//! timesteps, which is sufficient for the short sequences (~100 tokens) used
//! in YARA LLM evaluation.

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig1d};
use burn::prelude::*;
use burn::tensor::activation::silu;

// ── MambaRMSNormGated ──────────────────────────────────────────────────────

/// Grouped RMS normalization with optional sigmoid gating.
///
/// Uses `norm_before_gate=false` (Nemotron convention):
/// `output = rms_norm_grouped(x * silu(gate)) * weight`
#[derive(Module, Debug)]
pub struct MambaRMSNormGated<B: Backend> {
    pub(crate) weight: Param<Tensor<B, 1>>,
    group_size: usize,
    eps: f64,
}

/// Configuration for [`MambaRMSNormGated`].
#[derive(Config, Debug)]
pub struct MambaRMSNormGatedConfig {
    pub hidden_size: usize,
    pub group_size: usize,
    #[config(default = "1e-5")]
    pub eps: f64,
}

impl MambaRMSNormGatedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaRMSNormGated<B> {
        MambaRMSNormGated {
            weight: Initializer::Ones.init([self.hidden_size], device),
            group_size: self.group_size,
            eps: self.eps,
        }
    }
}

impl<B: Backend> MambaRMSNormGated<B> {
    /// Forward: `rms_norm_grouped(x * silu(gate)) * weight`.
    ///
    /// `x`: `[batch, seq, hidden]`, `gate`: same shape (optional).
    pub fn forward(&self, x: Tensor<B, 3>, gate: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        let n_groups = hidden / self.group_size;

        // Gate first (norm_before_gate=false)
        let x = match gate {
            Some(z) => x * silu(z),
            None => x,
        };

        // Grouped RMS norm: reshape → normalize per group → reshape back
        let x = x.reshape([batch * seq, n_groups, self.group_size]);
        let variance = x.clone().powf_scalar(2.0).mean_dim(2); // [bs, ng, 1]
        let rms = (variance + self.eps).sqrt();
        let x = (x / rms).reshape([batch, seq, hidden]);

        // Apply learnable weight [hidden] broadcast over [batch, seq, hidden]
        let w = self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        x * w
    }
}

// ── Mamba2Block ────────────────────────────────────────────────────────────

/// Configuration for [`Mamba2Block`].
#[derive(Config, Debug)]
pub struct Mamba2Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
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
}

impl Mamba2Config {
    fn intermediate_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    fn conv_dim(&self) -> usize {
        self.intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }

    fn projection_size(&self) -> usize {
        self.intermediate_size() + self.conv_dim() + self.num_heads
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Block<B> {
        let intermediate = self.intermediate_size();
        let conv_dim = self.conv_dim();
        let proj_size = self.projection_size();

        let in_proj = LinearConfig::new(self.d_model, proj_size)
            .with_bias(self.use_bias)
            .init(device);

        // Depthwise causal conv1d (groups=conv_dim, no padding — we pad manually)
        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, self.conv_kernel)
            .with_groups(conv_dim)
            .with_bias(self.use_conv_bias)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);

        let a_log = Initializer::Zeros.init([self.num_heads], device);
        let d_param = Initializer::Ones.init([self.num_heads], device);
        let dt_bias = Initializer::Ones.init([self.num_heads], device);

        let norm = MambaRMSNormGatedConfig::new(intermediate, intermediate / self.n_groups)
            .with_eps(self.rms_norm_eps)
            .init(device);

        let out_proj = LinearConfig::new(intermediate, self.d_model)
            .with_bias(self.use_bias)
            .init(device);

        Mamba2Block {
            in_proj,
            conv1d,
            a_log,
            d_param,
            dt_bias,
            norm,
            out_proj,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            n_groups: self.n_groups,
            conv_kernel: self.conv_kernel,
            ssm_state_size: self.ssm_state_size,
            intermediate_size: intermediate,
            conv_dim,
        }
    }
}

/// Mamba2 selective state-space block (sequential/recurrent mode).
///
/// Input/output: `[batch, seq_len, d_model]`.
#[derive(Module, Debug)]
pub struct Mamba2Block<B: Backend> {
    pub(crate) in_proj: Linear<B>,
    pub(crate) conv1d: Conv1d<B>,
    pub(crate) a_log: Param<Tensor<B, 1>>,
    pub(crate) d_param: Param<Tensor<B, 1>>,
    pub(crate) dt_bias: Param<Tensor<B, 1>>,
    pub(crate) norm: MambaRMSNormGated<B>,
    pub(crate) out_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
    ssm_state_size: usize,
    intermediate_size: usize,
    conv_dim: usize,
    conv_kernel: usize,
}

impl<B: Backend> Mamba2Block<B> {
    /// Forward pass over a full sequence (no KV cache).
    ///
    /// `x`: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();
        let device = x.device();
        let inter = self.intermediate_size;
        let gs = self.n_groups * self.ssm_state_size;

        // 1. Project: [batch, seq, d_model] → [batch, seq, proj_size]
        let proj = self.in_proj.forward(x);

        // Split: gate [inter] | conv_input [conv_dim] | dt [num_heads]
        let gate = proj.clone().narrow(2, 0, inter);
        let conv_input = proj.clone().narrow(2, inter, self.conv_dim);
        let dt_raw = proj.narrow(2, inter + self.conv_dim, self.num_heads);

        // 2. Causal conv1d: pad left, convolve, silu
        // conv_input: [batch, seq, conv_dim] → transpose → [batch, conv_dim, seq]
        let ci = conv_input.swap_dims(1, 2);
        // Left-pad with kernel_size-1 zeros for causal convolution
        let pad = Tensor::<B, 3>::zeros(
            [batch, self.conv_dim, self.conv_kernel - 1],
            &device,
        );
        let ci_padded = Tensor::cat(vec![pad, ci], 2);
        let conv_out = silu(self.conv1d.forward(ci_padded)); // [batch, conv_dim, seq]
        let conv_out = conv_out.swap_dims(1, 2); // [batch, seq, conv_dim]

        // Split conv output: x_ssm [inter] | B [gs] | C [gs]
        let x_ssm = conv_out.clone().narrow(2, 0, inter);
        let b_raw = conv_out.clone().narrow(2, inter, gs);
        let c_raw = conv_out.narrow(2, inter + gs, gs);

        // 3. SSM recurrence over the sequence
        let y = self.ssm_scan(x_ssm, b_raw, c_raw, dt_raw, batch, seq_len, &device);

        // 4. Gated norm + output projection
        let y = self.norm.forward(y, Some(gate));
        self.out_proj.forward(y)
    }

    /// Sequential SSM scan (no chunking).
    ///
    /// Processes each timestep with the Mamba2 recurrence:
    /// ```text
    /// dt = softplus(dt_raw + dt_bias)
    /// dA = exp(A * dt)
    /// state = dA * state + dt * B * x
    /// y = (state * C).sum(state_dim) + D * x
    /// ```
    fn ssm_scan(
        &self,
        x_ssm: Tensor<B, 3>,  // [batch, seq, inter]
        b_raw: Tensor<B, 3>,  // [batch, seq, n_groups*state]
        c_raw: Tensor<B, 3>,  // [batch, seq, n_groups*state]
        dt_raw: Tensor<B, 3>, // [batch, seq, num_heads]
        batch: usize,
        seq_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let nh = self.num_heads;
        let hd = self.head_dim;
        let ng = self.n_groups;
        let ns = self.ssm_state_size;
        let heads_per_group = nh / ng;

        // Precompute A = -exp(A_log): [num_heads]
        let a = self.a_log.val().exp().neg();

        // dt_bias: [num_heads]
        let bias = self.dt_bias.val();

        // D: [num_heads]
        let d_skip = self.d_param.val();

        // Reshape inputs for per-head processing
        // x: [batch, seq, nh, hd]
        let x_4d = x_ssm.reshape([batch, seq_len, nh, hd]);
        // B: [batch, seq, ng, ns] → expand to [batch, seq, nh, ns]
        let b_4d = b_raw.reshape([batch, seq_len, ng, ns]);
        let b_4d = b_4d
            .unsqueeze_dim::<5>(3)                           // [b, s, ng, 1, ns]
            .expand([batch, seq_len, ng, heads_per_group, ns])
            .reshape([batch, seq_len, nh, ns]);
        // C: same expansion
        let c_4d = c_raw.reshape([batch, seq_len, ng, ns]);
        let c_4d = c_4d
            .unsqueeze_dim::<5>(3)
            .expand([batch, seq_len, ng, heads_per_group, ns])
            .reshape([batch, seq_len, nh, ns]);
        // dt: [batch, seq, nh] + bias → softplus
        let dt = softplus(dt_raw + bias.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0));

        // State: [batch, nh, hd, ns] — zero-initialized
        let mut state = Tensor::<B, 4>::zeros([batch, nh, hd, ns], device);

        // Collect output per timestep
        let mut y_steps: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Slice timestep t
            let x_t = x_4d.clone().narrow(1, t, 1).squeeze_dim::<3>(1); // [b, nh, hd]
            let b_t = b_4d.clone().narrow(1, t, 1).squeeze_dim::<3>(1); // [b, nh, ns]
            let c_t = c_4d.clone().narrow(1, t, 1).squeeze_dim::<3>(1); // [b, nh, ns]
            let dt_t = dt.clone().narrow(1, t, 1).squeeze_dim::<2>(1);  // [b, nh]

            // dA = exp(A * dt): [b, nh] → expand to [b, nh, hd, ns]
            let a_dt = a.clone().unsqueeze_dim::<2>(0) * dt_t.clone(); // [b, nh]
            let da = a_dt
                .unsqueeze_dim::<3>(2)
                .unsqueeze_dim::<4>(3)
                .expand([batch, nh, hd, ns])
                .exp();

            // dB*x: dt[b,nh,1,1] * B[b,nh,1,ns] * x[b,nh,hd,1] → [b,nh,hd,ns]
            let dt_e = dt_t
                .unsqueeze_dim::<3>(2)
                .unsqueeze_dim::<4>(3); // [b,nh,1,1]
            let b_e = b_t.unsqueeze_dim::<4>(2);       // [b,nh,1,ns]
            let x_e = x_t.clone().unsqueeze_dim::<4>(3); // [b,nh,hd,1]
            let dbx = dt_e * b_e * x_e;

            // State update
            state = da * state + dbx;

            // Output: y = (state * C).sum(ns) + D * x
            let c_e = c_t.unsqueeze_dim::<4>(2); // [b,nh,1,ns]
            let y_t = (state.clone() * c_e).sum_dim(3).squeeze_dim::<3>(3); // [b,nh,hd]

            // D skip connection
            let d_e = d_skip
                .clone()
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(2); // [1,nh,1]
            let y_t = y_t + d_e * x_t; // [b,nh,hd]

            y_steps.push(y_t);
        }

        // Stack: list of [b,nh,hd] → [b,seq,nh,hd] → [b,seq,inter]
        let y: Tensor<B, 4> = Tensor::stack(y_steps, 1);
        y.reshape([batch, seq_len, self.intermediate_size])
    }
}

/// Softplus: ln(1 + exp(x)), with numerical stability.
fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // For large x, softplus(x) ≈ x (avoids exp overflow)
    // For small x, compute ln(1 + exp(x))
    // Simple implementation: always compute ln(1+exp(x)); f32 range is sufficient
    // for our use case (dt values are typically small).
    (x.exp() + 1.0).log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn mamba_rms_norm_gated_shape() {
        let device = Default::default();
        let norm = MambaRMSNormGatedConfig::new(64, 16).init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [2, 4, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let gate = Tensor::<B, 3>::random(
            [2, 4, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = norm.forward(x.clone(), Some(gate));
        assert_eq!(out.dims(), [2, 4, 64]);

        // Without gate
        let out_no_gate = norm.forward(x, None);
        assert_eq!(out_no_gate.dims(), [2, 4, 64]);
    }

    #[test]
    fn mamba2_block_shape() {
        let device = Default::default();
        let config = Mamba2Config {
            d_model: 64,
            num_heads: 4,
            head_dim: 8,
            n_groups: 2,
            ssm_state_size: 16,
            conv_kernel: 4,
            use_conv_bias: true,
            use_bias: false,
            rms_norm_eps: 1e-5,
        };
        // intermediate = 4*8 = 32
        // conv_dim = 32 + 2*2*16 = 96
        // proj = 32 + 96 + 4 = 132
        let block = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [1, 8, 64],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        let out = block.forward(x);
        assert_eq!(out.dims(), [1, 8, 64]);
    }

    #[test]
    fn mamba2_single_token() {
        let device = Default::default();
        let config = Mamba2Config {
            d_model: 32,
            num_heads: 2,
            head_dim: 4,
            n_groups: 1,
            ssm_state_size: 8,
            conv_kernel: 4,
            use_conv_bias: true,
            use_bias: false,
            rms_norm_eps: 1e-5,
        };
        let block = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [1, 1, 32],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        let out = block.forward(x);
        assert_eq!(out.dims(), [1, 1, 32]);
    }
}
