//! Local LLM inference via the Burn deep learning framework.
//!
//! [`BurnEvaluator`] implements [`LLMEvaluator`] using Burn for local model
//! inference, eliminating the need for an external HTTP server. Currently a
//! stub — full implementation in Phase 4.

use crate::engine::llm_evaluator::LLMEvaluator;
use crate::error::SyaraError;

/// Local LLM evaluator backed by the Burn framework.
///
/// Register via [`CompiledRules::register_llm_evaluator`] with a name that
/// `.syara` rules can reference in their `llm_name` field.
///
/// # Example (future, after Phase 4)
///
/// ```ignore
/// let evaluator = BurnEvaluator::from_dir("models/Qwen3.5-0.8B-Base")?;
/// compiled_rules.register_llm_evaluator("qwen3", Box::new(evaluator));
/// ```
pub struct BurnEvaluator {
    _private: (),
}

impl Default for BurnEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl BurnEvaluator {
    /// Create a stub evaluator. Will be replaced with `from_dir` in Phase 4.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl LLMEvaluator for BurnEvaluator {
    fn evaluate(&self, _pattern: &str, _input_text: &str) -> Result<(bool, String), SyaraError> {
        Err(SyaraError::LlmError(
            "BurnEvaluator: model not loaded (stub — see Phase 4)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_error() {
        let evaluator = BurnEvaluator::new();
        let result = evaluator.evaluate("pattern", "text");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not loaded"), "error should mention not loaded: {msg}");
    }

    #[test]
    fn stub_registers_in_registry() {
        use crate::config::Registry;
        let mut registry = Registry::new();
        registry.register_llm_evaluator("burn-test", Box::new(BurnEvaluator::new()));
        let evaluator = registry.get_llm_evaluator("burn-test");
        assert!(evaluator.is_ok(), "stub should register and be retrievable");
    }
}
