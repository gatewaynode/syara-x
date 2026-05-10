use thiserror::Error;

#[derive(Debug, Error)]
pub enum SyaraError {
    #[error("file not found: {0}")]
    FileNotFound(String),

    /// Parse error with line and column. `col` is 1-indexed within the
    /// current line; 0 is a sentinel meaning "no column info available"
    /// (used by error sites that don't have a Scanner in scope, e.g.
    /// rule-header / missing-condition errors that fire after section
    /// parsing). The Display impl omits ", col N" when col is 0.
    #[error("parse error at line {line}{}: {message}", parse_error_position_suffix(*col))]
    ParseError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("duplicate identifier '{0}' in rule '{1}'")]
    DuplicateIdentifier(String, String),

    #[error("undefined identifier '{identifier}' in condition of rule '{rule}'")]
    UndefinedIdentifier { identifier: String, rule: String },

    #[error("invalid pattern '{pattern}': {reason}")]
    InvalidPattern { pattern: String, reason: String },

    #[error("condition parse error: {0}")]
    ConditionParse(String),

    #[error("component '{name}' not found: {kind}")]
    ComponentNotFound { kind: String, name: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("semantic matching error: {0}")]
    SemanticError(String),

    #[error("classifier error: {0}")]
    ClassifierError(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("phash error: {0}")]
    PhashError(String),
}

/// Render `, col N` suffix for `ParseError` when `col > 0`, empty
/// otherwise. Kept out-of-line so the `#[error]` format string stays
/// readable.
fn parse_error_position_suffix(col: usize) -> String {
    if col > 0 {
        format!(", col {col}")
    } else {
        String::new()
    }
}
