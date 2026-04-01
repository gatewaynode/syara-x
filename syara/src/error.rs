use thiserror::Error;

#[derive(Debug, Error)]
pub enum SyaraError {
    #[error("file not found: {0}")]
    FileNotFound(String),

    #[error("parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

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
