/*!
SYARA-X — Super YARA in Rust.

Extends the YARA rule format with semantic, ML-classifier, and LLM-based matching.

# Example

```rust
use syara_x;

let rules = syara_x::compile_str(r#"
    rule test_rule {
        strings:
            $s1 = "hello world" nocase
        condition:
            $s1
    }
"#).unwrap();

let matches = rules.scan("Hello World");
assert_eq!(matches.iter().filter(|m| m.matched).count(), 1);
```
*/

pub mod cache;
pub mod compiled_rules;
pub mod compiler;
pub mod condition;
pub mod config;
pub mod engine;
pub mod error;
pub mod models;
pub mod parser;

pub use compiled_rules::CompiledRules;
pub use error::SyaraError;
pub use models::{Match, MatchDetail, Rule};

use std::path::Path;

/// Compile rules from a `.syara` file.
pub fn compile(path: impl AsRef<Path>) -> Result<CompiledRules, SyaraError> {
    let rules = parser::SyaraParser::new().parse_file(path)?;
    let registry = config::Registry::new();
    compiler::Compiler::compile(rules, registry)
}

/// Compile rules from a string.
pub fn compile_str(src: &str) -> Result<CompiledRules, SyaraError> {
    let rules = parser::SyaraParser::new().parse_str(src)?;
    let registry = config::Registry::new();
    compiler::Compiler::compile(rules, registry)
}
