/// Validates parsed rules and constructs a `CompiledRules` instance.
use std::collections::HashSet;
use crate::compiled_rules::CompiledRules;
use crate::config::Registry;
use crate::condition;
use crate::error::SyaraError;
use crate::models::Rule;

pub struct Compiler;

impl Compiler {
    pub fn compile(rules: Vec<Rule>, registry: Registry) -> Result<CompiledRules, SyaraError> {
        for rule in &rules {
            Self::validate(rule)?;
        }
        Ok(CompiledRules::new(rules, registry))
    }

    fn validate(rule: &Rule) -> Result<(), SyaraError> {
        // Collect all declared identifiers
        let mut declared: HashSet<&str> = HashSet::new();

        for r in &rule.strings {
            if !declared.insert(r.identifier.as_str()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.similarity {
            if !declared.insert(r.identifier.as_str()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.phash {
            if !declared.insert(r.identifier.as_str()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.classifier {
            if !declared.insert(r.identifier.as_str()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.llm {
            if !declared.insert(r.identifier.as_str()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }

        // Verify condition parses cleanly
        if !rule.condition.is_empty() {
            let expr = condition::parse(&rule.condition).map_err(|e| {
                SyaraError::ParseError {
                    line: 0,
                    message: format!("rule '{}' condition: {}", rule.name, e),
                }
            })?;

            // Check that all $identifiers referenced in condition are declared.
            // We do a simple text scan for $word tokens rather than AST walking,
            // which is sufficient and avoids an extra AST traversal.
            let id_re = regex::Regex::new(r"\$\w+").unwrap();
            let cond = &rule.condition;
            for m in id_re.find_iter(cond) {
                // Skip wildcard prefixes: `$prefix*` inside `any of ($prefix*)`
                if cond[m.end()..].starts_with('*') {
                    continue;
                }
                let id = m.as_str();
                if !declared.contains(id) {
                    return Err(SyaraError::UndefinedIdentifier {
                        identifier: id.to_owned(),
                        rule: rule.name.clone(),
                    });
                }
            }

            let _ = expr; // validated above
        }

        Ok(())
    }
}
