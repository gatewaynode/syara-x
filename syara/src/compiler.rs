/// Validates parsed rules and constructs a `CompiledRules` instance.
use std::collections::HashSet;
use crate::compiled_rules::CompiledRules;
use crate::config::Registry;
use crate::condition;
use crate::error::SyaraError;
use crate::models::Rule;

pub struct Compiler;

impl Compiler {
    pub fn compile(mut rules: Vec<Rule>, registry: Registry) -> Result<CompiledRules, SyaraError> {
        for rule in &mut rules {
            Self::validate_and_compile(rule)?;
        }
        Ok(CompiledRules::new(rules, registry))
    }

    fn validate_and_compile(rule: &mut Rule) -> Result<(), SyaraError> {
        // Collect all declared identifiers. Owned String so we can look up
        // against sigil-normalized keys built during condition validation.
        let mut declared: HashSet<String> = HashSet::new();

        for r in &rule.strings {
            if !declared.insert(r.identifier.clone()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.similarity {
            if !declared.insert(r.identifier.clone()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.phash {
            if !declared.insert(r.identifier.clone()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.classifier {
            if !declared.insert(r.identifier.clone()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }
        for r in &rule.llm {
            if !declared.insert(r.identifier.clone()) {
                return Err(SyaraError::DuplicateIdentifier(
                    r.identifier.clone(),
                    rule.name.clone(),
                ));
            }
        }

        // Verify condition parses cleanly
        if !rule.condition.is_empty() {
            // BUG-023: use ConditionParse (not ParseError with line: 0) for
            // condition errors — the compiler doesn't have source line info.
            let expr = condition::parse(&rule.condition).map_err(|e| {
                SyaraError::ConditionParse(format!("rule '{}': {}", rule.name, e))
            })?;

            // Check that all $identifiers and #counts referenced in the condition
            // are declared. Text scan is cheaper than AST walking; both sigils
            // point to the same pattern, so normalize `#name` → `$name` before
            // the declared-set lookup.
            let id_re = regex::Regex::new(r"[#$]\w+").unwrap();
            let cond = &rule.condition;
            for m in id_re.find_iter(cond) {
                // Skip wildcard prefixes: `$prefix*` inside `any of ($prefix*)`.
                if cond[m.end()..].starts_with('*') {
                    continue;
                }
                let id = m.as_str();
                let normalized = if let Some(rest) = id.strip_prefix('#') {
                    format!("${}", rest)
                } else {
                    id.to_owned()
                };
                if !declared.contains(&normalized) {
                    return Err(SyaraError::UndefinedIdentifier {
                        identifier: id.to_owned(),
                        rule: rule.name.clone(),
                    });
                }
            }

            rule.compiled_condition = Some(expr);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Rule, StringRule};
    use std::collections::HashMap;

    fn rule_with(strings: Vec<StringRule>, condition: &str) -> Rule {
        Rule {
            name: "r".to_owned(),
            tags: vec![],
            meta: HashMap::new(),
            strings,
            similarity: vec![],
            phash: vec![],
            classifier: vec![],
            llm: vec![],
            condition: condition.to_owned(),
            compiled_condition: None,
        }
    }

    fn string_rule(id: &str, pattern: &str) -> StringRule {
        StringRule {
            identifier: id.to_owned(),
            pattern: pattern.to_owned(),
            modifiers: vec![],
            is_regex: false,
        }
    }

    #[test]
    fn test_undefined_count_identifier_errors() {
        let mut rule = rule_with(vec![], "#s_missing >= 1");
        let err = Compiler::validate_and_compile(&mut rule).unwrap_err();
        match err {
            SyaraError::UndefinedIdentifier { identifier, .. } => {
                assert_eq!(identifier, "#s_missing");
            }
            other => panic!("expected UndefinedIdentifier, got {:?}", other),
        }
    }

    #[test]
    fn test_defined_count_identifier_ok() {
        let mut rule = rule_with(vec![string_rule("$s1", "hello")], "#s1 >= 1");
        let result = Compiler::validate_and_compile(&mut rule);
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
        assert!(rule.compiled_condition.is_some());
    }

    #[test]
    fn test_undefined_dollar_identifier_still_errors() {
        // Regression: the `[#$]\w+` regex change must not weaken `$`-identifier
        // validation.
        let mut rule = rule_with(vec![], "$s_missing");
        let err = Compiler::validate_and_compile(&mut rule).unwrap_err();
        assert!(matches!(err, SyaraError::UndefinedIdentifier { .. }));
    }
}
