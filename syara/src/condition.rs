/// Boolean condition AST and evaluator.
///
/// Replaces Python's `eval()` approach with a typed AST.
///
/// Grammar:
/// ```text
/// expr     = or_expr
/// or_expr  = and_expr ('or' and_expr)*
/// and_expr = not_expr ('and' not_expr)*
/// not_expr = 'not' primary | primary
/// primary  = '$' ident
///          | 'any' 'of' set
///          | 'all' 'of' set
///          | '(' expr ')'
/// set      = 'them'
///          | '(' '$' ident (',' '$' ident)* ')'
///          | '(' '$' prefix '*' ')'
/// ```
use std::collections::HashMap;
use crate::error::SyaraError;
use crate::models::MatchDetail;

#[derive(Debug, Clone)]
pub enum Expr {
    Identifier(String),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    AnyOf(SetExpr),
    AllOf(SetExpr),
}

#[derive(Debug, Clone)]
pub enum SetExpr {
    Them,
    Explicit(Vec<String>),
    Wildcard(String), // prefix before '*'
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),   // $name
    Keyword(String), // and, or, not, any, all, of, them
    LParen,
    RParen,
    Comma,
    Star,
    Eof,
}

struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek_char() {
            self.pos += c.len_utf8();
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek_char().map(|c| c.is_whitespace()).unwrap_or(false) {
            self.advance();
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        match self.peek_char() {
            None => Token::Eof,
            Some('(') => { self.advance(); Token::LParen }
            Some(')') => { self.advance(); Token::RParen }
            Some(',') => { self.advance(); Token::Comma }
            Some('*') => { self.advance(); Token::Star }
            Some('$') => {
                self.advance();
                let mut name = String::from("$");
                while self.peek_char().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) {
                    name.push(self.peek_char().unwrap());
                    self.advance();
                }
                Token::Ident(name)
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                let mut word = String::new();
                while self.peek_char().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) {
                    word.push(self.peek_char().unwrap());
                    self.advance();
                }
                Token::Keyword(word)
            }
            Some(c) => {
                // skip unknown character
                self.advance();
                let mut s = String::new();
                s.push(c);
                Token::Keyword(s)
            }
        }
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn tokenize(input: &str) -> Vec<Token> {
        let mut t = Tokenizer::new(input);
        let mut tokens = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            tokens.push(tok);
            if done { break; }
        }
        tokens
    }

    fn new(input: &str) -> Self {
        Self {
            tokens: Self::tokenize(input),
            pos: 0,
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn consume(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn consume_keyword(&mut self, kw: &str) -> Result<(), SyaraError> {
        match self.consume() {
            Token::Keyword(k) if k == kw => Ok(()),
            other => Err(SyaraError::ConditionParse(format!(
                "expected '{}', got {:?}",
                kw, other
            ))),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, SyaraError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, SyaraError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Token::Keyword(k) if k == "or") {
            self.consume();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, SyaraError> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Token::Keyword(k) if k == "and") {
            self.consume();
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, SyaraError> {
        if matches!(self.peek(), Token::Keyword(k) if k == "not") {
            self.consume();
            let inner = self.parse_primary()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, SyaraError> {
        match self.peek().clone() {
            Token::Ident(id) => {
                self.consume();
                Ok(Expr::Identifier(id))
            }
            Token::LParen => {
                self.consume();
                let inner = self.parse_expr()?;
                match self.consume() {
                    Token::RParen => Ok(inner),
                    other => Err(SyaraError::ConditionParse(format!(
                        "expected ')', got {:?}",
                        other
                    ))),
                }
            }
            Token::Keyword(ref k) if k == "any" => {
                self.consume();
                self.consume_keyword("of")?;
                let set = self.parse_set()?;
                Ok(Expr::AnyOf(set))
            }
            Token::Keyword(ref k) if k == "all" => {
                self.consume();
                self.consume_keyword("of")?;
                let set = self.parse_set()?;
                Ok(Expr::AllOf(set))
            }
            other => Err(SyaraError::ConditionParse(format!(
                "unexpected token {:?}",
                other
            ))),
        }
    }

    fn parse_set(&mut self) -> Result<SetExpr, SyaraError> {
        if matches!(self.peek(), Token::Keyword(k) if k == "them") {
            self.consume();
            return Ok(SetExpr::Them);
        }

        match self.consume() {
            Token::LParen => {}
            other => {
                return Err(SyaraError::ConditionParse(format!(
                    "expected '(' or 'them' in set, got {:?}",
                    other
                )))
            }
        }

        // Peek ahead: is this a wildcard pattern `$prefix*`?
        if let Token::Ident(id) = self.peek().clone() {
            // Check if followed by Star
            let next_pos = self.pos + 1;
            if self.tokens.get(next_pos) == Some(&Token::Star) {
                self.consume(); // consume ident
                self.consume(); // consume *
                match self.consume() {
                    Token::RParen => {}
                    other => {
                        return Err(SyaraError::ConditionParse(format!(
                            "expected ')' after wildcard, got {:?}",
                            other
                        )))
                    }
                }
                // Strip the leading '$' to get the prefix
                let prefix = id.trim_start_matches('$').to_owned();
                return Ok(SetExpr::Wildcard(prefix));
            }
        }

        // Explicit list: `($s1, $s2, ...)`
        let mut ids = Vec::new();
        loop {
            match self.consume() {
                Token::Ident(id) => ids.push(id),
                other => {
                    return Err(SyaraError::ConditionParse(format!(
                        "expected identifier in set, got {:?}",
                        other
                    )))
                }
            }
            match self.peek() {
                Token::Comma => { self.consume(); }
                Token::RParen => { self.consume(); break; }
                other => {
                    return Err(SyaraError::ConditionParse(format!(
                        "expected ',' or ')' in set, got {:?}",
                        other.clone()
                    )))
                }
            }
        }

        Ok(SetExpr::Explicit(ids))
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a condition string into an AST.
pub fn parse(condition: &str) -> Result<Expr, SyaraError> {
    let mut parser = Parser::new(condition);
    let expr = parser.parse_expr()?;
    Ok(expr)
}

/// Evaluate a condition against the current pattern match results.
/// `matches` maps identifier → list of match details (non-empty ⟹ matched).
pub fn evaluate(
    expr: &Expr,
    matches: &HashMap<String, Vec<MatchDetail>>,
) -> bool {
    match expr {
        Expr::Identifier(id) => {
            matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
        }
        Expr::Not(inner) => !evaluate(inner, matches),
        Expr::And(l, r) => evaluate(l, matches) && evaluate(r, matches),
        Expr::Or(l, r) => evaluate(l, matches) || evaluate(r, matches),
        Expr::AnyOf(set) => {
            let ids = resolve_set(set, matches);
            ids.iter().any(|id| {
                matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
            })
        }
        Expr::AllOf(set) => {
            let ids = resolve_set(set, matches);
            if ids.is_empty() {
                return false;
            }
            ids.iter().all(|id| {
                matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
            })
        }
    }
}

fn resolve_set(set: &SetExpr, matches: &HashMap<String, Vec<MatchDetail>>) -> Vec<String> {
    match set {
        SetExpr::Them => matches.keys().cloned().collect(),
        SetExpr::Explicit(ids) => ids.clone(),
        SetExpr::Wildcard(prefix) => {
            let full_prefix = format!("${}", prefix);
            matches
                .keys()
                .filter(|k| k.starts_with(&full_prefix))
                .cloned()
                .collect()
        }
    }
}

/// Optimistic short-circuit: would the condition be true if `identifier` matched?
/// Used to skip expensive LLM calls when they cannot change the outcome.
pub fn is_identifier_needed(
    identifier: &str,
    expr: &Expr,
    current_matches: &HashMap<String, Vec<MatchDetail>>,
) -> bool {
    let mut test = current_matches.clone();
    test.entry(identifier.to_owned())
        .or_default()
        .push(MatchDetail::new(identifier, ""));
    evaluate(expr, &test)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> HashMap<String, Vec<MatchDetail>> {
        HashMap::new()
    }

    fn hit(id: &str) -> HashMap<String, Vec<MatchDetail>> {
        let mut m = HashMap::new();
        m.insert(id.to_owned(), vec![MatchDetail::new(id, "x")]);
        m
    }

    fn hits(ids: &[&str]) -> HashMap<String, Vec<MatchDetail>> {
        let mut m = HashMap::new();
        for id in ids {
            m.insert(id.to_string(), vec![MatchDetail::new(*id, "x")]);
        }
        m
    }

    #[test]
    fn test_single_identifier() {
        let expr = parse("$s1").unwrap();
        assert!(!evaluate(&expr, &empty()));
        assert!(evaluate(&expr, &hit("$s1")));
    }

    #[test]
    fn test_and() {
        let expr = parse("$s1 and $s2").unwrap();
        assert!(!evaluate(&expr, &hit("$s1")));
        assert!(evaluate(&expr, &hits(&["$s1", "$s2"])));
    }

    #[test]
    fn test_or() {
        let expr = parse("$s1 or $s2").unwrap();
        assert!(evaluate(&expr, &hit("$s1")));
        assert!(!evaluate(&expr, &empty()));
    }

    #[test]
    fn test_not() {
        let expr = parse("not $s1").unwrap();
        assert!(evaluate(&expr, &empty()));
        assert!(!evaluate(&expr, &hit("$s1")));
    }

    #[test]
    fn test_any_of_them() {
        let expr = parse("any of them").unwrap();
        assert!(!evaluate(&expr, &empty()));
        assert!(evaluate(&expr, &hit("$s1")));
    }

    #[test]
    fn test_all_of_them() {
        let expr = parse("all of them").unwrap();
        let mut m = HashMap::new();
        m.insert("$s1".to_owned(), vec![MatchDetail::new("$s1", "x")]);
        m.insert("$s2".to_owned(), vec![]); // registered but not matched
        assert!(!evaluate(&expr, &m));

        let m2 = hits(&["$s1", "$s2"]);
        assert!(evaluate(&expr, &m2));
    }

    #[test]
    fn test_wildcard_set() {
        let expr = parse("any of ($dan*)").unwrap();
        let m = hits(&["$dan1", "$dan2"]);
        assert!(evaluate(&expr, &m));

        let expr2 = parse("all of ($dan*)").unwrap();
        let mut partial = HashMap::new();
        partial.insert("$dan1".to_owned(), vec![MatchDetail::new("$dan1", "x")]);
        partial.insert("$dan2".to_owned(), vec![]);
        assert!(!evaluate(&expr2, &partial));
    }

    #[test]
    fn test_nested_parens() {
        let expr = parse("$s1 and ($s2 or $s3)").unwrap();
        assert!(!evaluate(&expr, &hit("$s1")));
        assert!(evaluate(&expr, &hits(&["$s1", "$s3"])));
    }
}
