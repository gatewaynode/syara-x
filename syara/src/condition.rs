/// Condition AST and evaluator.
///
/// Replaces Python's `eval()` approach with a typed AST.
///
/// Grammar:
/// ```text
/// expr       = or_expr
/// or_expr    = and_expr ('or' and_expr)*
/// and_expr   = not_expr ('and' not_expr)*
/// not_expr   = 'not' cmp_expr | cmp_expr
/// cmp_expr   = add_expr (cmp_op add_expr)?        // non-associative, one cmp max
/// add_expr   = unary (('+'|'-') unary)*
/// unary      = '-' unary | primary
/// primary    = int_lit
///            | count_expr
///            | identifier
///            | '(' expr ')'
///            | 'any' 'of' set
///            | 'all' 'of' set
/// count_expr = '#' ident                          // stored as "$ident"
/// identifier = '$' ident
/// cmp_op     = '==' | '!=' | '<' | '<=' | '>' | '>='
/// set        = 'them'
///            | '(' '$' ident (',' '$' ident)* ')'
///            | '(' '$' prefix '*' ')'
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
    /// `#ident` — number of matches for the pattern. Stored with `$`-prefix
    /// so it shares the pattern-map key with `Identifier`.
    Count(String),
    IntLit(i64),
    Cmp(Box<Expr>, CmpOp, Box<Expr>),
    BinOp(Box<Expr>, ArithOp, Box<Expr>),
    Neg(Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum SetExpr {
    Them,
    Explicit(Vec<String>),
    Wildcard(String), // prefix before '*'
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp { Add, Sub }

/// Runtime value during evaluation. Private: the typed AST + post-parse
/// `type_check` guarantees `evaluate` always returns a `Bool` at the top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value {
    Bool(bool),
    Int(i64),
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),   // $name — stored with the leading '$'
    Count(String),   // #name — stored with a leading '$' (sigil-normalized)
    Keyword(String), // and, or, not, any, all, of, them
    Int(i64),
    Cmp(CmpOp),
    Plus,
    Minus,
    LParen,
    RParen,
    Comma,
    Star,
    Unknown(char), // BUG-022: unrecognized characters
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

    fn peek_char_at(&self, offset: usize) -> Option<char> {
        self.input[self.pos..].chars().nth(offset)
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
            Some('+') => { self.advance(); Token::Plus }
            Some('-') => { self.advance(); Token::Minus }
            Some('=') => {
                // '==' only; bare '=' is unknown.
                if self.peek_char_at(1) == Some('=') {
                    self.advance(); self.advance();
                    Token::Cmp(CmpOp::Eq)
                } else {
                    self.advance();
                    Token::Unknown('=')
                }
            }
            Some('!') => {
                // '!=' only; bare '!' is unknown (matches existing BUG-022 surface).
                if self.peek_char_at(1) == Some('=') {
                    self.advance(); self.advance();
                    Token::Cmp(CmpOp::Ne)
                } else {
                    self.advance();
                    Token::Unknown('!')
                }
            }
            Some('<') => {
                if self.peek_char_at(1) == Some('=') {
                    self.advance(); self.advance();
                    Token::Cmp(CmpOp::Le)
                } else {
                    self.advance();
                    Token::Cmp(CmpOp::Lt)
                }
            }
            Some('>') => {
                if self.peek_char_at(1) == Some('=') {
                    self.advance(); self.advance();
                    Token::Cmp(CmpOp::Ge)
                } else {
                    self.advance();
                    Token::Cmp(CmpOp::Gt)
                }
            }
            Some('$') => {
                self.advance();
                let mut name = String::from("$");
                while self.peek_char().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) {
                    name.push(self.peek_char().unwrap());
                    self.advance();
                }
                Token::Ident(name)
            }
            Some('#') => {
                // #ident — store with '$' prefix so it shares the pattern-map key.
                self.advance();
                if !self.peek_char().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) {
                    // Lone '#' or '# foo' — reject as Unknown.
                    return Token::Unknown('#');
                }
                let mut name = String::from("$");
                while self.peek_char().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) {
                    name.push(self.peek_char().unwrap());
                    self.advance();
                }
                Token::Count(name)
            }
            Some(c) if c.is_ascii_digit() => {
                let mut digits = String::new();
                while self.peek_char().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    digits.push(self.peek_char().unwrap());
                    self.advance();
                }
                match digits.parse::<i64>() {
                    Ok(n) => Token::Int(n),
                    Err(_) => Token::Unknown(digits.chars().next().unwrap_or('0')),
                }
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
                // BUG-022: produce Unknown token instead of silently treating as keyword
                self.advance();
                Token::Unknown(c)
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
            // Preserves existing "no `not not x`" restriction: parse_cmp never
            // produces a Not, so `not` cannot chain without parentheses.
            let inner = self.parse_cmp()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Expr, SyaraError> {
        let left = self.parse_add()?;
        if let Token::Cmp(op) = *self.peek() {
            self.consume();
            let right = self.parse_add()?;
            // Non-associative: a second comparison is a parse error.
            if let Token::Cmp(_) = *self.peek() {
                return Err(SyaraError::ConditionParse(
                    "chained comparisons are not supported; use 'and' to combine".to_owned()
                ));
            }
            return Ok(Expr::Cmp(Box::new(left), op, Box::new(right)));
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Expr, SyaraError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Plus => ArithOp::Add,
                Token::Minus => ArithOp::Sub,
                _ => break,
            };
            self.consume();
            let right = self.parse_unary()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, SyaraError> {
        if matches!(self.peek(), Token::Minus) {
            self.consume();
            let inner = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, SyaraError> {
        match self.peek().clone() {
            Token::Ident(id) => {
                self.consume();
                Ok(Expr::Identifier(id))
            }
            Token::Count(id) => {
                self.consume();
                Ok(Expr::Count(id))
            }
            Token::Int(n) => {
                self.consume();
                Ok(Expr::IntLit(n))
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

// ── Type check ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty { Bool, Int }

fn type_of(expr: &Expr) -> Result<Ty, SyaraError> {
    match expr {
        Expr::Identifier(_) | Expr::AnyOf(_) | Expr::AllOf(_) => Ok(Ty::Bool),
        Expr::Count(_) | Expr::IntLit(_) => Ok(Ty::Int),
        Expr::Not(inner) => {
            if type_of(inner)? != Ty::Bool {
                return Err(SyaraError::ConditionParse(
                    "type error: 'not' expects a boolean operand".to_owned()
                ));
            }
            Ok(Ty::Bool)
        }
        Expr::And(l, r) | Expr::Or(l, r) => {
            if type_of(l)? != Ty::Bool || type_of(r)? != Ty::Bool {
                return Err(SyaraError::ConditionParse(
                    "type error: 'and' / 'or' expect boolean operands".to_owned()
                ));
            }
            Ok(Ty::Bool)
        }
        Expr::Cmp(l, _, r) => {
            if type_of(l)? != Ty::Int || type_of(r)? != Ty::Int {
                return Err(SyaraError::ConditionParse(
                    "type error: comparison operators expect integer operands".to_owned()
                ));
            }
            Ok(Ty::Bool)
        }
        Expr::BinOp(l, _, r) => {
            if type_of(l)? != Ty::Int || type_of(r)? != Ty::Int {
                return Err(SyaraError::ConditionParse(
                    "type error: arithmetic operators expect integer operands".to_owned()
                ));
            }
            Ok(Ty::Int)
        }
        Expr::Neg(inner) => {
            if type_of(inner)? != Ty::Int {
                return Err(SyaraError::ConditionParse(
                    "type error: unary '-' expects an integer operand".to_owned()
                ));
            }
            Ok(Ty::Int)
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a condition string into an AST.
pub fn parse(condition: &str) -> Result<Expr, SyaraError> {
    let mut parser = Parser::new(condition);
    let expr = parser.parse_expr()?;
    if *parser.peek() != Token::Eof {
        return Err(SyaraError::ConditionParse(format!(
            "unexpected trailing token {:?}",
            parser.peek()
        )));
    }
    let ty = type_of(&expr)?;
    if ty != Ty::Bool {
        return Err(SyaraError::ConditionParse(
            "type error: condition must evaluate to a boolean".to_owned()
        ));
    }
    Ok(expr)
}

/// Evaluate a condition against the current pattern match results.
/// `matches` maps identifier → list of match details (non-empty ⟹ matched).
///
/// `type_of` at parse time guarantees the top-level is boolean; if
/// something slipped through (programmatic construction of `Expr`), the
/// defensive default here returns `false`.
pub fn evaluate(
    expr: &Expr,
    matches: &HashMap<String, Vec<MatchDetail>>,
) -> bool {
    match eval_val(expr, matches) {
        Value::Bool(b) => b,
        Value::Int(_) => false,
    }
}

fn eval_val(expr: &Expr, matches: &HashMap<String, Vec<MatchDetail>>) -> Value {
    match expr {
        Expr::Identifier(id) => Value::Bool(
            matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
        ),
        Expr::Count(id) => Value::Int(
            matches.get(id).map(|v| v.len() as i64).unwrap_or(0)
        ),
        Expr::IntLit(n) => Value::Int(*n),
        Expr::Not(inner) => Value::Bool(!expect_bool(eval_val(inner, matches))),
        Expr::Neg(inner) => Value::Int(
            expect_int(eval_val(inner, matches)).wrapping_neg()
        ),
        Expr::And(l, r) => Value::Bool(
            expect_bool(eval_val(l, matches)) && expect_bool(eval_val(r, matches))
        ),
        Expr::Or(l, r) => Value::Bool(
            expect_bool(eval_val(l, matches)) || expect_bool(eval_val(r, matches))
        ),
        Expr::Cmp(l, op, r) => Value::Bool(apply_cmp(
            *op,
            expect_int(eval_val(l, matches)),
            expect_int(eval_val(r, matches)),
        )),
        Expr::BinOp(l, op, r) => Value::Int(apply_arith(
            *op,
            expect_int(eval_val(l, matches)),
            expect_int(eval_val(r, matches)),
        )),
        Expr::AnyOf(set) => {
            let ids = resolve_set(set, matches);
            Value::Bool(ids.iter().any(|id| {
                matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
            }))
        }
        Expr::AllOf(set) => {
            let ids = resolve_set(set, matches);
            if ids.is_empty() {
                return Value::Bool(false);
            }
            Value::Bool(ids.iter().all(|id| {
                matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
            }))
        }
    }
}

fn expect_bool(v: Value) -> bool {
    match v { Value::Bool(b) => b, Value::Int(_) => false }
}

fn expect_int(v: Value) -> i64 {
    match v { Value::Int(n) => n, Value::Bool(_) => 0 }
}

fn apply_cmp(op: CmpOp, l: i64, r: i64) -> bool {
    match op {
        CmpOp::Eq => l == r,
        CmpOp::Ne => l != r,
        CmpOp::Lt => l < r,
        CmpOp::Le => l <= r,
        CmpOp::Gt => l > r,
        CmpOp::Ge => l >= r,
    }
}

fn apply_arith(op: ArithOp, l: i64, r: i64) -> i64 {
    match op {
        ArithOp::Add => l.wrapping_add(r),
        ArithOp::Sub => l.wrapping_sub(r),
    }
}

fn resolve_set(set: &SetExpr, matches: &HashMap<String, Vec<MatchDetail>>) -> Vec<String> {
    match set {
        SetExpr::Them => {
            let mut keys: Vec<String> = matches.keys().cloned().collect();
            keys.sort();
            keys
        }
        SetExpr::Explicit(ids) => ids.clone(),
        SetExpr::Wildcard(prefix) => {
            let full_prefix = format!("${}", prefix);
            let mut keys: Vec<String> = matches
                .keys()
                .filter(|k| k.starts_with(&full_prefix))
                .cloned()
                .collect();
            keys.sort();
            keys
        }
    }
}

/// Optimistic short-circuit: would the condition be true if `identifier` matched?
/// Used to skip expensive LLM calls when they cannot change the outcome.
///
/// With integer expressions in the tree, "matched" does not imply a specific
/// count value. If `identifier` appears inside any `Count` subtree we cannot
/// reason about it without actually running the matcher — return `true`
/// (pessimistic: don't skip). Otherwise the existing boolean hypothetical
/// evaluator is sound: substitute `Identifier(extra_id) → true` and evaluate
/// `Count`/arithmetic/comparisons against the current `matches` map.
#[cfg(any(feature = "llm", feature = "burn-llm"))]
pub fn is_identifier_needed(
    identifier: &str,
    expr: &Expr,
    current_matches: &HashMap<String, Vec<MatchDetail>>,
) -> bool {
    if mentions_count_of(expr, identifier) {
        return true;
    }
    match eval_hypothetical_val(expr, current_matches, identifier) {
        Value::Bool(b) => b,
        Value::Int(_) => false,
    }
}

/// Walks the tree checking whether `id` appears as `Expr::Count(id)` anywhere.
#[cfg(any(feature = "llm", feature = "burn-llm"))]
fn mentions_count_of(expr: &Expr, id: &str) -> bool {
    match expr {
        Expr::Count(c) => c == id,
        Expr::Identifier(_) | Expr::IntLit(_) | Expr::AnyOf(_) | Expr::AllOf(_) => false,
        Expr::Not(inner) | Expr::Neg(inner) => mentions_count_of(inner, id),
        Expr::And(l, r) | Expr::Or(l, r) | Expr::Cmp(l, _, r) | Expr::BinOp(l, _, r) => {
            mentions_count_of(l, id) || mentions_count_of(r, id)
        }
    }
}

/// Evaluate as if `extra_id` were matched, without cloning the map.
/// Guarded by the early `mentions_count_of` check: this function never sees
/// `extra_id` inside a `Count` subtree.
#[cfg(any(feature = "llm", feature = "burn-llm"))]
fn eval_hypothetical_val(
    expr: &Expr,
    matches: &HashMap<String, Vec<MatchDetail>>,
    extra_id: &str,
) -> Value {
    match expr {
        Expr::Identifier(id) => Value::Bool(
            id == extra_id || matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
        ),
        Expr::Count(id) => Value::Int(
            matches.get(id).map(|v| v.len() as i64).unwrap_or(0)
        ),
        Expr::IntLit(n) => Value::Int(*n),
        Expr::Not(inner) => Value::Bool(
            !expect_bool(eval_hypothetical_val(inner, matches, extra_id))
        ),
        Expr::Neg(inner) => Value::Int(
            expect_int(eval_hypothetical_val(inner, matches, extra_id)).wrapping_neg()
        ),
        Expr::And(l, r) => Value::Bool(
            expect_bool(eval_hypothetical_val(l, matches, extra_id))
                && expect_bool(eval_hypothetical_val(r, matches, extra_id))
        ),
        Expr::Or(l, r) => Value::Bool(
            expect_bool(eval_hypothetical_val(l, matches, extra_id))
                || expect_bool(eval_hypothetical_val(r, matches, extra_id))
        ),
        Expr::Cmp(l, op, r) => Value::Bool(apply_cmp(
            *op,
            expect_int(eval_hypothetical_val(l, matches, extra_id)),
            expect_int(eval_hypothetical_val(r, matches, extra_id)),
        )),
        Expr::BinOp(l, op, r) => Value::Int(apply_arith(
            *op,
            expect_int(eval_hypothetical_val(l, matches, extra_id)),
            expect_int(eval_hypothetical_val(r, matches, extra_id)),
        )),
        Expr::AnyOf(set) => {
            let ids = resolve_set(set, matches);
            Value::Bool(ids.iter().any(|id| {
                id == extra_id || matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
            }))
        }
        Expr::AllOf(set) => {
            let ids = resolve_set(set, matches);
            if ids.is_empty() {
                return Value::Bool(false);
            }
            Value::Bool(ids.iter().all(|id| {
                id == extra_id || matches.get(id).map(|v| !v.is_empty()).unwrap_or(false)
            }))
        }
    }
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

    fn counted(id: &str, n: usize) -> HashMap<String, Vec<MatchDetail>> {
        let mut m = HashMap::new();
        m.insert(id.to_owned(), (0..n).map(|_| MatchDetail::new(id, "x")).collect());
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
        m.insert("$s2".to_owned(), vec![]);
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

    // ── BUG-006: trailing tokens must produce an error ──────────────────────

    #[test]
    fn test_trailing_tokens_error() {
        let result = parse("$s1 $s2");
        assert!(result.is_err(), "trailing token $s2 should cause parse error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("trailing token"), "error should mention trailing: {msg}");
    }

    #[test]
    fn test_trailing_keyword_error() {
        let result = parse("$s1 and $s2 not");
        assert!(result.is_err(), "trailing 'not' should cause parse error");
    }

    #[test]
    fn test_trailing_paren_error() {
        let result = parse("$s1 and $s2)");
        assert!(result.is_err(), "unmatched ')' should cause parse error");
    }

    // ── BUG-007: hypothetical evaluation without cloning ────────────────────

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_is_identifier_needed_returns_true() {
        let expr = parse("$s1 and $llm1").unwrap();
        let m = hit("$s1");
        assert!(is_identifier_needed("$llm1", &expr, &m));
    }

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_is_identifier_needed_returns_false() {
        let expr = parse("$s1 and $s2 and $llm1").unwrap();
        let m = hit("$s1");
        assert!(!is_identifier_needed("$llm1", &expr, &m));
    }

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_is_identifier_needed_or_branch() {
        let expr = parse("$s1 or $llm1").unwrap();
        let m = hit("$s1");
        assert!(is_identifier_needed("$llm1", &expr, &m));
    }

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_is_identifier_needed_negated() {
        let expr = parse("not $llm1").unwrap();
        assert!(!is_identifier_needed("$llm1", &expr, &empty()));
    }

    // ── Additional condition coverage ───────────────────────────────────────

    #[test]
    fn test_explicit_set() {
        let expr = parse("any of ($s1, $s2)").unwrap();
        assert!(evaluate(&expr, &hit("$s2")));
        assert!(!evaluate(&expr, &hit("$s3")));
    }

    #[test]
    fn test_all_of_explicit_set() {
        let expr = parse("all of ($s1, $s2)").unwrap();
        assert!(!evaluate(&expr, &hit("$s1")));
        assert!(evaluate(&expr, &hits(&["$s1", "$s2"])));
    }

    #[test]
    fn test_all_of_empty_is_false() {
        let expr = parse("all of them").unwrap();
        assert!(!evaluate(&expr, &empty()));
    }

    #[test]
    fn test_operator_precedence() {
        let expr = parse("$s1 or $s2 and $s3").unwrap();
        assert!(evaluate(&expr, &hit("$s1")));
        assert!(!evaluate(&expr, &hit("$s2")));
        assert!(evaluate(&expr, &hits(&["$s2", "$s3"])));
    }

    #[test]
    fn test_double_not_is_parse_error() {
        // Grammar: not_expr = 'not' cmp_expr | cmp_expr
        // "not not $s1" is not valid — the inner `not` is not a cmp_expr.
        let result = parse("not not $s1");
        assert!(result.is_err());
    }

    #[test]
    fn test_not_with_parens() {
        let expr = parse("not (not $s1)").unwrap();
        assert!(!evaluate(&expr, &empty()));
        assert!(evaluate(&expr, &hit("$s1")));
    }

    #[test]
    fn test_empty_input_error() {
        let result = parse("");
        assert!(result.is_err());
    }

    // ── BUG-022: unknown chars must produce parse error ──────────────────────

    #[test]
    fn test_unknown_char_at_sign_is_error() {
        let result = parse("$s1 and @foo");
        assert!(result.is_err(), "@ should produce a parse error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Unknown"), "error should mention Unknown: {msg}");
    }

    /// Flipped from the previous `test_unknown_char_hash_is_error`: `#ident`
    /// is now a count expression. Bare `#` with no identifier is still rejected.
    #[test]
    fn test_hash_is_count_primary() {
        let expr = parse("#s1 == 0").unwrap();
        match expr {
            Expr::Cmp(l, CmpOp::Eq, r) => {
                match (*l, *r) {
                    (Expr::Count(ref c), Expr::IntLit(0)) if c == "$s1" => {}
                    other => panic!("expected Cmp(Count($s1), Eq, IntLit(0)), got {:?}", other),
                }
            }
            other => panic!("expected Cmp, got {:?}", other),
        }
    }

    #[test]
    fn test_bare_hash_is_error() {
        let result = parse("#");
        assert!(result.is_err(), "bare '#' should produce a parse error");
    }

    #[test]
    fn test_hash_space_ident_is_error() {
        let result = parse("# s1");
        assert!(result.is_err(), "'# s1' (hash + space + ident) should produce a parse error");
    }

    // ── BUG-034: SetExpr::Them returns keys in sorted order ──────────────────

    #[test]
    fn test_them_keys_sorted() {
        let expr = parse("all of them").unwrap();
        let mut m = HashMap::new();
        m.insert("$z".to_owned(), vec![MatchDetail::new("$z", "x")]);
        m.insert("$a".to_owned(), vec![MatchDetail::new("$a", "x")]);
        m.insert("$m".to_owned(), vec![MatchDetail::new("$m", "x")]);
        assert!(evaluate(&expr, &m));

        let set = SetExpr::Them;
        let resolved = super::resolve_set(&set, &m);
        assert_eq!(resolved, vec!["$a", "$m", "$z"]);
    }

    #[test]
    fn test_wildcard_keys_sorted() {
        let set = SetExpr::Wildcard("s".to_owned());
        let mut m = HashMap::new();
        m.insert("$s3".to_owned(), vec![]);
        m.insert("$s1".to_owned(), vec![]);
        m.insert("$s2".to_owned(), vec![]);
        let resolved = super::resolve_set(&set, &m);
        assert_eq!(resolved, vec!["$s1", "$s2", "$s3"]);
    }

    // ── #pattern count operators ─────────────────────────────────────────────

    #[test]
    fn test_count_cmp_parses() {
        let expr = parse("#s1 >= 1").unwrap();
        match expr {
            Expr::Cmp(l, CmpOp::Ge, r) => match (*l, *r) {
                (Expr::Count(c), Expr::IntLit(1)) if c == "$s1" => {}
                other => panic!("unexpected operands: {:?}", other),
            },
            other => panic!("expected Cmp, got {:?}", other),
        }
    }

    #[test]
    fn test_all_cmp_ops_parse() {
        for (input, expected) in [
            ("#s1 == 0", CmpOp::Eq),
            ("#s1 != 0", CmpOp::Ne),
            ("#s1 < 0", CmpOp::Lt),
            ("#s1 <= 0", CmpOp::Le),
            ("#s1 > 0", CmpOp::Gt),
            ("#s1 >= 0", CmpOp::Ge),
        ] {
            let expr = parse(input).unwrap();
            match expr {
                Expr::Cmp(_, op, _) => assert_eq!(op, expected, "wrong op for {input}"),
                other => panic!("expected Cmp for {input}, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_arith_parses() {
        // add_expr binds tighter than cmp: #a + #b >= 2 == Cmp(BinOp(#a, Add, #b), Ge, 2).
        let expr = parse("#a + #b >= 2").unwrap();
        match expr {
            Expr::Cmp(l, CmpOp::Ge, r) => {
                match *l {
                    Expr::BinOp(ll, ArithOp::Add, rr) => {
                        assert!(matches!(*ll, Expr::Count(ref c) if c == "$a"));
                        assert!(matches!(*rr, Expr::Count(ref c) if c == "$b"));
                    }
                    other => panic!("expected BinOp, got {:?}", other),
                }
                assert!(matches!(*r, Expr::IntLit(2)));
            }
            other => panic!("expected Cmp, got {:?}", other),
        }
    }

    #[test]
    fn test_not_over_cmp() {
        let expr = parse("not #s1 >= 2").unwrap();
        assert!(matches!(expr, Expr::Not(inner) if matches!(*inner, Expr::Cmp(..))));
    }

    #[test]
    fn test_non_associative_cmp_error() {
        let result = parse("#a < #b < #c");
        assert!(result.is_err(), "chained comparisons should be rejected");
    }

    #[test]
    fn test_trailing_arith_error() {
        assert!(parse("#a +").is_err());
    }

    #[test]
    fn test_count_zero_for_declared_unmatched() {
        let expr = parse("#s1 == 0").unwrap();
        let mut m = HashMap::new();
        m.insert("$s1".to_owned(), vec![]);
        assert!(evaluate(&expr, &m));

        let expr2 = parse("#s1 >= 1").unwrap();
        assert!(!evaluate(&expr2, &m));
    }

    #[test]
    fn test_count_for_matched() {
        let m = counted("$s1", 3);
        assert!(evaluate(&parse("#s1 == 3").unwrap(), &m));
        assert!(evaluate(&parse("#s1 >= 2").unwrap(), &m));
        assert!(evaluate(&parse("#s1 < 5").unwrap(), &m));
        assert!(evaluate(&parse("#s1 != 0").unwrap(), &m));
        assert!(!evaluate(&parse("#s1 > 10").unwrap(), &m));
    }

    #[test]
    fn test_count_for_absent_key() {
        // Defensive: key entirely missing from the map → count is 0.
        let expr = parse("#s1 == 0").unwrap();
        assert!(evaluate(&expr, &empty()));
    }

    #[test]
    fn test_sum_counts() {
        let expr = parse("(#a + #b) >= 2").unwrap();

        let mut m_ok = HashMap::new();
        m_ok.insert("$a".to_owned(), vec![MatchDetail::new("$a", "x")]);
        m_ok.insert("$b".to_owned(), vec![MatchDetail::new("$b", "x")]);
        assert!(evaluate(&expr, &m_ok));

        let mut m_no = HashMap::new();
        m_no.insert("$a".to_owned(), vec![MatchDetail::new("$a", "x")]);
        m_no.insert("$b".to_owned(), vec![]);
        assert!(!evaluate(&expr, &m_no));
    }

    #[test]
    fn test_mixed_bool_and_count() {
        let expr = parse("$s1 and #s2 >= 2").unwrap();

        let m_hit = {
            let mut m = counted("$s2", 2);
            m.insert("$s1".to_owned(), vec![MatchDetail::new("$s1", "x")]);
            m
        };
        assert!(evaluate(&expr, &m_hit));

        let mut m_miss = counted("$s2", 2);
        m_miss.insert("$s1".to_owned(), vec![]);
        assert!(!evaluate(&expr, &m_miss));
    }

    #[test]
    fn test_type_error_bool_in_arith() {
        let result = parse("$s1 + 2");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("type error"), "expected 'type error', got: {msg}");
    }

    #[test]
    fn test_type_error_int_in_bool_ctx() {
        let result = parse("#s1 and $s2");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("type error"), "expected 'type error', got: {msg}");
    }

    #[test]
    fn test_top_level_int_is_type_error() {
        let result = parse("#s1");
        assert!(result.is_err(), "top-level integer must be a type error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("type error"), "expected 'type error', got: {msg}");
    }

    #[test]
    fn test_negative_literal_in_cmp() {
        let expr = parse("#s1 >= -1").unwrap();
        assert!(evaluate(&expr, &empty()));
    }

    #[test]
    fn test_cmp_binds_tighter_than_and() {
        let expr = parse("#a >= 1 and #b >= 2").unwrap();
        assert!(matches!(expr, Expr::And(..)));
    }

    // ── is_identifier_needed with count-aware pessimism ──────────────────────

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_inn_pessimistic_when_llm_in_count() {
        // $llm1 appears inside Count → pessimistic, must return true.
        let expr = parse("#llm1 >= 2 or $s1").unwrap();
        let m = empty();
        assert!(is_identifier_needed("$llm1", &expr, &m));
    }

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_inn_recovers_when_count_already_false() {
        // $llm1 is NOT in a Count. Current count of $s1 is 0, so `#s1 >= 2`
        // is already false and the `and` branch makes the whole condition
        // false regardless of $llm1 — skip the LLM.
        let expr = parse("$llm1 and #s1 >= 2").unwrap();
        let mut m = HashMap::new();
        m.insert("$s1".to_owned(), vec![]);
        assert!(!is_identifier_needed("$llm1", &expr, &m));
    }

    #[test]
    #[cfg(any(feature = "llm", feature = "burn-llm"))]
    fn test_inn_recovers_or_branch_count_already_true() {
        // `#s1 >= 2` is already true via actual counts → condition is already
        // satisfied; adding $llm1 doesn't change it. The hypothetical still
        // evaluates to true, so caller's "skip if !needed" logic keeps
        // calling the LLM in this case — matching the existing test contract
        // for `$s1 or $llm1` above.
        let expr = parse("$llm1 or #s1 >= 2").unwrap();
        let m = counted("$s1", 3);
        assert!(is_identifier_needed("$llm1", &expr, &m));
    }
}
