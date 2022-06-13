pub mod ast;
pub mod lexer;
pub mod token;

use crate::parser::ast::{precedence_of, Expression, Infix, Literal, Program, Statement};
use crate::parser::lexer::Lexer;
use crate::parser::token::{Token, TokenType};

use self::ast::{Identifier, Postfix, PostfixModifier, Precedence, Prefix};

pub struct Parser {
    l: Lexer,
    cur_token: Option<Token>,
    cur_token_type: TokenType,
    peek_token: Option<Token>,
    peek_token_type: TokenType,
    errors: Vec<String>,
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

impl Parser {
    pub fn new(lex: Lexer) -> Self {
        let mut p = Self {
            l: lex,
            cur_token: None,
            peek_token: None,
            cur_token_type: TokenType::EOF,
            peek_token_type: TokenType::EOF,
            errors: vec![],
        };
        p.next_token();
        p.next_token();
        p
    }

    fn next_token(&mut self) {
        self.cur_token = self.peek_token.take();
        if self.cur_token.is_some() {
            self.cur_token_type = self.cur_token.as_ref().unwrap().token_type.clone();
        }
        self.peek_token = Some(self.l.next_token());
        if self.peek_token.is_some() {
            self.peek_token_type = self.peek_token.as_ref().unwrap().token_type.clone();
        }
    }

    pub fn build_ast(&mut self) -> Program {
        let mut p = vec![];
        // let mut cur_tok = self.cur_token_type.clone();

        while self.cur_token_type != TokenType::EOF {
            let stmt = self.parse_statement();
            if let Some(statement) = stmt {
                p.push(statement);
            }
            self.next_token();
            // cur_tok = self.cur_token_type.clone();
        }

        p
    }

    fn parse_statement(&mut self) -> Option<Statement> {
        match self.peek_token_type {
            TokenType::ASSIGN => self.parse_let_statement(),
            // TokenType::RETURN => self.parse_return_statement(),
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_statement(&mut self) -> Option<Statement> {
        // match self.cur_token.as_ref().unwrap().token_type {
        //     TokenType::IDENTIFIER(_) => self.next_token(),
        //     _ => {
        //         self.peek_error(TokenType::IDENTIFIER(String::from("any")));
        //         return None;
        //     }
        // }

        let ident = match self.parse_ident() {
            Some(name) => name,
            None => return None,
        };

        if !self.expect_peek(TokenType::ASSIGN) {
            return None;
        }

        self.next_token();

        let expression = match self.parse_expression(Precedence::Lowest) {
            Some(expr) => Some(expr),
            None => {
                self.errors.push(format!(
                    "could not parse assign statement: '{}'",
                    self.cur_token.as_ref().unwrap().lexeme,
                    // self.cur_token.as_ref().unwrap().line
                ));
                return None;
            }
        };

        if self.peek_token.is_some()
            && self.peek_token.as_ref().unwrap().token_type == TokenType::SEMICOLON
        {
            self.next_token();
        }

        Some(Statement::Let(ident, expression.unwrap()))
    }

    // fn parse_return_statement(&mut self) -> Option<Statement> {
    //     self.next_token();

    //     let expression = match self.parse_expression(Precedence::Lowest) {
    //         Some(expr) => expr,
    //         None => return None,
    //     };

    //     if self.peek_token == TokenType::SEMICOLON {
    //         self.next_token();
    //     }

    //     Some(Statement::Return(expression))
    // }

    fn parse_expression_statement(&mut self) -> Option<Statement> {
        let expression = self.parse_expression(Precedence::Lowest);
        // let expression = self.parse_expression();
        if self.peek_token_type == TokenType::SEMICOLON {
            self.next_token();
        }
        if expression.is_none() {
            // TODO:
            // self.errorserrors
            self.errors.push(format!(
                "could not parse assign statement: '{}'",
                self.cur_token.as_ref().unwrap().lexeme,
                // self.cur_token.as_ref().unwrap().line
            ));
            return None;
        }
        Some(Statement::Expression(expression.unwrap()))
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Option<Expression> {
        // println!("cur {:?} peek {:?}", self.cur_token, self.peek_token);

        let mut left_expression = match self.cur_token_type {
            TokenType::IDENTIFIER(_) => self.parse_ident_expression(),
            TokenType::NUMBER(_) => self.parse_number_expression(),
            TokenType::STRING(_) => self.parse_string_expression(),
            TokenType::BOOL(_) => self.parse_bool_expr(),
            TokenType::LBRACKET => self.parse_array_ident_expr(),
            // TokenType::LBRACE => self.parse_hash_expr(),
            TokenType::MINUS => self.parse_prefix_expression(),
            TokenType::LPAREN => self.parse_grouped_expr(),
            // TokenType::IF => self.parse_if_expression(),
            // TokenType::FUNCTION => self.parse_func_expression(),
            _ => {
                // self.error_no_prefix_Parser();
                None
            }
        };

        while self.peek_token_type != TokenType::SEMICOLON
            && precedence < precedence_of(self.peek_token_type.clone())
        {
            match self.peek_token_type {
                TokenType::PLUS
                | TokenType::MINUS
                | TokenType::ASTERISK
                | TokenType::SLASH
                // | TokenType::WITHOUT
                // | TokenType::FLOORMIN
                // | TokenType::CEILMAX
                | TokenType::AND
                | TokenType::OR
                | TokenType::EQUAL
                | TokenType::LESSTHAN
                | TokenType::LESSTHANEQUAL
                | TokenType::GREATERTHAN
                | TokenType::GREATERTHANEQUAL
                // | TokenType::REDUCE
                // | TokenType::SWAP
                // | TokenType::FIRSTPICK
                // | TokenType::MAGMOD
                // | TokenType::EQ
                // | TokenType::NE
                // | TokenType::GTE
                // | TokenType::GT
                // | TokenType::LT
                // | TokenType::LTE
                => {
                    self.next_token();
                    left_expression = self.parse_infix_expression(left_expression.unwrap());
                }
                // TokenType::TALLY => {self.parse_postfix_expression(left_expression.unwrap());}
                TokenType::LBRACKET => {
                    self.next_token();
                    left_expression = self.parse_index_expr(left_expression.unwrap());
                }
                TokenType::REDUCE => {
                    left_expression = self.parse_postfix_expression(left_expression.unwrap());
                }
                // TokenType::LPAREN => {
                //     self.next_token();
                //     left_expression = self.parse_call_expression(left_expression.unwrap());
                // }
                _ => return left_expression,
            }
        }
        left_expression
    }

    // fn parse_call_expression(&mut self, func: Expression) -> Option<Expression> {
    //     let args = match self.parse_expression_list(TokenType::RPAREN) {
    //         Some(args) => args,
    //         None => return None,
    //     };

    //     Some(Expression::Call {
    //         func: Box::new(func),
    //         args,
    //     })
    // }

    // fn parse_func_expression(&mut self) -> Option<Expression> {
    //     if !self.expect_peek(TokenType::LPAREN) {
    //         return None;
    //     }

    //     let params = match self.parse_func_params() {
    //         Some(params) => params,
    //         None => return None,
    //     };

    //     if !self.expect_peek(TokenType::LBRACE) {
    //         return None;
    //     }

    //     Some(Expression::Func {
    //         params,
    //         body: self.parse_block_statement(),
    //         name: String::from(""),
    //     })
    // }

    // fn parse_hash_expr(&mut self) -> Option<Expression> {
    //     let mut pairs = Vec::new();

    //     while self.peek_token != TokenType::RBRACE {
    //         self.next_token();

    //         let key = match self.parse_expression(Precedence::Lowest) {
    //             Some(expr) => expr,
    //             None => return None,
    //         };

    //         if !self.expect_peek(TokenType::COLON) {
    //             return None;
    //         }

    //         self.next_token();

    //         let value = match self.parse_expression(Precedence::Lowest) {
    //             Some(expr) => expr,
    //             None => return None,
    //         };

    //         pairs.push((key, value));

    //         if self.peek_token != TokenType::RBRACE && !self.expect_peek(TokenType::COMMA) {
    //             return None;
    //         }
    //     }

    //     if !self.expect_peek(TokenType::RBRACE) {
    //         return None;
    //     }

    //     Some(Expression::Literal(Literal::Hash(pairs)))
    // }

    fn parse_array_ident_expr(&mut self) -> Option<Expression> {
        self.parse_expression_list(TokenType::RBRACKET)
            .map(|l| Expression::Literal(Literal::Array(l)))
    }

    fn parse_index_expr(&mut self, left: Expression) -> Option<Expression> {
        self.next_token();

        let index = match self.parse_expression(Precedence::Lowest) {
            Some(expr) => expr,
            None => return None,
        };

        if !self.expect_peek(TokenType::RBRACKET) {
            return None;
        }

        Some(Expression::Index(Box::new(left), Box::new(index)))
    }

    // fn parse_func_params(&mut self) -> Option<Vec<Ident>> {
    //     let mut params = vec![];
    //     if self.peek_token == TokenType::RPAREN {
    //         self.next_token();
    //         return Some(params);
    //     }

    //     self.next_token();

    //     match self.parse_ident() {
    //         Some(ident) => params.push(ident),
    //         None => return None,
    //     }

    //     while self.peek_token == TokenType::COMMA {
    //         self.next_token();
    //         self.next_token();

    //         match self.parse_ident() {
    //             Some(ident) => params.push(ident),
    //             None => return None,
    //         }
    //     }

    //     if !self.expect_peek(TokenType::RPAREN) {
    //         return None;
    //     }

    //     Some(params)
    // }

    fn parse_expression_list(&mut self, end: TokenType) -> Option<Vec<Expression>> {
        let mut list = vec![];

        if self.peek_token.as_ref().unwrap().token_type == end {
            self.next_token();
            return Some(list);
        }

        self.next_token();

        match self.parse_expression(Precedence::Lowest) {
            Some(expr) => list.push(expr),
            None => return None,
        }

        while self.peek_token.as_ref().unwrap().token_type == TokenType::COMMA {
            self.next_token();
            self.next_token();

            match self.parse_expression(Precedence::Lowest) {
                Some(expr) => list.push(expr),
                None => return None,
            }
        }

        if !self.expect_peek(end) {
            return None;
        }

        Some(list)
    }

    // fn parse_block_statement(&mut self) -> BlockStatement {
    //     self.next_token();
    //     let mut block = vec![];
    //     while self.cur_token != TokenType::RBRACE && self.cur_token != TokenType::EOF {
    //         match self.parse_statement() {
    //             Some(stmt) => block.push(stmt),
    //             None => {}
    //         }
    //         self.next_token();
    //     }

    //     block
    // }

    // fn parse_if_expression(&mut self) -> Option<Expression> {
    //     if !self.expect_peek(TokenType::LPAREN) {
    //         return None;
    //     }

    //     self.next_token();

    //     let cond = match self.parse_expression(Precedence::Lowest) {
    //         Some(expr) => expr,
    //         None => return None,
    //     };

    //     if !self.expect_peek(TokenType::RPAREN) || !self.expect_peek(TokenType::LBRACE) {
    //         return None;
    //     }

    //     let consequence = self.parse_block_statement();
    //     let mut alternative = None;

    //     if self.peek_token == TokenType::ELSE {
    //         self.next_token();
    //         if !self.expect_peek(TokenType::LBRACE) {
    //             return None;
    //         }

    //         alternative = Some(self.parse_block_statement());
    //     }

    //     Some(Expression::If {
    //         condition: Box::new(cond),
    //         consequence,
    //         alternative,
    //     })
    // }

    fn token_to_infix(tt: TokenType) -> Option<Infix> {
        match tt {
            TokenType::PLUS => Some(Infix::Plus),
            TokenType::MINUS => Some(Infix::Minus),
            TokenType::ASTERISK => Some(Infix::Multiply),
            TokenType::SLASH => Some(Infix::Divide),
            // TokenType::WITHOUT => Some(Infix::Without), // not sure if this will be a problem :)
            // TokenType::FIRSTPICK => Some(Infix::FirstPick),
            // TokenType::FLOORMIN => Some(Infix::FloorMin),
            // TokenType::CEILMAX => Some(Infix::CeilMax),
            // TokenType::MAGMOD => Some(Infix::MagMod),
            TokenType::AND => Some(Infix::And),
            TokenType::OR => Some(Infix::Or),
            TokenType::EQUAL => Some(Infix::Equal),
            TokenType::GREATERTHAN => Some(Infix::GreaterThan),
            TokenType::GREATERTHANEQUAL => Some(Infix::GreaterThanEqual),
            TokenType::LESSTHAN => Some(Infix::LessThan),
            TokenType::LESSTHANEQUAL => Some(Infix::LessThanEqual),
            _ => None,
        }
    }

    fn consume_infix_modifiers(&mut self) -> Option<Infix> {
        match self.peek_token_type {
            TokenType::PLUS
            | TokenType::MINUS
            | TokenType::ASTERISK
            | TokenType::SLASH
            // | TokenType::FIRSTPICK
            // | TokenType::MAGMOD
            // | TokenType::WITHOUT
             => {
                self.next_token();
                Some(
                    Parser::token_to_infix(self.cur_token_type.clone())
                        .expect("saw a token, but ended up getting none on infix conversion"),
                )
            }
            // TokenType::SWAP => {
            //     self.next_token();
            //     self.consume_infix_modifiers()
            //         .map(|infix| Infix::Modifier(InfixModifier::Swap, Box::new(infix)))
            // }
            _ => None,
        }
    }

    fn parse_infix_expression(&mut self, left: Expression) -> Option<Expression> {
        let infix = match self.cur_token_type {
            TokenType::PLUS
            | TokenType::MINUS
            | TokenType::ASTERISK
            | TokenType::SLASH
            // | TokenType::FIRSTPICK
            // | TokenType::FLOORMIN
            // | TokenType::CEILMAX
            // | TokenType::MAGMOD
            | TokenType::AND
            | TokenType::OR
            | TokenType::EQUAL
            | TokenType::LESSTHAN
            | TokenType::LESSTHANEQUAL
            | TokenType::GREATERTHAN
            | TokenType::GREATERTHANEQUAL
            // | TokenType::WITHOUT 
            => match Parser::token_to_infix(self.cur_token_type.clone()) {
                Some(infix) => infix,
                None => return None,
            },
            // TokenType::REDUCE => {
            //     let result_infix = self
            //         .consume_infix_modifiers()
            //         .expect("expected infix::modifier to return");
            //     // let modifying = match self.peek_token_type.clone() {
            //     //     TokenType::PLUS | TokenType::MINUS | TokenType::ASTERISK | TokenType::SLASH => {
            //     //         Parser::token_to_infix()
            //     //     }
            //     //     None => return None,
            //     // };
            //     // let mut result_infix = Infix::Modifier(InfixModifier::Swap, Box::new(modifying));
            //     // while let Some(nextToken) = Parser::token_to_infix(self.peek_token_type) {
            //     //     self.next_token();
            //     // }

            //     Infix::Modifier(InfixModifier::Reduce, Box::new(result_infix))
            // }
            // TokenType::SWAP => {
            //     let result_infix = self
            //         .consume_infix_modifiers()
            //         .expect("expected infix::modifier to return");
            //     // let modifying = match self.peek_token_type.clone() {
            //     //     TokenType::PLUS | TokenType::MINUS | TokenType::ASTERISK | TokenType::SLASH => {
            //     //         Parser::token_to_infix()
            //     //     }
            //     //     None => return None,
            //     // };
            //     // let mut result_infix = Infix::Modifier(InfixModifier::Swap, Box::new(modifying));
            //     // while let Some(nextToken) = Parser::token_to_infix(self.peek_token_type) {
            //     //     self.next_token();
            //     // }

            //     Infix::Modifier(InfixModifier::Swap, Box::new(result_infix))
            // }
            // TokenType::GT => Infix::Gt,
            // TokenType::GTE => Infix::Gte,
            // TokenType::LT => Infix::Lt,
            // TokenType::LTE => Infix::Lte,
            // TokenType::EQ => Infix::Eq,
            // TokenType::NE => Infix::Ne,
            _ => return None,
        };

        // if infix != Infix::Reduce {
        let p = precedence_of(self.cur_token_type.clone());
        self.next_token();
        self.parse_expression(p).map(|expression| Expression::Infix(
                infix,
                Box::new(left),
                Box::new(expression),
            ))
        // } else {
        //     // reduce case
        //     self.next_token();
        //     match self.cur_token_type {
        //         TokenType::ASTERISK | TokenType::MINUS | TokenType::PLUS | TokenType::SLASH => {
        //             let symbol = self.cur_token_type.clone();
        //             // self.next_token();

        //             Some(Expression::Infix(
        //                 infix,
        //                 Box::new(left),
        //                 Box::new(Expression::Literal(Literal::Symbol(symbol))),
        //             ))
        //         }
        //         // TODO: throw error for incompatible reduce
        //         _ => None,
        //     }
        // }
    }

    fn parse_ident(&mut self) -> Option<Identifier> {
        match self.cur_token_type {
            TokenType::IDENTIFIER(ref mut ident) => Some(Identifier(ident.clone())),
            _ => None,
        }
    }

    fn parse_ident_expression(&mut self) -> Option<Expression> {
        self.parse_ident().map(Expression::Identifier)
    }

    fn parse_bool_expr(&mut self) -> Option<Expression> {
        match self.cur_token_type {
            TokenType::BOOL(val) => Some(Expression::Literal(Literal::Bool(val))),
            _ => None,
        }
    }

    fn parse_number_expression(&mut self) -> Option<Expression> {
        match self.cur_token_type {
            TokenType::NUMBER(i) => Some(Expression::Literal(Literal::Number(i))),
            _ => None,
        }
    }

    fn parse_grouped_expr(&mut self) -> Option<Expression> {
        self.next_token();

        let exp = self.parse_expression(Precedence::Lowest);

        if !self.expect_peek(TokenType::RPAREN) {
            return None;
        }

        exp
    }

    fn parse_postfix_expression(&mut self, left: Expression) -> Option<Expression> {
        self.next_token();
        match self.cur_token_type {
            // TokenType::TALLY => Some(Expression::Postfix(Postfix::Tally, Box::new(left))),
            TokenType::REDUCE => match Parser::token_to_infix(self.peek_token_type.clone()) {
                Some(_) => {
                    self.next_token();
                    Some(Expression::Postfix(
                        Postfix::Modifier(
                            PostfixModifier::Reduce,
                            Box::new(Expression::Literal(Literal::Symbol(
                                self.cur_token_type.clone(),
                            ))),
                        ),
                        Box::new(left),
                    ))
                }
                // TODO: throw error on unexpected token for reduce?
                None => None,
            },
            _ => None,
        }
    }

    fn parse_prefix_expression(&mut self) -> Option<Expression> {
        let prefix = match self.cur_token_type {
            // TokenType::TALLY => Prefix::Tally,
            TokenType::MINUS => Prefix::Minus,
            _ => return None,
        };
        self.next_token();
        self.parse_expression(Precedence::Prefix)
            .map(|expression| Expression::Prefix(prefix, Box::new(expression)))
    }

    fn parse_string_expression(&mut self) -> Option<Expression> {
        match self.cur_token_type {
            TokenType::STRING(ref mut s) => Some(Expression::Literal(Literal::String(s.clone()))),
            _ => None,
        }
    }

    fn expect_peek(&mut self, t: TokenType) -> bool {
        if variant_eq(&t, &self.peek_token_type) {
            self.next_token();
            true
        } else {
            self.peek_error(t);
            false
        }
    }

    pub fn errors(&self) -> Vec<String> {
        self.errors.clone()
    }

    fn peek_error(&mut self, t: TokenType) {
        self.errors.push(format!(
            "expected token to be {:?}, but got {:?}",
            t,
            self.peek_token_type,
            // self.peek_token.as_ref().unwrap().line
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::*;

    #[test]
    fn test_let_statement() {
        let input = String::from("a <- 10");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Let(
            Identifier(String::from("a")),
            Expression::Literal(Literal::Number(10.0)),
        )];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_minus_prefix_statement() {
        let input = String::from("-10");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Prefix(
            Prefix::Minus,
            Box::new(Expression::Literal(Literal::Number(10.0))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_double_assign() {
        let input = String::from("a <- <-");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![];

        println!("errors: {:?}", errors);
        assert_eq!(errors.len(), 1, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_simple_multiply() {
        let input = String::from("10*2");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::Multiply,
            Box::new(Expression::Literal(Literal::Number(10.0))),
            Box::new(Expression::Literal(Literal::Number(2.0))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_plus_arrays_statement() {
        let input = String::from("[1,2,3]+[4,5,6]");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::Plus,
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(1.0)),
                Expression::Literal(Literal::Number(2.0)),
                Expression::Literal(Literal::Number(3.0)),
            ]))),
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(4.0)),
                Expression::Literal(Literal::Number(5.0)),
                Expression::Literal(Literal::Number(6.0)),
            ]))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_lessthan_statement() {
        let input = String::from("[1,2,3]<[4,5,6]");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::LessThan,
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(1.0)),
                Expression::Literal(Literal::Number(2.0)),
                Expression::Literal(Literal::Number(3.0)),
            ]))),
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(4.0)),
                Expression::Literal(Literal::Number(5.0)),
                Expression::Literal(Literal::Number(6.0)),
            ]))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_greaterthan_statement() {
        let input = String::from("[1,2,3]>[4,5,6]");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::GreaterThan,
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(1.0)),
                Expression::Literal(Literal::Number(2.0)),
                Expression::Literal(Literal::Number(3.0)),
            ]))),
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(4.0)),
                Expression::Literal(Literal::Number(5.0)),
                Expression::Literal(Literal::Number(6.0)),
            ]))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_greaterthanequal_statement() {
        let input = String::from("[1,2,3]>=[4,5,6]");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::GreaterThanEqual,
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(1.0)),
                Expression::Literal(Literal::Number(2.0)),
                Expression::Literal(Literal::Number(3.0)),
            ]))),
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(4.0)),
                Expression::Literal(Literal::Number(5.0)),
                Expression::Literal(Literal::Number(6.0)),
            ]))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_lessthanequal_statement() {
        let input = String::from("[1,2,3]<=[4,5,6]");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::LessThanEqual,
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(1.0)),
                Expression::Literal(Literal::Number(2.0)),
                Expression::Literal(Literal::Number(3.0)),
            ]))),
            Box::new(Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Number(4.0)),
                Expression::Literal(Literal::Number(5.0)),
                Expression::Literal(Literal::Number(6.0)),
            ]))),
        ))];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_arithmetic_precedence() {
        let input = String::from("5 + 6 * 7 / 8 - 9");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
            Infix::Minus,
            Box::new(Expression::Infix(
                Infix::Divide,
                Box::new(Expression::Infix(
                    Infix::Multiply,
                    Box::new(Expression::Infix(
                        Infix::Plus,
                        Box::new(Expression::Literal(Literal::Number(5.0))),
                        Box::new(Expression::Literal(Literal::Number(6.0))),
                    )),
                    Box::new(Expression::Literal(Literal::Number(7.0))),
                )),
                Box::new(Expression::Literal(Literal::Number(8.0))),
            )),
            Box::new(Expression::Literal(Literal::Number(9.0))),
        ))];

        //      let expected_ast: Vec<Statement> = vec![Statement::Expression(Expression::Infix(
        //          Infix::Plus,
        //          Box::new(Expression::Literal(Literal::Number(5))),
        //          Box::new(Expression::Infix(
        //              Infix::Multiply,
        //              Box::new(Expression::Literal(Literal::Number(6))),
        //              Box::new(Expression::Infix(
        //                  Infix::Divide,
        //                  Box::new(Expression::Literal(Literal::Number(7))),
        //                  Box::new(Expression::Infix(
        //                      Infix::Minus,
        //                      Box::new(Expression::Literal(Literal::Number(8))),
        //                      Box::new(Expression::Literal(Literal::Number(9))),
        //                  )),
        //              )),
        //          )),
        //      ))];

        assert_eq!(errors.len(), 0);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_array() {
        let input = String::from("a <- [10, 20, 30];a[1]");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![
            Statement::Let(
                Identifier(String::from("a")),
                Expression::Literal(Literal::Array(vec![
                    Expression::Literal(Literal::Number(10.0)),
                    Expression::Literal(Literal::Number(20.0)),
                    Expression::Literal(Literal::Number(30.0)),
                ])),
            ),
            Statement::Expression(Expression::Index(
                Box::new(Expression::Identifier(Identifier(String::from("a")))),
                Box::new(Expression::Literal(Literal::Number(1.0))),
            )),
        ];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_reduce() {
        let input =
            String::from("[10, 20, 30]\\+;[10, 20, 30]\\-;[10, 20, 30]\\*;[10, 20, 30]\\/;");
        let l = Lexer::new(input);
        let mut parser = Parser::new(l);
        let ast = parser.build_ast();
        let errors = parser.errors();

        let expected_ast: Vec<Statement> = vec![
            Statement::Expression(Expression::Postfix(
                Postfix::Modifier(
                    PostfixModifier::Reduce,
                    Box::new(Expression::Literal(Literal::Symbol(TokenType::PLUS))),
                ),
                Box::new(Expression::Literal(Literal::Array(vec![
                    Expression::Literal(Literal::Number(10.0)),
                    Expression::Literal(Literal::Number(20.0)),
                    Expression::Literal(Literal::Number(30.0)),
                ]))),
            )),
            Statement::Expression(Expression::Postfix(
                Postfix::Modifier(
                    PostfixModifier::Reduce,
                    Box::new(Expression::Literal(Literal::Symbol(TokenType::MINUS))),
                ),
                Box::new(Expression::Literal(Literal::Array(vec![
                    Expression::Literal(Literal::Number(10.0)),
                    Expression::Literal(Literal::Number(20.0)),
                    Expression::Literal(Literal::Number(30.0)),
                ]))),
            )),
            Statement::Expression(Expression::Postfix(
                Postfix::Modifier(
                    PostfixModifier::Reduce,
                    Box::new(Expression::Literal(Literal::Symbol(TokenType::ASTERISK))),
                ),
                Box::new(Expression::Literal(Literal::Array(vec![
                    Expression::Literal(Literal::Number(10.0)),
                    Expression::Literal(Literal::Number(20.0)),
                    Expression::Literal(Literal::Number(30.0)),
                ]))),
            )),
            Statement::Expression(Expression::Postfix(
                Postfix::Modifier(
                    PostfixModifier::Reduce,
                    Box::new(Expression::Literal(Literal::Symbol(TokenType::SLASH))),
                ),
                Box::new(Expression::Literal(Literal::Array(vec![
                    Expression::Literal(Literal::Number(10.0)),
                    Expression::Literal(Literal::Number(20.0)),
                    Expression::Literal(Literal::Number(30.0)),
                ]))),
            )),
        ];

        assert_eq!(errors.len(), 0, "{:?}", errors);

        assert_eq!(ast, expected_ast);
    }
}
