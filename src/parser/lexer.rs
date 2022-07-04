use crate::parser::token::{Token, TokenType as TT};

pub struct Lexer {
    input: String,
    position: usize,
    read_pos: usize,
    line: usize,
    column: usize,
    ch: Option<char>,
}

fn is_letter(ch: char) -> bool {
    ('a'..='z').contains(&ch) || ('A'..='Z').contains(&ch) || ch == '_'
}

fn is_digit(ch: char) -> bool {
    ('0'..='9').contains(&ch)
}

impl Lexer {
    pub fn new(input: String) -> Self {
        let mut s = Self {
            input,
            position: 0,
            read_pos: 0,
            line: 0,
            column: 0,
            ch: None,
        };
        s.read_char();
        s
    }

    fn read_char(&mut self) {
        if self.read_pos >= self.input.len() {
            self.ch = None;
        } else {
            match self.input.char_indices().nth(self.read_pos) {
                Some((_, ch)) => {
                    self.ch = Some(ch);
                }
                None => {
                    self.ch = None;
                }
            };
        }
        self.position = self.read_pos;
        self.read_pos += 1;
        self.column += 1;
    }

    fn skip_whitespace(&mut self) {
        while self.ch.is_some() {
            match self.ch.unwrap() {
                ' ' | '\t' | '\r' => {
                    // println!("clearing whitespace");
                    self.read_char();
                }
                '\n' => {
                    self.line += 1;
                    self.column = 0;
                    self.read_char();
                }
                _ => {
                    break;
                }
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        if self.read_pos >= self.input.len() {
            None
        } else {
            self.input.chars().nth(self.read_pos)
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        let mut maintain_ch = false;
        let start_line = self.line;
        let start_pos = if self.position > self.input.len() {
            self.input.len()
        } else {
            self.position
        };
        let start_col = self.column;
        let t = if self.ch.is_none() {
            TT::EOF
        } else {
            match self.ch.unwrap() {
                '=' => TT::EQUAL,
                // ':' => TT::COLON,
                ';' => TT::SEMICOLON,
                '(' => TT::LPAREN,
                ')' => TT::RPAREN,
                '{' => TT::LBRACE,
                '}' => TT::RBRACE,
                '[' => TT::LBRACKET,
                ']' => TT::RBRACKET,
                '+' => TT::PLUS,
                '-' => TT::MINUS,
                '/' => TT::SLASH,
                '*' => TT::ASTERISK,
                '\\' => TT::REDUCE,
                '∧' => TT::AND,
                '∨' => TT::OR,
                '!' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        TT::NOTEQUAL
                    } else {
                        TT::BANG
                    }
                }
                '>' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        TT::GREATERTHANEQUAL
                    } else {
                        TT::GREATERTHAN
                    }
                }
                '<' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '-' {
                        self.read_char();
                        TT::ASSIGN
                    } else if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        TT::LESSTHANEQUAL
                    } else {
                        TT::LESSTHAN
                    }
                }
                '≤' => TT::LESSTHANEQUAL,
                ',' => TT::COMMA,
                '"' => {
                    return self.read_string(start_col, start_line);
                }
                x => {
                    maintain_ch = true;
                    if is_letter(x) {
                        let literal = self.read_identifier();
                        match literal.as_str() {
                            "fn" => TT::FUNCTION,
                            "while" => TT::WHILE,
                            "if" => TT::IF,
                            "else" => TT::ELSE,
                            "piset" => TT::PISET,
                            // "piget" => TT::PIGET,
                            "return" => TT::RETURN,
                            _ => TT::IDENTIFIER(literal),
                        }
                    } else if is_digit(x) {
                        let literal = self.read_number();
                        TT::NUMBER(literal.parse::<f64>().unwrap())
                    } else {
                        TT::ILLEGAL
                    }
                }
            }
        };
        if !maintain_ch {
            self.read_char();
        }

        let end_pos = if self.position > self.input.len() {
            self.input.len()
        } else {
            self.position
        };

        // println!(
        //     "len: {:?}, start pos: {:?}, end_pos: {:?}, token_type: {:?}",
        //     self.input.len(),
        //     start_pos,
        //     end_pos,
        //     t,
        // );

        // println!("lexeme: {:?}", self.input[start_pos..end_pos].to_string(),);
        Token {
            lexeme: self
                .input
                .char_indices()
                .skip(start_pos)
                .take(end_pos - start_pos)
                .map(|(_, c)| c)
                .collect::<String>(),
            token_type: t,
        }
    }

    fn read_string(&mut self, _start_col: usize, _start_line: usize) -> Token {
        self.read_char();

        let spos = self.position;
        loop {
            match self.ch {
                Some('"') => {
                    let literal = self.input[spos..self.position].to_string();
                    self.read_char();
                    return Token {
                        token_type: TT::STRING(literal.clone()),
                        lexeme: literal,
                    };
                }
                Some('\n') => {
                    self.line += 1;
                    self.read_char();
                }
                None => {
                    return Token {
                        token_type: TT::ILLEGAL,
                        lexeme: String::from("\""),
                    }
                }
                _ => {
                    self.read_char();
                }
            }
        }
    }

    fn read_identifier(&mut self) -> String {
        let pos = self.position;
        while self.ch.is_some() && is_letter(self.ch.unwrap()) {
            self.read_char();
        }
        return self
            .input
            .char_indices()
            .skip(pos)
            .take(self.position - pos)
            .map(|(_, c)| c)
            .collect::<String>();
    }

    fn read_number(&mut self) -> String {
        let pos = self.position;
        while self.ch.is_some() && (is_digit(self.ch.unwrap()) || self.ch.unwrap() == '.') {
            self.read_char();
        }
        return self
            .input
            .char_indices()
            .skip(pos)
            .take(self.position - pos)
            .map(|(_, c)| c)
            .collect::<String>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_next_token_basics() {
        // let input = "=+(){},;";
        let input = "+()[],;if";
        let expected: [TT; 9] = [
            // TT::ASSIGN,
            TT::PLUS,
            TT::LPAREN,
            TT::RPAREN,
            TT::LBRACKET,
            TT::RBRACKET,
            // TT::LBRACE,
            // TT::RBRACE,
            TT::COMMA,
            TT::SEMICOLON,
            TT::IF,
            TT::EOF,
        ];

        let mut r = Lexer::new(String::from(input));
        for (i, item) in expected.iter().enumerate() {
            let next = r.next_token().token_type;
            assert_eq!(*item, next, "{}", i);
        }
    }

    #[test]
    fn test_next_token_more_symbols() {
        //let input = "+-><!/*==!=>=<=";
        let input = "+<><=>==-/*<-\\while";
        let expected: Vec<TT> = vec![
            // TT::FIRSTPICK,
            // TT::TALLY,
            TT::PLUS,
            TT::LESSTHAN,
            TT::GREATERTHAN,
            TT::LESSTHANEQUAL,
            TT::GREATERTHANEQUAL,
            TT::EQUAL,
            // TT::AND,
            // TT::OR,
            // TT::CEILMAX,
            // TT::FLOORMIN,
            // TT::WITHOUT,
            // TT::MAGMOD,
            TT::MINUS,
            // TT::GT,
            // TT::LT,
            // TT::BANG,
            TT::SLASH,
            // TT::SWAP,
            TT::ASTERISK,
            // TT::EQ,
            // TT::NE,
            // TT::GTE,
            // TT::LTE,
            TT::ASSIGN,
            TT::REDUCE,
            TT::WHILE,
            TT::EOF,
        ];

        let mut r = Lexer::new(String::from(input));
        for expected_token in expected {
            let next = r.next_token().token_type;
            assert_eq!(expected_token, next);
        }
    }

    #[test]
    fn test_identifier() {
        //let input = "+-><!/*==!=>=<=";
        let input = r#"someIdentifier * otherIdentifier;
test / differentTest;
"testing
asdf"
"#;
        let expected: [Token; 10] = [
            Token {
                token_type: TT::IDENTIFIER(String::from("someIdentifier")),
                lexeme: String::from("someIdentifier"),
            },
            Token {
                token_type: TT::ASTERISK,
                lexeme: String::from("*"),
            },
            Token {
                token_type: TT::IDENTIFIER(String::from("otherIdentifier")),
                lexeme: String::from("otherIdentifier"),
            },
            Token {
                token_type: TT::SEMICOLON,
                lexeme: String::from(";"),
            },
            Token {
                token_type: TT::IDENTIFIER(String::from("test")),
                lexeme: String::from("test"),
            },
            Token {
                token_type: TT::SLASH,
                lexeme: String::from("/"),
            },
            Token {
                token_type: TT::IDENTIFIER(String::from("differentTest")),
                lexeme: String::from("differentTest"),
            },
            Token {
                token_type: TT::SEMICOLON,
                lexeme: String::from(";"),
            },
            Token {
                token_type: TT::STRING(String::from("testing\nasdf")),
                lexeme: String::from("testing\nasdf"),
            },
            Token {
                token_type: TT::EOF,
                lexeme: String::from(""),
            },
        ];

        let mut r = Lexer::new(String::from(input));
        for (i, item) in expected.iter().enumerate() {
            let next = r.next_token();
            assert_eq!(*item, next, "{}", i);
        }
    }

    #[test]
    fn test_number() {
        let input = r#"1.25
"#;
        let expected: [Token; 1] = [Token {
            token_type: TT::NUMBER(1.25),
            lexeme: String::from("1.25"),
        }];

        let mut r = Lexer::new(String::from(input));
        for (i, item) in expected.iter().enumerate() {
            let next = r.next_token();
            assert_eq!(*item, next, "{}", i);
        }
    }
}
