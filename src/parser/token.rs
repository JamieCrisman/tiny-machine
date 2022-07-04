#[allow(clippy::upper_case_acronyms)]
#[derive(PartialEq, Clone, Debug)]
pub enum TokenType {
    ILLEGAL,
    EOF,

    IDENTIFIER(String),
    NUMBER(f64),
    STRING(String),
    BOOL(bool),

    PLUS,
    MINUS,
    ASTERISK,
    SLASH,

    // EqualEqual
    BANG,
    COMMA,
    // COLON,
    SEMICOLON,

    ASSIGN, // <-
    LPAREN,
    RPAREN,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    FUNCTION,
    IF,
    ELSE,
    RETURN,
    REDUCE,
    AND,
    OR,

    GREATERTHAN,
    GREATERTHANEQUAL,
    LESSTHAN,
    LESSTHANEQUAL,
    EQUAL,
    NOTEQUAL,
    WHILE,
    PISET, // switch to built in function
    // PIGET, 
}

#[derive(PartialEq, Clone, Debug)]
pub struct Token {
    pub token_type: TokenType,
    pub lexeme: String,
}
