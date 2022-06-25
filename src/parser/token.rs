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
    //LBRACE,
    //RBRACE,
    LBRACKET,
    RBRACKET,
    // FUNCTION,
    // LET,
    // IF,
    // ELSE,
    // RETURN,
    REDUCE,
    // TALLY,
    // SWAP,
    // WITHOUT,
    // FIRSTPICK,
    // CEILMAX,
    // FLOORMIN,
    // MAGMOD,
    AND,
    OR,

    GREATERTHAN,
    GREATERTHANEQUAL,
    LESSTHAN,
    LESSTHANEQUAL,
    EQUAL,
    NOTEQUAL,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Token {
    pub token_type: TokenType,
    pub lexeme: String,
}
