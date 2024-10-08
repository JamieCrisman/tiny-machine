use crate::parser::token::TokenType as TT;

use std::fmt;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Identifier(pub String);

impl From<TT> for Identifier {
    fn from(t: TT) -> Self {
        match t {
            TT::IDENTIFIER(a) => Self(a),
            _ => Self(String::from("unknown")),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    // Blank,
    Let(Identifier, Expression),
    Return(Expression),
    Expression(Expression),
    While(Expression, BlockStatement),
    //If(Expression, BlockStatement, Option<BlockStatement>),
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Self::Blank => write!(f, ""),
            Self::Let(i, e) => write!(f, "{} <- {};", i.0, e),
            Self::Return(e) => write!(f, "return {};", e),
            Self::Expression(e) => write!(f, "{}", e),
            Self::While(e, b) => write!(f, "{} {:?}", e, b),
            // _ => write!(f, "D:"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    // Blank,
    Identifier(Identifier),
    Literal(Literal),
    Postfix(Postfix, Box<Expression>),
    Prefix(Prefix, Box<Expression>),
    Infix(Infix, Box<Expression>, Box<Expression>),
    Index(Box<Expression>, Box<Expression>),
    If {
        condition: Box<Expression>,
        consequence: BlockStatement,
        alternative: Option<BlockStatement>,
    },
    //While {
    //    condition: Box<Expression>,
    //    body: BlockStatement,
    //},
    // TODO: generalize this
    Piset {
        params: Vec<Expression>,
    },
    Func {
        params: Vec<Identifier>,
        body: BlockStatement,
        name: String,
    },
    Call {
        func: Box<Expression>,
        args: Vec<Expression>,
    },
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Self::Blank => write!(f, ""),
            Self::Identifier(i) => write!(f, "{}", i.0),
            Self::Literal(l) => write!(f, "{}", l),
            Self::Prefix(p, e) => {
                write!(f, "({}{})", p, e)
            }
            Self::Postfix(p, e) => {
                write!(f, "({}{})", e, p)
            }
            Self::Infix(i, e1, e2) => {
                write!(f, "({} {} {})", e1, i, e2)
            } // _ => write!(f, "D:"),
            Self::Piset { params } => {
                write!(f, "piset({:?})", params)
            }
            Self::If {
                condition: _,
                consequence: _,
                alternative: _,
            } => {
                write!(
                    f,
                    "TODO // Can't implement display for Vec, need to wrap it"
                )
                //                if (alternative.is_some()) {
                //                    write!(
                //                        f,
                //                        "if ({}) {{\n\t{}\n}} else {{\n\t{}}}",
                //                        condition, consequence, alternative
                //                    )
                //                } else {
                //                    write!(f, "if ({}) {{\n\t{}\n}}", condition, consequence)
                //      //                }
            }
            Self::Func {
                params: _,
                body: _,
                name: _,
            } => {
                write!(f, "Todo")
            }
            Self::Call { func: _, args: _ } => {
                write!(f, "todo")
            }
            Self::Index(_exp, _exp2) => {
                write!(f, "todo")
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Prefix {
    // Tally,
    Minus,
    Bang,
    // Swap,
}

impl fmt::Display for Prefix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Prefix::Minus => write!(f, "-"),
            Prefix::Bang => write!(f, "-"),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum PostfixModifier {
    Reduce,
}

impl fmt::Display for PostfixModifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PostfixModifier::Reduce => write!(f, "\\"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Postfix {
    Modifier(PostfixModifier, Box<Expression>),
}

impl fmt::Display for Postfix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Postfix::Tally => write!(f, "≢"),
            Postfix::Modifier(postfix_modifier, infix) => {
                write!(f, "{}{}", postfix_modifier, infix)
            } // Postfix::Swap => write!(f, "⍨"),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Infix {
    // Modifier(InfixModifier, Box<Infix>),
    // Unknown,
    Plus,
    Minus,
    Divide,
    Multiply,
    // Without,
    // FirstPick,
    // FloorMin,
    // CeilMax,
    // MagMod,
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    // Reduce,
    // Eq,
    // Ne,
    // Gte,
    // Gt,
    // Lte,
    // Lt,
}

impl fmt::Display for Infix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Infix::Unknown => write!(f, "oh god unknown infix"),
            Infix::Plus => write!(f, "+"),
            Infix::Minus => write!(f, "-"),
            Infix::Divide => write!(f, "/"),
            Infix::Multiply => write!(f, "*"),
            Infix::And => write!(f, "∧"),
            Infix::Or => write!(f, "∨"),
            Infix::GreaterThan => write!(f, ">"),
            Infix::GreaterThanEqual => write!(f, "≥"),
            Infix::Equal => write!(f, "="),
            Infix::NotEqual => write!(f, "!="),
            Infix::LessThan => write!(f, "<"),
            Infix::LessThanEqual => write!(f, "≤"),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum Literal {
    Number(f64),
    String(String),
    Bool(bool),
    Symbol(TT),
    Array(Vec<Expression>),
    // Hash(Vec<(Expression, Expression)>),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Number(i) => write!(f, "{}", i),
            Self::String(s) => write!(f, "{}", s),
            Self::Bool(b) => {
                if *b {
                    write!(f, "true")
                } else {
                    write!(f, "false")
                }
            } // _ => write!(f, "D:"),
            Self::Symbol(t) => {
                write!(f, "{:?}", t)
            }
            Self::Array(v) => write!(f, "{:?}", v),
            // Self::Hash(v) => write!(f, "{:?}", v),
        }
    }
}

pub type BlockStatement = Vec<Statement>;

pub type Program = BlockStatement;

pub fn precedence_of(t: TT) -> Precedence {
    match t {
        TT::PLUS
        | TT::MINUS
        | TT::AND
        | TT::OR
        | TT::LESSTHAN
        | TT::LESSTHANEQUAL
        | TT::GREATERTHAN
        | TT::GREATERTHANEQUAL
        // Maybe raise precedence for comparisons?
        | TT::NOTEQUAL
        | TT::EQUAL => Precedence::Sum,
        TT::ASTERISK | TT::SLASH => Precedence::Sum,
        TT::BANG => Precedence::Prefix,
        TT::REDUCE => Precedence::Product,
        TT::LBRACKET => Precedence::Index,
        TT::LPAREN => Precedence::Call,
        _ => Precedence::Lowest,
    }
}

#[derive(PartialEq, Eq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    Lowest,
    // Equals,      // ==
    // LessGreater, // > or <
    Sum,     // +
    Product, // *
    Prefix,  // -X or !X
    Call,    // myFunction(x)
    Index,   // Index,       // array[index]
}
