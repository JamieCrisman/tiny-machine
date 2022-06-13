pub mod symbol_table;
use std::{
    borrow::BorrowMut,
    fmt::{self, Display},
    ops::{Deref, DerefMut},
    vec::Drain,
};

use crate::{
    code::{make, Opcode},
    parser::{
        ast::{
            Expression, Identifier, Infix, Literal, Postfix, PostfixModifier, Prefix, Program,
            Statement,
        },
        token::TokenType,
    },
};

use self::symbol_table::SymbolTable;

#[derive(Clone, Debug, PartialEq)]
pub struct Instructions {
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct EmittedInstruction {
    op: Opcode,
    operands: Option<Vec<i32>>,
    // position: usize,
}

pub struct CompilationScope {
    instructions: Vec<EmittedInstruction>,
}

#[derive(Debug)]
pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Objects,
}

pub struct Compiler {
    pub constants: Objects,
    pub symbol_table: SymbolTable,
    scope_index: usize,
    scopes: Vec<CompilationScope>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CompileError {
    Reason(String),
}

impl Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            CompileError::Reason(s) => write!(f, "Compile Error: {}", s),
        }
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            constants: Objects::new(),
            symbol_table: SymbolTable::new(),
            scope_index: 0,
            scopes: vec![CompilationScope {
                instructions: vec![],
            }],
        }
    }

    pub fn read(&mut self, program: Program) -> Result<(), CompileError> {
        for s in program {
            if let Err(e) = self.compile_statement(s) {
                return Err(e);
            }
        }
        Ok(())
    }

    fn compile_statement(&mut self, statement: Statement) -> Result<(), CompileError> {
        match statement {
            Statement::Blank => Ok(()),
            Statement::Expression(e) => {
                self.compile_expression(e)?;
                self.emit(Opcode::Pop, None);
                Ok(())
            }
            Statement::Let(l, e) => self.compile_let(l, e),
            Statement::Return(e) => self.compile_return(e),
            // _ => Err(CompileError::Reason(
            //     "Not Implemented statement".to_string(),
            // )),
        }
    }

    fn compile_let(&mut self, l: Identifier, e: Expression) -> Result<(), CompileError> {
        let symbol = self.symbol_table.borrow_mut().define(l.0.as_str());
        self.compile_expression(e)?;
        match symbol.scope {
            symbol_table::SymbolScope::Global => {
                self.emit(Opcode::SetGlobal, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Local => {
                self.emit(Opcode::SetLocal, Some(vec![symbol.index as i32]))
            }
            _ => return Err(CompileError::Reason("Invalid symbol scope".to_string())),
        };

        Ok(())
    }

    fn compile_expression(&mut self, exp: Expression) -> Result<(), CompileError> {
        match exp {
            Expression::Blank => Ok(()),
            Expression::Identifier(ident) => {
                let symbol = match self.symbol_table.resolve(ident.0.clone()) {
                    None => {
                        return Err(CompileError::Reason(format!(
                            "undefined variable: {}",
                            ident.0
                        )))
                    }
                    Some(val) => val,
                };
                let val = Some(vec![symbol.index as i32]);
                match symbol.scope {
                    symbol_table::SymbolScope::Global => self.emit(Opcode::GetGlobal, val),
                    symbol_table::SymbolScope::Local => self.emit(Opcode::GetLocal, val),
                    // symbol_table::SymbolScope::BuiltIn => self.emit(Opcode::BuiltinFunc, val),
                    // symbol_table::SymbolScope::Free => self.emit(Opcode::GetFree, val),
                    // symbol_table::SymbolScope::Function => self.emit(Opcode::CurrentClosure, None),
                    _ => {
                        return Err(CompileError::Reason(String::from(
                            "Unimplemented symbol scope",
                        )))
                    }
                };
                Ok(())
            }
            Expression::Infix(i, exp_a, exp_b) => self.compile_infix(i, exp_a, *exp_b),
            Expression::Prefix(p, exp) => self.compile_prefix(p, *exp),
            Expression::Literal(literal) => self.compile_literal(literal),
            Expression::Postfix(p, exp) => self.compile_postfix(p, *exp),
            // Expression::If {
            //     condition,
            //     consequence,
            //     alternative,
            // } => self.compile_if(condition, consequence, alternative),
            Expression::Index(expr, ind_expr) => self.compile_index(*expr, *ind_expr),
            // Expression::Func { params, body, name } => self.compile_function(params, body, name),
            // Expression::Call { args, func } => self.compile_call(args, func),
            // _ => Err(CompileError::Reason("Not Implemented".to_string())),
        }
    }

    fn compile_infix(
        &mut self,
        infix: Infix,
        exp_a: Box<Expression>,
        exp_b: Expression,
    ) -> Result<(), CompileError> {
        // if infix == Infix::Lt {
        //     self.compile_expression(*exp_b)?;
        //     self.compile_expression(*exp_a)?;
        //     self.emit(Opcode::GreaterThan, None);
        //     return Ok(());
        // }
        if let Err(e) = self.compile_expression(*exp_a) {
            return Err(e);
        }

        // if let Err(e) = self.compile_expression(*exp_a) {
        //     return Err(e);
        // }

        let _is_monadic = false;

        match exp_b {
            other_expression => {
                if let Err(e) = self.compile_expression(other_expression) {
                    return Err(e);
                }
            }
        };

        match infix {
            Infix::Plus => self.emit(Opcode::Add, None),
            Infix::Divide => self.emit(Opcode::Divide, None),
            Infix::Multiply => self.emit(Opcode::Multiply, None),
            Infix::Minus => self.emit(Opcode::Subtract, None),
            Infix::And => self.emit(Opcode::And, None),
            Infix::Or => self.emit(Opcode::Or, None),
            Infix::Equal => self.emit(Opcode::Equal, None),
            Infix::LessThan => self.emit(Opcode::LessThan, None),
            Infix::LessThanEqual => self.emit(Opcode::LessThanEqual, None),
            Infix::GreaterThan => self.emit(Opcode::GreaterThan, None),
            Infix::GreaterThanEqual => self.emit(Opcode::GreaterThanEqual, None),
            // Infix::Reduce => self.emit(Opcode::Reduce, None),
        };
        Ok(())
    }

    fn compile_prefix(
        &mut self,
        prefix: crate::parser::ast::Prefix,
        exp: Expression,
    ) -> Result<(), CompileError> {
        self.compile_expression(exp)?;
        match prefix {
            Prefix::Minus => self.emit(Opcode::Minus, None),
            // Prefix::Swap =>
        };
        Ok(())
    }

    fn compile_postfix(&mut self, postfix: Postfix, exp: Expression) -> Result<(), CompileError> {
        self.compile_expression(exp)?;
        match postfix {
            // Postfix::Tally => self.emit(Opcode::Tally, None),
            Postfix::Modifier(modifier, infix_op) => match modifier {
                PostfixModifier::Reduce => {
                    self.compile_expression(*infix_op)?;
                    self.emit(Opcode::Reduce, None)
                }
            },
            // _ => 0,
        };
        Ok(())
    }

    fn compile_return(&mut self, e: Expression) -> Result<(), CompileError> {
        self.compile_expression(e)?;
        self.emit(Opcode::ReturnValue, None);
        Ok(())
    }

    fn emit(&mut self, op: Opcode, operands: Option<Vec<i32>>) -> usize {
        // let ins = make(op.clone(), operands);
        let pos = match self.current_instructions() {
            Some(instr) => instr.len(),
            None => 0,
        };
        self.scopes[self.scope_index]
            .instructions
            .push(EmittedInstruction {
                op,
                operands,
                // position: pos,
            });
        pos
    }

    fn add_constant(&mut self, obj: Object) -> usize {
        self.constants.push(obj);
        self.constants.len() - 1
    }

    fn current_instructions(&mut self) -> Option<Vec<EmittedInstruction>> {
        let instructions = match self.scopes.get(self.scope_index) {
            Some(s) => s.instructions.clone(),
            None => return None,
        };
        Some(instructions)
    }

    fn compile_index(
        &mut self,
        expr: Expression,
        ind_expr: Expression,
    ) -> Result<(), CompileError> {
        self.compile_expression(expr)?;
        self.compile_expression(ind_expr)?;
        self.emit(Opcode::Index, None);
        Ok(())
    }

    fn compile_literal(&mut self, l: Literal) -> Result<(), CompileError> {
        match l {
            Literal::Number(num) => {
                let ind = self.add_constant(Object::Number(num)) as i32;
                self.emit(Opcode::Constant, Some(vec![ind]))
            }
            Literal::Symbol(tt) => {
                let ind = self.add_constant(Object::Symbol(SymbolType::from(tt))) as i32;
                self.emit(Opcode::Constant, Some(vec![ind]))
            }
            Literal::Bool(b) => {
                if b {
                    self.emit(Opcode::True, None)
                } else {
                    self.emit(Opcode::False, None)
                }
            }
            // Literal::String(s) => {
            //     let operand = Some(vec![self.add_constant(Object::String(s)) as i32]);
            //     self.emit(Opcode::Constant, operand)
            // }
            Literal::Array(elements) => {
                let size = Some(vec![elements.len() as i32]);
                for element in elements {
                    // println!("element: {:?}", element);
                    self.compile_expression(element)?;
                }
                self.emit(Opcode::Array, size);
                return Ok(());
            }
            // Literal::Hash(hash) => {
            //     // TODO:: sort by hash key?

            //     for (k, v) in hash.iter() {
            //         self.compile_expression(k.clone())?;
            //         self.compile_expression(v.clone())?;
            //     }
            //     self.emit(Opcode::Hash, Some(vec![(hash.len() * 2) as i32]));

            //     return Ok(());
            // }
            _ => {
                return Err(CompileError::Reason(
                    "Not an Implemented literal".to_string(),
                ))
            }
        };
        Ok(())
    }

    pub fn bytecode(&self) -> Bytecode {
        let instructions = self.scopes[self.scope_index]
            .instructions
            .iter()
            .map(|instr| make(instr.op.clone(), instr.operands.clone()))
            .filter(|r| r.is_ok())
            .map(|res| res.ok().unwrap())
            .reduce(|mut acc, mut item| {
                acc.data.append(&mut item.data);
                acc
            });

        Bytecode {
            instructions: instructions.unwrap(),
            constants: self.constants.clone(),
        }
    }
}

/// This is just a thin wrapper around the original Compiler.constants member
/// that makes it easier to keep track of what we're looking at in top-level code.
/// It just wraps the original type decl for compiler Objects.
/// All mentions of Vec<Object> elsewhere have been changed to instances of Constansts
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct Objects(Vec<Object>);
// impl Vec's methods as needed
impl Objects {
    pub fn new() -> Self {
        Objects(vec![])
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Objects(Vec::with_capacity(capacity))
    }

    pub fn push(&mut self, obj: Object) {
        self.0.push(obj)
    }

    pub fn pop(&mut self) -> Option<Object> {
        self.0.pop()
    }

    pub fn insert(&mut self, idx: usize, element: Object) {
        self.0.insert(idx, element)
    }

    pub fn remove(&mut self, idx: usize) -> Object {
        self.0.remove(idx)
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn drain<'a>(&'a mut self) -> Drain<'a, Object> {
        let len = self.0.len();
        self.0.drain(..len)
    }
}

// Impl Deref and DerefMut will automatically provide us with
// a whole host of methods available for slices.
// Probably not necessary now, but it's little work for a lot of gain.
// Ref: https://doc.rust-lang.org/nomicon/vec/vec-deref.html
impl Deref for Objects {
    type Target = [Object];
    fn deref(&self) -> &Self::Target {
        Deref::deref(&self.0)
    }
}

impl DerefMut for Objects {
    fn deref_mut(&mut self) -> &mut [Object] {
        DerefMut::deref_mut(&mut self.0)
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(PartialEq, Clone, Debug, Hash)]
pub enum SymbolType {
    UNKNOWN,
    PLUS,
    MINUS,
    ASTERISK,
    SLASH,
    // MIN,
    // MAX,
    MOD,
    AND,
    OR,
    EQUAL,
    LESSTHAN,
    LESSTHANEQUAL,
    GREATERTHAN,
    GREATERTHANEQUAL,
}

impl From<TokenType> for SymbolType {
    fn from(item: TokenType) -> Self {
        match item {
            TokenType::MINUS => SymbolType::MINUS,
            TokenType::PLUS => SymbolType::PLUS,
            TokenType::ASTERISK => SymbolType::ASTERISK,
            TokenType::SLASH => SymbolType::SLASH,
            // TokenType::CEILMAX => SymbolType::MAX,
            // TokenType::FLOORMIN => SymbolType::MIN,
            // TokenType::MAGMOD => SymbolType::MOD,
            TokenType::AND => SymbolType::AND,
            TokenType::OR => SymbolType::OR,
            _ => SymbolType::UNKNOWN,
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum Object {
    Number(f64),
    Symbol(SymbolType),
    // String(String),
    // Bool(bool),
    Array(Vec<Object>),
    // Hash(HashMap<Object, Object>),
    // Func(Vec<Ident>, BlockStatement, Rc<RefCell<Env>>, String),
    // Builtin(i32, BuiltInFunc),
    Null,
    // ReturnValue(Box<Object>),
    // Error(String),
    // CompiledFunction {
    //     instructions: Instructions,
    //     num_locals: i32,
    //     num_parameters: i32,
    // },
    // Closure {
    //     Fn: Box<Object>, // technically specifically ObjectCompiledFunction
    //     Free: Vec<Object>,
    // },
}

impl Object {
    pub fn object_type(&self) -> ObjectType {
        match self {
            Object::Number(_) => ObjectType::Number,
            Object::Symbol(_) => ObjectType::Symbol,
            // Object::String(_) => ObjectType::String,
            // Object::Bool(_) => ObjectType::Bool,
            Object::Array(_) => ObjectType::Array,
            // Object::Hash(_) => ObjectType::Hash,
            // Object::Func(_, _, _, _) => ObjectType::Func,
            // Object::Builtin(_, _) => ObjectType::Builtin,
            Object::Null => ObjectType::Null,
            // Object::ReturnValue(_) => ObjectType::ReturnValue,
            // Object::Error(_) => ObjectType::Error,
            // Object::CompiledFunction {
            // instructions: _,
            // num_locals: _,
            // num_parameters: _,
            // } => ObjectType::CompiledFunction,
            // Object::Closure { Fn: _, Free: _ } => ObjectType::Closure,
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum ObjectType {
    Number,
    // String,
    // Bool,
    Array,
    // Hash,
    // Func,
    // Builtin,
    Null,
    // ReturnValue,
    // Error,
    // CompiledFunction,
    // Closure,
    Symbol,
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ObjectType::*;
        let fmt_str = match self {
            Number => "Number".to_string(),
            // String => "String".to_string(),
            // Bool => "Bool".to_string(),
            Array => "Array".to_string(),
            // Hash => "Hash".to_string(),
            // Func => "Func".to_string(),
            // Builtin => "Builtin".to_string(),
            Null => "Null".to_string(),
            // ReturnValue => "ReturnValue".to_string(),
            // Error => "Error".to_string(),
            // CompiledFunction => "CompiledFunction".to_string(),
            // Closure => "Closure".to_string(),
            Symbol => "Symbol".to_string(),
        };
        write!(f, "{}", fmt_str)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Object::Number(ref value) => write!(f, "{:.}", value),
            // Object::String(ref value) => write!(f, "{}", value),
            // Object::Bool(ref value) => write!(f, "{}", value),
            Object::Symbol(ref s) => write!(f, "{:?}", s),
            Object::Array(ref objects) => {
                let mut result = String::new();
                for (i, obj) in objects.iter().enumerate() {
                    if i < 1 {
                        result.push_str(&format!("{}", obj));
                    } else {
                        result.push_str(&format!(", {}", obj));
                    }
                }
                write!(f, "[ {} ]", result)
            }
            // Object::Hash(ref hash) => {
            //     let mut result = String::new();
            //     for (i, (k, v)) in hash.iter().enumerate() {
            //         if i < 1 {
            //             result.push_str(&format!("{}: {}", k, v));
            //         } else {
            //             result.push_str(&format!(", {}: {}", k, v));
            //         }
            //     }
            //     write!(f, "{{{}}}", result)
            // }
            // Object::Func(ref params, _, _, _) => {
            //     let mut result = String::new();
            //     for (i, Ident(ref s)) in params.iter().enumerate() {
            //         if i < 1 {
            //             result.push_str(&format!("{}", s));
            //         } else {
            //             result.push_str(&format!(", {}", s));
            //         }
            //     }
            //     write!(f, "{}({}) {{ ... }}", "func", result)
            // }
            // Object::Builtin(_, _) => write!(f, "[builtin function]"),
            Object::Null => write!(f, "null"),
            // Object::ReturnValue(ref value) => write!(f, "{}", value),
            // Object::Error(ref value) => write!(f, "{}", value),
            // Object::Closure { Fn: _, Free: _ } => write!(f, "Closure"),
            // Object::CompiledFunction {
            //     instructions: _,
            //     num_locals,
            //     num_parameters,
            // } => write!(
            //     f,
            //     "Compiled Function with {} locals and {} parameters",
            //     num_locals, num_parameters
            // ),
        }
    }
}
