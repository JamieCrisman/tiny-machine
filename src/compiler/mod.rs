pub mod builtin;
pub mod envir;
pub mod symbol_table;

use core::cell::RefCell;
use std::rc::Rc;
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
            BlockStatement, Expression, Identifier, Infix, Literal, Postfix, PostfixModifier,
            Prefix, Program, Statement,
        },
        token::TokenType,
    },
};

use self::envir::Envir;
use self::symbol_table::SymbolTable;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instructions {
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct EmittedInstruction {
    op: Opcode,
    operands: Option<Vec<i32>>,
    position: usize,
}

pub struct CompilationScope {
    instructions: Vec<EmittedInstruction>,
    // these let us hold a temporary value to modify later
    last_instruction: Option<EmittedInstruction>,
    previous_instruction: Option<EmittedInstruction>,
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
            symbol_table: SymbolTable::new_with_builtins(),
            scope_index: 0,
            scopes: vec![CompilationScope {
                instructions: vec![],
                last_instruction: None,
                previous_instruction: None,
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

    pub fn compile(&mut self, program: Vec<Statement>) -> Result<(), CompileError> {
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
            Statement::While(c, b) => self.compile_while(c, b),
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
                    symbol_table::SymbolScope::BuiltIn => self.emit(Opcode::BuiltinFunc, val),
                    symbol_table::SymbolScope::Free => self.emit(Opcode::GetFree, val),
                    symbol_table::SymbolScope::Function => self.emit(Opcode::CurrentClosure, None),
                    // _ => {
                    //     return Err(CompileError::Reason(String::from(
                    //         "Unimplemented symbol scope",
                    //     )))
                    // }
                };
                Ok(())
            }
            Expression::Infix(i, exp_a, exp_b) => self.compile_infix(i, exp_a, *exp_b),
            Expression::Prefix(p, exp) => self.compile_prefix(p, *exp),
            Expression::Literal(literal) => self.compile_literal(literal),
            Expression::Postfix(p, exp) => self.compile_postfix(p, *exp),
            Expression::If {
                condition,
                consequence,
                alternative,
            } => self.compile_if(condition, consequence, alternative),
            //Expression::While {
            //     condition,
            //     body,
            //} => self.compile_while(condition, body),
            Expression::Index(expr, ind_expr) => self.compile_index(*expr, *ind_expr),
            Expression::Piset { params } => self.compile_piset(params),
            Expression::Func { params, body, name } => self.compile_function(params, body, name),
            Expression::Call { args, func } => self.compile_call(args, func),
            //Expression::Call { args: _, func: _ } => {
            //  Err(CompileError::Reason("Not Implemented".to_string()))
            //} // _ => Err(CompileError::Reason("Not Implemented".to_string())),
        }
    }

    fn compile_call(
        &mut self,
        args: Vec<Expression>,
        func: Box<Expression>,
    ) -> Result<(), CompileError> {
        self.compile_expression(*func)?;
        let len = args.len() as i32;
        for a in args {
            self.compile_expression(a)?;
        }
        self.emit(Opcode::Call, Some(vec![len]));
        Ok(())
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
            Infix::NotEqual => self.emit(Opcode::NotEqual, None),
            Infix::LessThan => self.emit(Opcode::LessThan, None),
            Infix::LessThanEqual => self.emit(Opcode::LessThanEqual, None),
            Infix::GreaterThan => self.emit(Opcode::GreaterThan, None),
            Infix::GreaterThanEqual => self.emit(Opcode::GreaterThanEqual, None),
            // Infix::Reduce => self.emit(Opcode::Reduce, None),
        };
        Ok(())
    }

    fn compile_piset(&mut self, params: Vec<Expression>) -> Result<(), CompileError> {
        for exp in params {
            self.compile_expression(exp)?;
        }

        self.emit(Opcode::Piset, None);
        // piset doesn't return a value, so as an expression, it should return null
        self.emit(Opcode::Null, None);

        Ok(())
    }

    fn compile_prefix(&mut self, prefix: Prefix, exp: Expression) -> Result<(), CompileError> {
        self.compile_expression(exp)?;
        match prefix {
            Prefix::Bang => self.emit(Opcode::Bang, None),
            Prefix::Minus => self.emit(Opcode::Minus, None),
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

    fn compile_while(
        &mut self,
        condition: Expression,
        body: BlockStatement,
    ) -> Result<(), CompileError> {
        let condition_start = self.scope_to_byte_position();
        self.compile_expression(condition)?;
        // this gets properly set later (back patching)
        let jump_not_truthy_pos = self.emit(Opcode::JumpNotTruthy, Some(vec![9999]));
        //self.emit(Opcode::Pop, None); // TODO:???

        self.compile(body)?;
        //if self.last_instruction_is(Opcode::Pop) {
        //    self.remove_last_pop();
        //}

        self.emit(Opcode::Jump, Some(vec![condition_start]));

        let after_body_pos = self.scope_to_byte_position();
        self.change_operand(jump_not_truthy_pos, Some(vec![after_body_pos as i32]));
        // emit pop?
        // self.emit(Opcode::Pop, None); // TODO:???
        //

        Ok(())
    }

    fn compile_if(
        &mut self,
        condition: Box<Expression>,
        consequence: BlockStatement,
        alternative: Option<BlockStatement>,
    ) -> Result<(), CompileError> {
        self.compile_expression(*condition)?;
        // this gets properly set later (back patching)
        let jump_not_truthy_pos = self.emit(Opcode::JumpNotTruthy, Some(vec![9999]));
        self.compile(consequence)?;

        if self.last_instruction_is(Opcode::Pop) {
            self.remove_last_pop();
        }
        // to be backpatched
        let jump_pos = self.emit(Opcode::Jump, Some(vec![9999]));
        //let after_consequence_pos = self.scopes[self.scope_index].instructions.len();
        let after_consequence_pos = self.scope_to_byte_position();
        self.change_operand(
            jump_not_truthy_pos,
            Some(vec![after_consequence_pos as i32]),
        );

        if let Some(else_stmt) = alternative {
            self.compile(else_stmt)?;

            // we remove the last pop because when if is an expression, it needs to return
            // a value
            if self.last_instruction_is(Opcode::Pop) {
                self.remove_last_pop();
            }
        } else {
            // likewise, if needs to return a value, if nothing is returned from the if, it
            // should return a null
            self.emit(Opcode::Null, None);
        }
        // let after_alternative_pos = self.scopes[self.scope_index].instructions.len();
        let after_alternative_pos = self.scope_to_byte_position();
        self.change_operand(jump_pos, Some(vec![after_alternative_pos as i32]));

        Ok(())
    }

    fn change_operand(&mut self, opcode_pos: usize, operands: Option<Vec<i32>>) {
        let op = &self
            .current_instructions()
            .expect("expected instructions to exist")[opcode_pos];
        self.replace_instruction(
            opcode_pos,
            EmittedInstruction {
                op: op.op.clone(),
                operands,
                position: op.position,
            },
        );
    }

    fn scope_to_byte_position(&self) -> i32 {
        self.scopes[self.scope_index]
            .instructions
            .iter()
            .fold(0_usize, |mut sum, x| {
                sum += x.op.operand_width();
                sum + 1
            }) as i32
    }

    fn replace_instruction(&mut self, pos: usize, new_instruction: EmittedInstruction) {
        self.scopes[self.scope_index].instructions[pos] = new_instruction;
    }

    fn last_instruction_is(&mut self, op: Opcode) -> bool {
        self.scopes
            .get(self.scope_index)
            .unwrap()
            .last_instruction
            .is_some()
            && self
                .scopes
                .get(self.scope_index)
                .unwrap()
                .last_instruction
                .as_ref()
                .unwrap()
                .op
                == op
    }

    fn remove_last_pop(&mut self) {
        // self.instructions
        let pos = self.scopes[self.scope_index]
            .last_instruction
            .as_ref()
            .unwrap()
            .position;
        while self.scopes[self.scope_index].instructions.len() > pos {
            self.scopes[self.scope_index].instructions.pop();
        }
        self.scopes[self.scope_index].last_instruction =
            self.scopes[self.scope_index].previous_instruction.clone();
    }

    fn compile_function(
        &mut self,
        params: Vec<Identifier>,
        body: BlockStatement,
        name: String,
    ) -> Result<(), CompileError> {
        self.enter_scope();
        if !name.is_empty() {
            self.symbol_table.borrow_mut().define_function(name);
        }
        let param_len = params.len() as i32;
        for p in params {
            self.symbol_table.borrow_mut().define(p.0.as_str());
        }

        self.compile(body)?;
        if self.last_instruction_is(Opcode::Pop) {
            self.replace_last_pop_with_return();
        }
        if !self.last_instruction_is(Opcode::ReturnValue) {
            self.emit(Opcode::Return, None);
        }
        let num_locals = self.symbol_table.num_definitions;
        let free_symbols = self.symbol_table.free_symbols.clone();
        let instr = self.leave_scope();

        for sym in free_symbols.iter() {
            self.load_symbol(sym.clone());
        }

        let compiled_fn = Object::CompiledFunction {
            instructions: instr,
            num_locals: num_locals as i32,
            num_parameters: param_len,
        };
        let constant_val = Some(vec![
            self.add_constant(compiled_fn) as i32,
            free_symbols.len() as i32,
        ]);
        self.emit(Opcode::Closure, constant_val);
        Ok(())
    }

    fn load_symbol(&mut self, symbol: symbol_table::Symbol) {
        match symbol.scope {
            symbol_table::SymbolScope::Global => {
                self.emit(Opcode::GetGlobal, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Local => {
                self.emit(Opcode::GetLocal, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::BuiltIn => {
                self.emit(Opcode::BuiltinFunc, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Free => {
                self.emit(Opcode::GetFree, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Function => self.emit(Opcode::CurrentClosure, None),
        };
    }

    fn replace_last_pop_with_return(&mut self) {
        let last_pos = self.scopes[self.scope_index]
            .last_instruction
            .as_ref()
            .unwrap()
            .position;

        self.replace_instruction(
            last_pos,
            EmittedInstruction {
                op: Opcode::ReturnValue,
                operands: None,
                // this might not be right
                position: last_pos,
            },
        );
        self.scopes[self.scope_index]
            .last_instruction
            .as_mut()
            .unwrap()
            .op = Opcode::ReturnValue;
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
        let recent_emit = EmittedInstruction {
            op,
            operands,
            position: pos,
        };
        self.scopes[self.scope_index]
            .instructions
            .push(recent_emit.clone());
        self.set_last_instruction(recent_emit);
        pos
    }

    fn set_last_instruction(&mut self, last: EmittedInstruction) {
        let prev = self
            .scopes
            .get(self.scope_index)
            .unwrap()
            .last_instruction
            .clone();
        self.scopes[self.scope_index].previous_instruction = prev;
        self.scopes[self.scope_index].last_instruction = Some(last);
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

    fn enter_scope(&mut self) {
        let scope = CompilationScope {
            instructions: vec![],
            last_instruction: None,
            previous_instruction: None,
        };
        self.scopes.push(scope);
        self.scope_index += 1;
        let new_table = SymbolTable::new_with_outer(Box::new(self.symbol_table.to_owned()));
        self.symbol_table = new_table;
    }

    fn leave_scope(&mut self) -> Instructions {
        let Bytecode {
            instructions: result,
            constants: _,
        } = self.bytecode();

        self.scopes.pop();
        self.scope_index -= 1;
        let outer = self.symbol_table.outer.as_ref().unwrap();
        self.symbol_table = *outer.to_owned();
        // TODO: drop? inner?
        result
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

    pub fn len(&self) -> usize {
        self.0.len()
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
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SymbolType {
    UNKNOWN,
    PLUS,
    MINUS,
    ASTERISK,
    SLASH,
    // MIN,
    // MAX,
    // MOD,
    AND,
    OR,
    EQUAL,
    NOTEQUAL,
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

pub type BuiltInFunc = fn(Vec<Object>) -> Object;

#[derive(PartialEq, Clone, Debug)]
pub enum Object {
    Number(f64),
    Symbol(SymbolType),
    // String(String),
    // Bool(bool),
    Array(Vec<Object>),
    // Hash(HashMap<Object, Object>),
    Func(Vec<Identifier>, BlockStatement, Rc<RefCell<Envir>>, String),
    Builtin(i32, BuiltInFunc),
    Null,
    ReturnValue(Box<Object>),
    Error(String),
    CompiledFunction {
        instructions: Instructions,
        num_locals: i32,
        num_parameters: i32,
    },
    Closure {
        Fn: Box<Object>, // technically specifically ObjectCompiledFunction
        Free: Vec<Object>,
    },
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
            Object::Func(_, _, _, _) => ObjectType::Func,
            Object::Builtin(_, _) => ObjectType::Builtin,
            Object::Null => ObjectType::Null,
            Object::ReturnValue(_) => ObjectType::ReturnValue,
            Object::Error(_) => ObjectType::Error,
            Object::CompiledFunction {
                instructions: _,
                num_locals: _,
                num_parameters: _,
            } => ObjectType::CompiledFunction,
            Object::Closure { Fn: _, Free: _ } => ObjectType::Closure,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum ObjectType {
    Number,
    // String,
    // Bool,
    Array,
    // Hash,
    Func,
    Builtin,
    Null,
    ReturnValue,
    Error,
    CompiledFunction,
    Closure,
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
            Func => "Func".to_string(),
            Builtin => "Builtin".to_string(),
            Null => "Null".to_string(),
            ReturnValue => "ReturnValue".to_string(),
            Error => "Error".to_string(),
            CompiledFunction => "CompiledFunction".to_string(),
            Closure => "Closure".to_string(),
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
            Object::Func(ref params, _, _, _) => {
                let mut result = String::new();
                for (i, Identifier(ref s)) in params.iter().enumerate() {
                    if i < 1 {
                        result.push_str(s);
                    } else {
                        result.push_str(&format!(", {}", s));
                    }
                }
                write!(f, "func({}) {{ ... }}", result)
            }
            Object::Builtin(_, _) => write!(f, "[builtin function]"),
            Object::Null => write!(f, "null"),
            Object::ReturnValue(ref value) => write!(f, "{}", value),
            Object::Error(ref value) => write!(f, "{}", value),
            Object::Closure { Fn: _, Free: _ } => write!(f, "Closure"),
            Object::CompiledFunction {
                instructions: _,
                num_locals,
                num_parameters,
            } => write!(
                f,
                "Compiled Function with {} locals and {} parameters",
                num_locals, num_parameters
            ),
        }
    }
}

#[cfg(test)]
mod tests {

    use std::vec;

    // use crate::evaluator::builtins::new_builtins;
    use crate::compiler::*;
    use crate::parser;
    use crate::parser::lexer::Lexer;

    struct CompilerTestCase {
        input: String,
        expected_constants: Vec<Object>,
        expected_instructions: Vec<Instructions>,
    }

    fn parse(input: String) -> parser::ast::Program {
        let l = Lexer::new(input);
        let mut p = parser::Parser::new(l);
        p.build_ast()
    }

    #[test]
    fn test_assignment() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "a <- 1;a".to_string(),
                expected_constants: vec![Object::Number(1.0)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "a <- 1;a <- a + 1; a".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(1.0)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Add, None).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_functions() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "fn() { return 5 + 10 }".to_string(),
                expected_constants: vec![
                    Object::Number(5.0),
                    Object::Number(10.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { 5 + 10 }".to_string(),
                expected_constants: vec![
                    Object::Number(5.0),
                    Object::Number(10.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { 1; 2 }".to_string(),
                expected_constants: vec![
                    Object::Number(1.0),
                    Object::Number(2.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Pop, None).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { }".to_string(),
                expected_constants: vec![Object::CompiledFunction {
                    instructions: concat_instructions(vec![make(Opcode::Return, None).unwrap()]),
                    num_locals: 0,
                    num_parameters: 0,
                }],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_recursive() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "countDown <- fn(x) { countDown(x-1) }; countDown(1);".to_string(),
                expected_constants: vec![
                    Object::Number(1.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::CurrentClosure, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Subtract, None).unwrap(),
                            make(Opcode::Call, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::Number(1.0),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "wrapper <- fn() {countDown <- fn(x) { countDown(x-1) }; countDown(1);}; wrapper();".to_string(),
                expected_constants: vec![
                    Object::Number(1.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::CurrentClosure, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Subtract, None).unwrap(),
                            make(Opcode::Call, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::Number(1.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Closure, Some(vec![1,0])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![2])).unwrap(),
                            make(Opcode::Call, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![3, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Call, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_function_calls() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "fn() { 24 }()".to_string(),
                expected_constants: vec![
                    Object::Number(24.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Call, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "noArg <- fn() { 24 };noArg();".to_string(),
                expected_constants: vec![
                    Object::Number(24.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Call, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "aArg <- fn(a) { };aArg(24);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(
                            vec![make(Opcode::Return, None).unwrap()],
                        ),
                        num_parameters: 1,
                        num_locals: 1,
                    },
                    Object::Number(24.0),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "someArgs <- fn(a,b,c) { };someArgs(24,25,26);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(
                            vec![make(Opcode::Return, None).unwrap()],
                        ),
                        num_locals: 3,
                        num_parameters: 3,
                    },
                    Object::Number(24.0),
                    Object::Number(25.0),
                    Object::Number(26.0),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Call, Some(vec![3])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "someArg <- fn(a) { a };someArg(24);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::Number(24.0),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "someArg <- fn(a, b, c) { a;b;c };someArg(24,25,26);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Pop, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![1])).unwrap(),
                            make(Opcode::Pop, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![2])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 3,
                        num_parameters: 3,
                    },
                    Object::Number(24.0),
                    Object::Number(25.0),
                    Object::Number(26.0),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Call, Some(vec![3])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_let_scopes() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "one <- 100; fn() { one; };".to_string(),
                expected_constants: vec![
                    Object::Number(100.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { one <- 100; one; };".to_string(),
                expected_constants: vec![
                    Object::Number(100.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { one <- 100; two <- 200; one+two };".to_string(),
                expected_constants: vec![
                    Object::Number(100.0),
                    Object::Number(200.0),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![1])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 2,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            // CompilerTestCase {
            //     input: "let one = 1; one;".to_string(),
            //     expected_constants: vec![Object::Number(1).0],
            //     expected_instructions: vec![
            //         make(Opcode::Constant, Some(vec![0])).unwrap(),
            //         make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::Pop, None).unwrap(),
            //     ],
            // },
            // CompilerTestCase {
            //     input: "let one = 1; let two = one; two;".to_string(),
            //     expected_constants: vec![Object::Number(1).0],
            //     expected_instructions: vec![
            //         make(Opcode::Constant, Some(vec![0])).unwrap(),
            //         make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::SetGlobal, Some(vec![1])).unwrap(),
            //         make(Opcode::GetGlobal, Some(vec![1])).unwrap(),
            //         make(Opcode::Pop, None).unwrap(),
            //     ],
            // },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_piset() {
        let tests: Vec<CompilerTestCase> = vec![CompilerTestCase {
            input: "piset(1,2,3,4)".to_string(),
            expected_constants: vec![
                Object::Number(1.0),
                Object::Number(2.0),
                Object::Number(3.0),
                Object::Number(4.0),
            ],
            expected_instructions: vec![
                make(Opcode::Constant, Some(vec![0])).unwrap(),
                make(Opcode::Constant, Some(vec![1])).unwrap(),
                make(Opcode::Constant, Some(vec![2])).unwrap(),
                make(Opcode::Constant, Some(vec![3])).unwrap(),
                make(Opcode::Piset, None).unwrap(),
                make(Opcode::Null, None).unwrap(),
                make(Opcode::Pop, None).unwrap(),
            ],
        }];

        run_compiler_test(tests);
    }

    #[test]
    fn test_bool_arithmetic() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "1 > 2".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(2.0)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::GreaterThan, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 < 2".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(2.0)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::LessThan, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 = 2".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(2.0)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Equal, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 != 2".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(2.0)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::NotEqual, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_while_loop() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "while (1) { 25 }".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(25.0)],
                expected_instructions: vec![
                    // 00
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    // 03
                    make(Opcode::JumpNotTruthy, Some(vec![13])).unwrap(),
                    // 06
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    // 09
                    make(Opcode::Pop, None).unwrap(),
                    make(Opcode::Jump, Some(vec![0])).unwrap(),
                    // 10
                    // 14
                ],
            },
            CompilerTestCase {
                input: "a <- 0; while (a < 10) { a <- 10 };".to_string(),
                expected_constants: vec![
                    Object::Number(0.0),
                    Object::Number(10.0),
                    Object::Number(10.0),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::LessThan, None).unwrap(),
                    make(Opcode::JumpNotTruthy, Some(vec![25])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Jump, Some(vec![6])).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_if_conditionals() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "if (1) { 10 } else { 20 }; 3333;".to_string(),
                expected_constants: vec![
                    Object::Number(1.0),
                    Object::Number(10.0),
                    Object::Number(20.0),
                    Object::Number(3333.0),
                ],
                expected_instructions: vec![
                    // 00
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    // 03
                    make(Opcode::JumpNotTruthy, Some(vec![12])).unwrap(),
                    // 06
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    // 09
                    make(Opcode::Jump, Some(vec![15])).unwrap(),
                    // 10
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    // 13
                    make(Opcode::Pop, None).unwrap(),
                    // 14
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    // 17
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "if (1) { 10 }; 3333;".to_string(),
                expected_constants: vec![
                    Object::Number(1.0),
                    Object::Number(10.0),
                    Object::Number(3333.0),
                ],
                expected_instructions: vec![
                    // 00
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    // 03
                    make(Opcode::JumpNotTruthy, Some(vec![12])).unwrap(),
                    // 06
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    // 09
                    make(Opcode::Jump, Some(vec![13])).unwrap(),
                    // 12
                    make(Opcode::Null, None).unwrap(),
                    // 13
                    make(Opcode::Pop, None).unwrap(),
                    // 14
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    // 15
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            // if(1) {2};
            CompilerTestCase {
                input: "if (1) { 2 };".to_string(),
                expected_constants: vec![Object::Number(1.0), Object::Number(2.0)],
                expected_instructions: vec![
                    // 00
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    // 03
                    make(Opcode::JumpNotTruthy, Some(vec![12])).unwrap(),
                    // 06
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    // 09
                    make(Opcode::Jump, Some(vec![13])).unwrap(),
                    // 10
                    make(Opcode::Null, None).unwrap(),
                    // 11
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    /*
     */

    #[test]
    fn test_piset_loop() {
        let tests: Vec<CompilerTestCase> = vec![CompilerTestCase {
            input: r#"x <- 0
y <- 0
c <- 0
while(1) {
	piset(x,y,1,c)
	x <- x+1
	if (x >= 240) {
		x <- 0
	}
	if (x = 0) {
		y <- y + 1
	}
	if (y >= 160) {
		y <- 0
	}
}"#
            .to_string(),
            expected_constants: vec![
                Object::Number(0.0),
                Object::Number(0.0),
                Object::Number(0.0),
                Object::Number(1.0),
                Object::Number(1.0),
                Object::Number(1.0),
                Object::Number(240.0),
                Object::Number(0.0),
                Object::Number(0.0),
                Object::Number(1.0),
                Object::Number(160.0),
                Object::Number(0.0),
            ],
            expected_instructions: vec![
                // x, y, c
                make(Opcode::Constant, Some(vec![0])).unwrap(), // +
                make(Opcode::SetGlobal, Some(vec![0])).unwrap(), // -
                make(Opcode::Constant, Some(vec![1])).unwrap(), // +
                make(Opcode::SetGlobal, Some(vec![1])).unwrap(), // -
                make(Opcode::Constant, Some(vec![2])).unwrap(), // +
                make(Opcode::SetGlobal, Some(vec![2])).unwrap(), // -
                // while
                make(Opcode::Constant, Some(vec![3])).unwrap(), // +
                make(Opcode::JumpNotTruthy, Some(vec![119])).unwrap(), // -
                // piset
                make(Opcode::GetGlobal, Some(vec![0])).unwrap(), // +
                make(Opcode::GetGlobal, Some(vec![1])).unwrap(), // +
                make(Opcode::Constant, Some(vec![4])).unwrap(),  // +
                make(Opcode::GetGlobal, Some(vec![2])).unwrap(), // +
                make(Opcode::Piset, None).unwrap(),              // - - - -
                make(Opcode::Null, None).unwrap(),               // +
                make(Opcode::Pop, None).unwrap(),                // -
                // 0 so far
                // increment x
                make(Opcode::GetGlobal, Some(vec![0])).unwrap(), // +
                make(Opcode::Constant, Some(vec![5])).unwrap(),  // +
                make(Opcode::Add, None).unwrap(),                // - - +
                make(Opcode::SetGlobal, Some(vec![0])).unwrap(), // -
                // check if x is >= 240 // 0
                make(Opcode::GetGlobal, Some(vec![0])).unwrap(), // +
                make(Opcode::Constant, Some(vec![6])).unwrap(),  // +
                make(Opcode::GreaterThanEqual, None).unwrap(),   // - - +
                make(Opcode::JumpNotTruthy, Some(vec![68])).unwrap(), // -
                // set to zero if >= 240 // 0
                make(Opcode::Constant, Some(vec![7])).unwrap(), // +
                make(Opcode::SetGlobal, Some(vec![0])).unwrap(), // -
                make(Opcode::Jump, Some(vec![69])).unwrap(),    // ?
                make(Opcode::Null, None).unwrap(),              // +
                make(Opcode::Pop, None).unwrap(),               // -
                // if x = 0
                make(Opcode::GetGlobal, Some(vec![0])).unwrap(), // +
                make(Opcode::Constant, Some(vec![8])).unwrap(),  // +
                make(Opcode::Equal, None).unwrap(),              // - - +
                make(Opcode::JumpNotTruthy, Some(vec![93])).unwrap(), // -
                // y <- y + 1 // 0
                make(Opcode::GetGlobal, Some(vec![1])).unwrap(), // +
                make(Opcode::Constant, Some(vec![9])).unwrap(),  // +
                make(Opcode::Add, None).unwrap(),                // - - +
                make(Opcode::SetGlobal, Some(vec![1])).unwrap(), // -
                make(Opcode::Jump, Some(vec![94])).unwrap(),     // ?
                make(Opcode::Null, None).unwrap(),               // +
                make(Opcode::Pop, None).unwrap(),                // -
                // if y >= 160  // 0
                make(Opcode::GetGlobal, Some(vec![1])).unwrap(), // +
                make(Opcode::Constant, Some(vec![10])).unwrap(), // +
                make(Opcode::GreaterThanEqual, None).unwrap(),   // - - +
                make(Opcode::JumpNotTruthy, Some(vec![114])).unwrap(), // -
                // y <- 0 // 0
                make(Opcode::Constant, Some(vec![11])).unwrap(), // +
                make(Opcode::SetGlobal, Some(vec![1])).unwrap(), // -
                make(Opcode::Jump, Some(vec![115])).unwrap(),    // ?
                make(Opcode::Null, None).unwrap(),               // +   <<<<<
                // DIDN'T POP NULL?
                make(Opcode::Pop, None).unwrap(),
                make(Opcode::Jump, Some(vec![18])).unwrap(), // ?
                                                             //make(Opcode::Null, None).unwrap(), // +
                                                             //make(Opcode::Pop, None).unwrap(), // -
            ],
        }];

        run_compiler_test(tests);
    }

    fn run_compiler_test(tests: Vec<CompilerTestCase>) {
        for test in tests {
            println!("testing {}", test.input);
            let program = parse(test.input.clone());
            //let st = SymbolTable::new();
            //let constants = vec![];
            let mut c = Compiler::new();
            // println!("{:?}", program);
            let compile_result = c.compile(program);
            assert!(
                compile_result.is_ok(),
                "{:?}",
                compile_result
                    .err()
                    .unwrap_or(CompileError::Reason("uh... not sure".to_string()))
            );

            let bytecode = c.bytecode();
            // println!("{:?}", test.input);
            let instruction_result =
                test_instructions(test.expected_instructions, bytecode.instructions);
            assert!(instruction_result.is_ok());

            let constant_result = test_constants(test.expected_constants, bytecode.constants);
            assert!(constant_result.is_ok());
        }
    }

    fn test_instructions(
        expected: Vec<Instructions>,
        got: Instructions,
    ) -> Result<(), CompileError> {
        let concatted = concat_instructions(expected);
        println!("len {}: {:?}", got.data.len(), got.data);
        println!("len {}: {:?}", concatted.data.len(), concatted.data);
        if got.data.len() != concatted.data.len() {
            assert_eq!(concatted.data.len(), got.data.len());
        }

        println!("{:?}", got.data);
        println!("{:?}", concatted.data);
        for (i, ins) in concatted.data.iter().enumerate() {
            assert_eq!(got.data.get(i).unwrap(), ins);
            // if got.get(i).unwrap() != ins {
            //     return Err(CompileError::Reason(format!(
            //         "wrong instruction at {}, got: {:?} wanted: {:?}",
            //         i, concatted, got
            //     )));
            // }
        }

        Ok(())
    }

    fn concat_instructions(expected: Vec<Instructions>) -> Instructions {
        let mut out: Vec<u8> = vec![];
        for e in expected {
            for b in e.data {
                out.push(b);
            }
        }
        Instructions { data: out }
    }

    fn test_constants(expected: Vec<Object>, got: Objects) -> Result<(), CompileError> {
        assert_eq!(expected.len(), got.len());

        for (i, c) in expected.iter().enumerate() {
            match c {
                Object::Number(v) => match got.get(i).unwrap() {
                    Object::Number(v2) => assert_eq!(v, v2),
                    _ => return Err(CompileError::Reason("wrong comparison types".to_string())),
                },
                //    Object::String(s) => match got.get(i).unwrap() {
                //        Object::String(s2) => assert_eq!(s, s2),
                //        _ => return Err(CompileError::Reason("wrong comparison types".to_string())),
                //    },
                //    Object::CompiledFunction {
                //        instructions: Instructions { data },
                //        num_locals,
                //        num_parameters,
                //    } => match got.get(i).unwrap() {
                //        Object::CompiledFunction {
                //            instructions: Instructions { data: data2 },
                //            num_locals: locals2,
                //            num_parameters: params2,
                //        } => {
                //            assert_eq!(data, data2);
                //            assert_eq!(num_locals, locals2);
                //            assert_eq!(num_parameters, params2);
                //        }
                //        _ => return Err(CompileError::Reason("wrong comparison types".to_string())),
                //    },
                _ => {}
            }
        }

        Ok(())
    }
}
