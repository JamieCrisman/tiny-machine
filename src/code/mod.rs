use crate::compiler::Instructions;

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Opcode {
    // NoOp = 0,
    // RelJump = 1,
    // Constant = 2,
    // Clear = 3,
    // Increment = 4,
    // Decrement = 5,
    // PushAccumulator = 6,
    // LoadAccumulator = 7,
    // ApplyScreen = 8,
    // GetGlobal = 9,
    // SetGlobal = 10,
    // Pop = 11,
    Constant = 0,
    Pop = 1,
    Add = 2,
    Subtract = 3,
    Multiply = 4,
    Divide = 5,
    True = 6,
    False = 7,
    Equal = 8,
    NotEqual = 9,
    GreaterThan = 10,
    Minus = 11,
    Bang = 12,
    Jump = 13,
    JumpNotTruthy = 14,
    Null = 15,
    GetGlobal = 16,
    SetGlobal = 17,
    Array = 18,
    Hash = 19,
    Index = 20,
    Call = 21,
    ReturnValue = 22,
    Return = 23,
    GetLocal = 24,
    SetLocal = 25,
    // BuiltinFunc = 26,
    // Closure = 27,
    // GetFree = 28,
    // CurrentClosure = 29,
    Reduce = 30,
    // Tally = 31,
    // Swap = 32,
    // Without = 33,
    // Not = 34,
    // Pick = 35,
    // First = 36,
    // Floor = 37,
    // Min = 38,
    // Ceil = 39,
    // Max = 40,
    // Magnitude = 41,
    // Modulus = 42,
    And = 43,
    Or = 44,
    LessThan = 45,         // could do without?
    LessThanEqual = 46,    // could do without?
    GreaterThanEqual = 47, // could do without?
    Piset = 48,
}

impl From<u8> for Opcode {
    fn from(orig: u8) -> Self {
        match orig {
            // 0 => Opcode::NoOp,
            // 1 => Opcode::RelJump,
            // 2 => Opcode::Constant,
            // 3 => Opcode::Clear,
            // 4 => Opcode::Increment,
            // 5 => Opcode::Decrement,
            // 6 => Opcode::PushAccumulator,
            // 7 => Opcode::LoadAccumulator,
            // 8 => Opcode::ApplyScreen,
            // 9 => Opcode::GetGlobal,
            // 10 => Opcode::SetGlobal,
            // 11 => Opcode::Pop,
            // _ => panic!("Unknown opcode {}", orig)
            0 => Opcode::Constant,
            1 => Opcode::Pop,
            2 => Opcode::Add,
            3 => Opcode::Subtract,
            4 => Opcode::Multiply,
            5 => Opcode::Divide,
            6 => Opcode::True,
            7 => Opcode::False,
            8 => Opcode::Equal,
            9 => Opcode::NotEqual,
            10 => Opcode::GreaterThan,
            11 => Opcode::Minus,
            12 => Opcode::Bang,
            13 => Opcode::Jump,
            14 => Opcode::JumpNotTruthy,
            15 => Opcode::Null,
            16 => Opcode::GetGlobal,
            17 => Opcode::SetGlobal,
            18 => Opcode::Array,
            19 => Opcode::Hash,
            20 => Opcode::Index,
            21 => Opcode::Call,
            22 => Opcode::ReturnValue,
            23 => Opcode::Return,
            24 => Opcode::GetLocal,
            25 => Opcode::SetLocal,
            // 26 => Opcode::BuiltinFunc,
            // 27 => Opcode::Closure,
            // 28 => Opcode::GetFree,
            // 29 => Opcode::CurrentClosure,
            30 => Opcode::Reduce,
            // 31 => Opcode::Tally,
            // 32 => Opcode::Swap,
            // 33 => Opcode::Without,
            // 34 => Opcode::Not,
            // 35 => Opcode::Pick,
            // 36 => Opcode::First,
            // 37 => Opcode::Floor,
            // 38 => Opcode::Min,
            // 39 => Opcode::Ceil,
            // 40 => Opcode::Max,
            // 41 => Opcode::Magnitude,
            // 42 => Opcode::Modulus,
            43 => Opcode::And,
            44 => Opcode::Or,
            45 => Opcode::LessThan,         // could do without?
            46 => Opcode::LessThanEqual,    // could do without?
            47 => Opcode::GreaterThanEqual, // could do without?
            48 => Opcode::Piset,
            _ => panic!("Unknown value: {}", orig),
        }
    }
}

impl Opcode {
    pub fn widths(&self) -> Vec<i16> {
        match self {
            // Opcode::NoOp
            // | Opcode::Increment
            // | Opcode::Decrement
            // | Opcode::PushAccumulator
            // | Opcode::Pop
            // | Opcode::Clear => vec![],
            // Opcode::Constant
            // | Opcode::SetGlobal
            // | Opcode::ApplyScreen
            // | Opcode::GetGlobal
            // | Opcode::RelJump
            // | Opcode::LoadAccumulator => {
            //     vec![2]
            // }
            Opcode::Constant
            | Opcode::Jump
            | Opcode::JumpNotTruthy
            | Opcode::GetGlobal
            | Opcode::SetGlobal
            | Opcode::Array
            | Opcode::Hash
            => vec![2],
            Opcode::SetLocal
            | Opcode::GetLocal
            | Opcode::Call
            // | Opcode::BuiltinFunc
            // | Opcode::GetFree 
            => vec![1],
            // Opcode::Closure => Some(vec![2, 1]),
            Opcode::Add
            | Opcode::Divide
            | Opcode::Subtract
            | Opcode::Multiply
            | Opcode::Pop
            | Opcode::True
            | Opcode::GreaterThan
            | Opcode::Equal
            | Opcode::NotEqual
            | Opcode::Minus
            | Opcode::Bang
            | Opcode::False
            | Opcode::Null
            | Opcode::Index
            | Opcode::ReturnValue
            | Opcode::Return
            | Opcode::Piset
            // | Opcode::CurrentClosure
            | Opcode::Reduce
            // | Opcode::Tally
            // | Opcode::Swap
            // | Opcode::Without
            // | Opcode::Not
            // | Opcode::Pick
            // | Opcode::First
            // | Opcode::Floor
            // | Opcode::Min
            // | Opcode::Max
            // | Opcode::Ceil
            // | Opcode::Magnitude
            // | Opcode::Modulus
            | Opcode::And
            | Opcode::Or
            | Opcode::LessThan
            | Opcode::LessThanEqual
            | Opcode::GreaterThanEqual => vec![],
        }
    }

    pub fn operand_width(&self) -> usize {
        match self {
            // Opcode::NoOp
            // | Opcode::Clear
            // | Opcode::Increment
            // | Opcode::Decrement
            // | Opcode::PushAccumulator
            // | Opcode::Pop
            // | Opcode::LoadAccumulator => 0,
            // Opcode::Constant  | Opcode::RelJump | Opcode::ApplyScreen | Opcode::SetGlobal | Opcode::GetGlobal => {
            //     self.widths().iter().sum::<i16>() as usize
            // }
            Opcode::Constant
            | Opcode::Jump
            | Opcode::JumpNotTruthy
            | Opcode::GetGlobal
            | Opcode::SetGlobal
            | Opcode::Array
            | Opcode::SetLocal
            | Opcode::GetLocal
            | Opcode::Call
            | Opcode::Hash
            // | Opcode::Closure
            // | Opcode::BuiltinFunc
            // | Opcode::GetFree 
            => self.widths().iter().sum::<i16>() as usize, // expensive way to say 1,2
            Opcode::Add
            | Opcode::Divide
            | Opcode::Subtract
            | Opcode::Multiply
            | Opcode::Pop
            | Opcode::True
            | Opcode::GreaterThan
            | Opcode::Equal
            | Opcode::NotEqual
            | Opcode::Minus
            | Opcode::Bang
            | Opcode::False
            | Opcode::Null
            | Opcode::Index
            | Opcode::ReturnValue
            | Opcode::Return
            // | Opcode::CurrentClosure
            | Opcode::Reduce
            // | Opcode::Tally
            // | Opcode::Swap
            // | Opcode::Without
            // | Opcode::Not
            // | Opcode::Pick
            // | Opcode::First
            // | Opcode::Floor
            // | Opcode::Min
            // | Opcode::Max
            // | Opcode::Ceil
            // | Opcode::Magnitude
            // | Opcode::Modulus
            | Opcode::Piset
            | Opcode::And
            | Opcode::Or
            | Opcode::LessThan
            | Opcode::LessThanEqual
            | Opcode::GreaterThanEqual => 0,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum MakeError {
    Reason(String),
}

pub fn make(op: Opcode, operands: Option<Vec<i32>>) -> Result<Instructions, MakeError> {
    let mut instruction_len = 1;
    instruction_len += op.operand_width();
    let mut result = vec![0; instruction_len as usize];
    result[0] = op.clone() as u8;
    if instruction_len > 1 {
        if operands.is_none() {
            return Err(MakeError::Reason(format!(
                "Expected operand for opcode {:?}",
                op
            )));
        }

        let mut offset = 1_usize;
        let ww = op.widths();
        for (i, o) in operands.unwrap().iter().enumerate() {
            let w = ww.get(i);
            if let Some(v) = w {
                match v {
                    2 => {
                        for (ii, b) in (*o as i16).to_be_bytes().iter().enumerate() {
                            result[offset + ii] = *b;
                        }
                    }
                    1 => {
                        for b in (*o as i16).to_be_bytes().iter() {
                            result[offset] = *b;
                        }
                    }
                    _ => {}
                }
                offset += *v as usize;
            }
        }
    }
    Ok(Instructions { data: result })
}
