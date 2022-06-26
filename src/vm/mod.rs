mod frame;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::compiler::{Bytecode, Object, ObjectType, SymbolType};
use crate::{code::Opcode, compiler::Objects};

use self::frame::Frame;

const DEFAULT_STACK_SIZE: usize = 2048;

const NULL: Object = Object::Null;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum VMError {
    Reason(String),
}

pub struct Palette {
    colors: [[u8; 4]; 8],
}

pub struct VM {
    screen: Vec<u8>,
    screen_width: usize,
    screen_height: usize,
    frames: Vec<Frame>,
    frames_index: usize,
    stack: Objects,
    constants: Objects,
    sp: usize,

    last_popped: Option<Object>,
    // accumulator: u16,
    globals: Objects,
    palettes: [Palette; 3],
    debug: bool,
    time_per_op: HashMap<Opcode, Duration>,
    op_counts: HashMap<Opcode, u64>,
    halted: bool,
}

fn is_truthy(obj: Object) -> bool {
    match obj {
        Object::Number(n) => n != 0.0,
        Object::Null => false, // ?
        _ => true,
    }
}

impl VM {
    pub fn flash(
        bytecode: Bytecode,
        screen_width: usize,
        screen_height: usize,
        screen_size: usize,
        debug: bool,
    ) -> Self {
        let frames = vec![Frame::new(bytecode.instructions.clone(), 0, 0, 0)];

        // let frames = vec![Frame::new(bytecode.instructions.clone(), 0, 0, vec![], 0)];
        Self {
            op_counts: HashMap::new(),
            time_per_op: HashMap::new(),
            screen: vec![0; screen_size],
            screen_width,
            screen_height,
            frames,
            frames_index: 1,
            stack: Objects::with_capacity(DEFAULT_STACK_SIZE),
            constants: bytecode.constants,
            sp: 0,
            // accumulator: 0,
            last_popped: None,
            globals: Objects::new(),
            debug,
            halted: false,
            palettes: [
                Palette {
                    colors: [
                        // https://lospec.com/palette-list/generic-8
                        [28, 17, 33, 255],
                        [237, 236, 233, 255],
                        [161, 59, 59, 255],
                        [243, 127, 154, 255],
                        [238, 150, 26, 255],
                        [45, 83, 101, 255],
                        [64, 169, 51, 255],
                        [37, 166, 197, 255],
                    ],
                },
                Palette {
                    colors: [
                        // https://lospec.com/palette-list/pollen8
                        [115, 70, 76, 255],
                        [171, 86, 117, 255],
                        [238, 106, 124, 255],
                        [255, 167, 165, 255],
                        [255, 224, 126, 255],
                        [255, 231, 214, 255],
                        [114, 220, 187, 255],
                        [52, 172, 186, 255],
                    ],
                },
                Palette {
                    colors: [
                        // https://raw.githubusercontent.com/hundredrabbits/Themes/master/themes/orca.svg
                        [0, 0, 0, 255],
                        [34, 34, 34, 255],
                        [68, 68, 68, 255],
                        [119, 119, 119, 255],
                        [221, 221, 221, 255],
                        [255, 255, 255, 255],
                        [114, 222, 194, 255],
                        [255, 181, 69, 255],
                    ],
                },
            ],
        }
    }

    fn current_frame(&mut self) -> &Frame {
        &self.frames[self.frames_index - 1_usize]
        //return self
        //    .frames
        //    .get((self.frames_index - 1) as usize)
        //    .as_ref()
        //    .unwrap();
    }

    fn set_ip(&mut self, new_ip: i64) {
        self.frames[(self.frames_index - 1) as usize].ip = new_ip;
    }

    pub fn execute(&mut self) -> Result<u8, VMError> {
        if self.halted {
            return Ok(1);
        }

        // println!("instructions: {:?}",self.current_frame().instructions().expect("stuff").data);

        if self.current_frame().ip
            >= (self
                .current_frame()
                .instructions()
                .expect("expected instructions")
                .data
                .len() as i64
                - 1)
        {
            // let sp = self.sp;
            // println!(
            //     " ------- got opcode: ip: {} sp: {}",
            //     self.current_frame().ip,
            //     sp,
            // );
            self.halted = true;
            return Ok(1);
        }
        let start_time = Instant::now();
        let init_ip = self.current_frame().ip;
        self.set_ip((init_ip + 1) as i64);
        let ip = self.current_frame().ip as usize;
        // let instr = self.current_frame().instructions().unwrap();
        // println!("cur ip {} instr: {:?}", ip, instr);
        // self.set_ip(ip as i64);
        let cur_instructions = self
            .current_frame()
            .instructions()
            .expect("expected instructions");
        let op = Opcode::from(*cur_instructions.data.get(ip).expect("expected byte"));
        // let op = unsafe { Opcode::from(*cur_instructions.data.get_unchecked(ip)) };
        // println!(" ------- got opcode: {:?} ip: {} sp: {}", op, ip, self.sp);
        // println!("{:?}", cur_instructions.data);
        let result: u8 = match op {
            // Opcode::NoOp => 1,
            Opcode::Pop => {
                self.pop();
                1
            }
            // Opcode::RelJump => {
            //     let buff = [
            //         *cur_instructions.data.get(ip + 1).expect("expected byte"),
            //         *cur_instructions.data.get(ip + 2).expect("expected byte"),
            //     ];
            //     let rel_index = i16::from_be_bytes(buff);
            //     // println!("rel index: {}", rel_index);
            //     self.set_ip(ip as i64 + rel_index as i64 - 1);
            //     4
            // }
            // Opcode::Clear => {
            //     self.accumulator = 0;
            //     1
            // }
            // Opcode::Increment => {
            //     // println!("accumulate!");
            //     self.accumulator = self.accumulator.wrapping_add(1);
            //     self.accumulator %= self.screen.capacity() as u16;
            //     1
            // }
            // Opcode::Decrement => {
            //     self.accumulator = self.accumulator.wrapping_sub(1);
            //     if self.accumulator >= self.screen.capacity() as u16 {
            //         self.accumulator = self.screen.capacity() as u16 - 1
            //     }
            //     1
            // }
            // Opcode::PushAccumulator => {
            //     // println!("push acc");
            //     self.push2(self.accumulator)?;
            //     2
            // }
            // Opcode::LoadAccumulator => {
            //     let buff = [self.pop(), self.pop()];

            //     let accumulator_val = u16::from_be_bytes(buff);
            //     self.accumulator = accumulator_val;
            //     2
            // }
            // Opcode::ApplyScreen => {
            //     let pixel_value = self.pop();
            //     let buff = [self.pop(), self.pop()];

            //     let screen_pixel_index = u16::from_be_bytes(buff);

            //     // std::mem::replace(&mut self.screen[screenPixelIndex as usize], *constVal);
            //     self.screen[screen_pixel_index as usize] = pixel_value;
            //     // println!(
            //     //     "index to value {}:{}",
            //     //     screen_pixel_index as usize, pixel_value
            //     // );
            //     3
            // }
            Opcode::Constant => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];

                let const_index = u16::from_be_bytes(buff);
                let new_ip = self.current_frame().ip + op.operand_width() as i64;
                self.set_ip(new_ip);
                let val = self.constants[const_index as usize].to_owned();
                self.push(val)?;

                2
            }
            Opcode::GetGlobal => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];
                let global_index = u16::from_be_bytes(buff);
                self.set_ip((ip + 2) as i64);
                let val: Object = self.globals.get(global_index as usize).unwrap().clone();
                self.push(val)?;
                2
            }
            Opcode::SetGlobal => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];
                let global_index = u16::from_be_bytes(buff);
                self.set_ip((ip + 2) as i64);
                let pop = self.pop();
                self.globals.insert(global_index as usize, pop);
                2
            }
            Opcode::Add
            | Opcode::Divide
            | Opcode::Multiply
            | Opcode::Subtract
            | Opcode::And
            | Opcode::Or
            | Opcode::LessThan
            | Opcode::LessThanEqual
            | Opcode::GreaterThan
            | Opcode::GreaterThanEqual
            | Opcode::NotEqual
            | Opcode::Equal => self.execute_binary_operation(op.clone())?,
            Opcode::True => todo!(),
            Opcode::False => todo!(),
            Opcode::Minus => {
                self.execute_minus_operator()?;
                // TODO: figure out how to reduce this cost without allowing things to loop forever?
                1
            }
            Opcode::Bang => {
                self.execute_bang_operator()?;
                1
            }
            Opcode::Jump => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];
                let jump_target = u16::from_be_bytes(buff);
                self.set_ip(jump_target as i64 - 1);

                // maybe lower cost?
                2
            }
            Opcode::JumpNotTruthy => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];
                let jump_target = u16::from_be_bytes(buff);
                self.set_ip((ip + 2) as i64);
                let condition = self.pop();
                if !is_truthy(condition) {
                    self.set_ip((jump_target - 1) as i64);
                }

                // maybe lower cost?
                4
            }
            Opcode::Null => {
                // TODO: zero?
                self.push(NULL)?;
                1
            }
            Opcode::Array => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];
                let element_count = u16::from_be_bytes(buff);
                self.set_ip((ip + 2) as i64);
                let array = self.build_array(self.sp - element_count as usize, self.sp);
                self.sp -= element_count as usize;
                self.push(array)?;
                1
            }
            Opcode::Hash => todo!(),
            Opcode::Index => {
                let index = self.pop();
                let left = self.pop();
                self.execute_index_expression(left, index)?;
                1
            }
            Opcode::Call => todo!(),
            Opcode::ReturnValue => todo!(),
            Opcode::Return => todo!(),
            Opcode::GetLocal => todo!(),
            Opcode::SetLocal => todo!(),
            Opcode::Reduce => self.execute_reduce_operation()?,
            Opcode::Piset => {
                self.execute_piset()?;
                1
            }
        };

        if self.debug {
            let amount = match self.time_per_op.get(&op) {
                Some(time_so_far) => *time_so_far + start_time.elapsed(),
                None => start_time.elapsed(),
            };
            self.time_per_op.insert(op.clone(), amount);
            let count = match self.op_counts.get(&op) {
                Some(v) => v + 1,
                None => 1,
            };
            self.op_counts.insert(op, count);
        }
        Ok(result)
    }

    fn arith_objects(op: SymbolType, a: Object, b: Object) -> Result<Object, VMError> {
        match (a, b) {
            (Object::Number(an), Object::Number(bn)) => match op {
                SymbolType::PLUS => Ok(Object::Number(an + bn)),
                SymbolType::MINUS => Ok(Object::Number(an - bn)),
                SymbolType::ASTERISK => Ok(Object::Number(an * bn)),
                SymbolType::SLASH => Ok(Object::Number(an / bn)),
                SymbolType::AND => Ok(Object::Number(an * bn)),
                SymbolType::OR => Ok(Object::Number((an + bn) - (an * bn))),
                SymbolType::NOTEQUAL => {
                    if f64::abs(an - bn) > f64::EPSILON {
                        Ok(Object::Number(1.0))
                    } else {
                        Ok(Object::Number(0.0))
                    }
                }
                SymbolType::EQUAL => {
                    if f64::abs(an - bn) < f64::EPSILON {
                        Ok(Object::Number(1.0))
                    } else {
                        Ok(Object::Number(0.0))
                    }
                }
                SymbolType::LESSTHAN => {
                    if an < bn {
                        Ok(Object::Number(1.0))
                    } else {
                        Ok(Object::Number(0.0))
                    }
                }
                SymbolType::LESSTHANEQUAL => {
                    if an <= bn {
                        Ok(Object::Number(1.0))
                    } else {
                        Ok(Object::Number(0.0))
                    }
                }
                SymbolType::GREATERTHAN => {
                    if an > bn {
                        Ok(Object::Number(1.0))
                    } else {
                        Ok(Object::Number(0.0))
                    }
                }
                SymbolType::GREATERTHANEQUAL => {
                    if an >= bn {
                        Ok(Object::Number(1.0))
                    } else {
                        Ok(Object::Number(0.0))
                    }
                }
                _ => Err(VMError::Reason(format!(
                    "Unexpected operation type: {:?}",
                    op
                ))),
            },
            (Object::Array(arr), Object::Number(num))
            | (Object::Number(num), Object::Array(arr)) => {
                let mut result = vec![];
                for i in arr {
                    match VM::arith_objects(op.clone(), i, Object::Number(num)) {
                        Ok(val) => result.push(val),
                        Err(err) => return Err(err),
                    }
                }

                Ok(Object::Array(result))
            }
            (Object::Array(arr1), Object::Array(arr2)) => {
                if arr1.len() != arr2.len() {
                    return Err(VMError::Reason(format!(
                        "Mismatch array length ({} and {})",
                        arr1.len(),
                        arr2.len()
                    )));
                }
                let mut result = vec![];
                for (aa, bb) in arr1.iter().zip(arr2.iter()) {
                    match VM::arith_objects(op.clone(), aa.to_owned(), bb.to_owned()) {
                        Ok(val) => result.push(val),
                        Err(err) => return Err(err),
                    }
                }
                Ok(Object::Array(result))
            }
            (a, b) => Err(VMError::Reason(format!(
                "Unexpected addition of types {:?} and {:?}",
                a, b
            ))),
        }
    }

    fn execute_array_reduce(
        &mut self,
        operation: SymbolType,
        target: Vec<Object>,
    ) -> Result<(), VMError> {
        let mut iter = target.iter();
        let mut result_object: Object = iter.next().unwrap().to_owned();
        for obj in iter {
            result_object =
                match VM::arith_objects(operation.clone(), result_object, obj.to_owned()) {
                    Ok(res) => res,
                    Err(err) => return Err(err),
                };
        }
        self.push(result_object)?;

        Ok(())
    }

    fn execute_reduce_operation(&mut self) -> Result<u8, VMError> {
        let operation = self.pop();
        let target = self.pop();

        let cycles: u8;
        match (target, operation) {
            (Object::Number(n), _) => {
                // self.execute_binary_integer_operation(op, left, right)?;
                cycles = 1;
                self.push(Object::Number(n))?
            }
            (Object::Array(array_target), Object::Symbol(symbol_type)) => {
                cycles = 4;
                self.execute_array_reduce(symbol_type, array_target)?;
            }
            // (ObjectType::String, ObjectType::String) => {
            //     self.execute_binary_string_operation(op, left, right)?;
            // }
            (a, b) => {
                return Err(VMError::Reason(format!(
                    "Unsupported reduce action for {:?} and {:?}",
                    a, b
                )))
            }
        };

        // self.push(Object::Int(left + right))?;

        Ok(cycles)
    }

    fn execute_binary_operation(&mut self, op: Opcode) -> Result<u8, VMError> {
        let right = self.pop();
        let left = self.pop();
        let operation = match op {
            Opcode::Subtract => SymbolType::MINUS,
            Opcode::Add => SymbolType::PLUS,
            Opcode::Multiply => SymbolType::ASTERISK,
            Opcode::Divide => SymbolType::SLASH,
            Opcode::And => SymbolType::AND,
            Opcode::Or => SymbolType::OR,
            Opcode::Equal => SymbolType::EQUAL,
            Opcode::NotEqual => SymbolType::NOTEQUAL,
            Opcode::LessThan => SymbolType::LESSTHAN,
            Opcode::LessThanEqual => SymbolType::LESSTHANEQUAL,
            Opcode::GreaterThan => SymbolType::GREATERTHAN,
            Opcode::GreaterThanEqual => SymbolType::GREATERTHANEQUAL,
            _ => {
                return Err(VMError::Reason(format!(
                    "Unsupported operation for {:?} and {:?}",
                    left, right
                )))
            }
        };

        let left_size: u8;
        let right_size: u8;

        match (left.object_type(), right.object_type()) {
            (ObjectType::Number, ObjectType::Number)
            | (ObjectType::Array, ObjectType::Number)
            | (ObjectType::Array, ObjectType::Array) => {
                left_size = match left.object_type() {
                    ObjectType::Array => 4,
                    _ => 1,
                };
                right_size = match right.object_type() {
                    ObjectType::Array => 4,
                    _ => 1,
                };
                let result = VM::arith_objects(operation, left, right)?;
                self.push(result)?;
            }
            (a, b) => {
                return Err(VMError::Reason(format!(
                    "Unsupported binary action for {:?} and {:?}",
                    a, b
                )))
            }
        };

        Ok(right_size + left_size)
    }

    fn push(&mut self, obj: Object) -> Result<(), VMError> {
        if self.sp >= self.stack.capacity() {
            return Err(VMError::Reason("Stack overflow".to_string()));
        }

        self.stack.insert(self.sp, obj);
        self.sp += 1;
        Ok(())
    }

    fn build_array(&mut self, start_index: usize, end_index: usize) -> Object {
        let mut elements: Vec<Object> = vec![];
        if start_index != end_index {
            for pos in start_index..end_index {
                let item = self
                    .stack
                    .get(pos)
                    .expect("expected a valid index position")
                    .clone();
                elements.push(item);
            }
        }

        Object::Array(elements)
    }

    fn execute_index_expression(&mut self, left: Object, index: Object) -> Result<(), VMError> {
        if left.object_type() == ObjectType::Array && index.object_type() == ObjectType::Number {
            self.execute_array_index(left, index)
        // } else if left.object_type() == ObjectType::Hash {
        // return self.execute_hash_index(left, index);
        } else {
            return Err(VMError::Reason(format!(
                "index operator not supported for {:?}",
                left.object_type()
            )));
        }
    }

    fn execute_array_index(&mut self, left: Object, index: Object) -> Result<(), VMError> {
        let index_val = match index {
            Object::Number(i) => i,
            _ => {
                return Err(VMError::Reason(format!(
                    "expected array type, but got {:?}",
                    left.object_type()
                )))
            }
        } as i64;

        match left {
            Object::Array(a) => {
                let max = (a.len() - 1) as i64;
                if index_val < 0 || index_val > max {
                    return Err(VMError::Reason(format!(
                        "index[{}] is out of bounds (max {})",
                        index_val, max,
                    )));
                }

                return self.push(
                    a.get(index_val as usize)
                        .expect("expected a value from index")
                        .clone(),
                );
            }
            _ => {
                return Err(VMError::Reason(format!(
                    "expected array type, but got {:?}",
                    left.object_type()
                )))
            }
        }
    }

    fn is_not_zero(arr: Vec<Object>) -> Vec<Object> {
        let result: Vec<Object> = arr
            .iter()
            .map(|obj| match obj {
                Object::Number(n) => {
                    if *n == 0.0 {
                        Object::Number(1.0)
                    } else {
                        Object::Number(0.0)
                    }
                }
                Object::Array(ar) => Object::Array(VM::is_not_zero(ar.to_owned())),
                _ => Object::Number(0.0),
            })
            .collect::<Vec<Object>>();

        result
    }

    fn execute_bang_operator(&mut self) -> Result<(), VMError> {
        let op = self.pop();

        match op {
            Object::Number(n) => {
                if n == 0.0 {
                    self.push(Object::Number(1.0))
                } else {
                    self.push(Object::Number(0.0))
                }
            }
            Object::Array(arr) => self.push(Object::Array(VM::is_not_zero(arr))),
            Object::Null => self.push(Object::Number(1.0)),
            _ => self.push(Object::Number(0.0)),
        }
    }
    fn execute_minus_operator(&mut self) -> Result<(), VMError> {
        let op = self.pop();

        match op {
            Object::Number(n) => self.push(Object::Number(-n)),
            _ => Err(VMError::Reason(format!(
                "unsupported minus type: {:?} for {:?}",
                op.object_type(),
                op,
            ))),
        }
    }

    // fn push2(&mut self, obj: u16) -> Result<(), VMError> {
    //     if self.sp >= self.stack.capacity() {
    //         return Err(VMError::Reason("Stack overflow".to_string()));
    //     }

    //     let [high, low] = obj.to_be_bytes();

    //     self.stack.insert(self.sp, low);
    //     self.sp += 1;
    //     self.stack.insert(self.sp, high);
    //     self.sp += 1;
    //     Ok(())
    // }

    pub fn pop(&mut self) -> Object {
        let val = self.stack.remove(self.sp - 1);
        self.last_popped = Some(val.clone());
        self.sp -= 1;
        val
    }

    fn execute_piset(&mut self) -> Result<(), VMError> {
        let color = self.pop();
        let pal = self.pop();
        let y = self.pop();
        let x = self.pop();

        println!("x {:?} y {:?}", x, y);

        let xval = match x {
            Object::Number(n) => n as usize,
            v => {
                return Err(VMError::Reason(format!(
                    "invalid x value for piset {:?}",
                    v
                )))
            }
        };

        let yval = match y {
            Object::Number(n) => n as usize,
            v => {
                return Err(VMError::Reason(format!(
                    "invalid y value for piset {:?}",
                    v
                )))
            }
        };

        let colorval = match color {
            Object::Number(n) => n as u8 & 0b0000_1111,
            v => {
                return Err(VMError::Reason(format!(
                    "invalid color value for piset {:?}",
                    v
                )))
            }
        };

        let paletteval = match pal {
            Object::Number(n) => n as u8 & 0b0000_1111,
            v => {
                return Err(VMError::Reason(format!(
                    "invalid palette value for piset {:?}",
                    v
                )))
            }
        };

        println!(
            "x {} y {} pal {} color {}",
            xval, yval, paletteval, colorval
        );

        let pixel = (paletteval << 4) + colorval;
        self.screen[self.screen_width * yval + xval] = pixel;

        Ok(())
    }

    pub fn update_screen(&mut self, screen: &mut [u8]) {
        for (i, pixel) in screen.chunks_exact_mut(4).enumerate() {
            let val = unsafe { self.screen.get_unchecked(i) };
            let palette = (val >> 4) as u8 % self.palettes.len() as u8;
            let color = (val & 0b0000_1111) % self.palettes[palette as usize].colors.len() as u8;
            // println!("palette: {} color: {} val: {}", palette, color, val);
            pixel.copy_from_slice(&self.palettes[palette as usize].colors[color as usize]);
        }
    }

    pub fn ip(&mut self) -> i64 {
        self.current_frame().ip
    }

    pub fn sp(&self) -> usize {
        self.sp
    }

    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }

    pub fn globals_size(&self) -> usize {
        self.globals.len()
    }

    pub fn operation_times(&self) -> HashMap<String, Duration> {
        self.time_per_op
            .iter()
            .map(|(op, dur)| {
                return (
                    format!("{:?}", op),
                    Duration::from_secs_f64(
                        dur.as_secs_f64() / *self.op_counts.get(op).unwrap() as f64,
                        // dur.as_secs_f64(),
                    ),
                );
            })
            .collect()
    }

    fn last_popped(&self) -> Option<Object> {
        self.last_popped.clone()
    }

    pub fn halted(&self) -> bool {
        self.halted
    }
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    // use crate::code::*;
    // use crate::compiler::symbol_table::SymbolTable;
    use crate::compiler::*;
    // use crate::evaluator::object::*;
    // use crate::lexer;
    use crate::parser;
    use crate::parser::lexer;
    use crate::vm::VM;
    // use std::cell::RefCell;
    // use std::collections::HashMap;
    // use std::rc::Rc;

    struct VMTestCase {
        input: String,
        expected_top: Option<Object>,
        expected_cycles: i32,
    }

    #[test]
    fn test_arithmetic() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(2.0)),
            input: "1+1".to_string(),
            expected_cycles: 7,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_arithmetic_2() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(-6.0)),
            input: "5 - 10 - 1".to_string(),
            expected_cycles: 11,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_add_reduce() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(15.0)),
            input: "[1,2,3,4,5]\\+".to_string(),
            expected_cycles: 18,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_add_reduce_inner_array() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Array(vec![
                Object::Number(6.0),
                Object::Number(7.0),
                Object::Number(8.0),
            ])),
            input: "[[1, 2, 3],5]\\+".to_string(),
            expected_cycles: 17,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_add_reduce_inner_arrays() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Array(vec![
                Object::Number(5.0),
                Object::Number(7.0),
                Object::Number(9.0),
            ])),
            input: "[[1, 2, 3], [4, 5, 6]]\\+".to_string(),
            expected_cycles: 22,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_inner_arrays_multiply() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Array(vec![
                Object::Number(4.0),
                Object::Number(10.0),
                Object::Number(18.0),
            ])),
            input: "[[1, 2, 3], [4, 5, 6]]\\*".to_string(),
            expected_cycles: 22,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_arrays_multiply() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(720.0)),
            input: "[1, 2, 3, 4, 5, 6]\\*".to_string(),
            expected_cycles: 20,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_inner_arrays_div() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Array(vec![
                Object::Number(0.25),
                Object::Number(2.0 / 5.0),
                Object::Number(0.5),
            ])),
            input: "[[1, 2, 3], [4, 5, 6]]\\/".to_string(),
            expected_cycles: 22,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_arrays_sub() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(0.0)),
            input: "[100, 25, 25, 25, 25]\\-".to_string(),
            expected_cycles: 18,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_arrays_sub_zero_one() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(-1.0)),
            input: "[0, 1]\\-".to_string(),
            expected_cycles: 12,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_arrays_sub_one_zero() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(1.0)),
            input: "[1, 0]\\-".to_string(),
            expected_cycles: 12,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_arrays_inner_arrays_sub() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Array(vec![
                Object::Number(-20.0),
                Object::Number(-20.0),
            ])),
            input: "[[10, 20], [30, 40]]\\-".to_string(),
            expected_cycles: 18,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_reduce_arrays_inner_arrays_sub_2() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Array(vec![
                Object::Number(60.0),
                Object::Number(40.0),
            ])),
            input: "[[100, 100], [10, 20], [30, 40]]\\-".to_string(),
            expected_cycles: 23,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_add_reduce_inner_array_into_value() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(21.0)),
            input: "a <- [1, 2, 3, 4, 5, 6]\\+;a".to_string(),
            expected_cycles: 24,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_if_statement() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(2.0)),
                input: "if(1) {2};".to_string(),
                expected_cycles: 13,
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "if(0) {2};".to_string(),
                expected_cycles: 13,
            },
            VMTestCase {
                expected_top: Some(Object::Number(2.0)),
                input: "if(1) {2} else {3};".to_string(),
                expected_cycles: 13,
            },
            VMTestCase {
                expected_top: Some(Object::Number(3.0)),
                input: "if(0) {2} else {3};".to_string(),
                expected_cycles: 13,
            },
            VMTestCase {
                expected_top: Some(Object::Number(2.0)),
                input: "if([0,0,0,1]\\+) {2} else {3};".to_string(),
                expected_cycles: 26,
            },
            VMTestCase {
                expected_top: Some(Object::Number(3.0)),
                input: "if([0,0,0,0]\\+) {2} else {3};".to_string(),
                expected_cycles: 26,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_while_loop() {
        // "a <- 0; while (a < 10) { a <- 10 };"

        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(10.0)),
                input: "a <- 0; while (a < 10) { a <- a + 1 };a".to_string(),
                expected_cycles: 330,
            },
            VMTestCase {
                expected_top: Some(Object::Number(100.0)),
                input: "a <- 0; while (a < 10) { a <- 100 };a".to_string(),
                expected_cycles: 35,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_negative_number() {
        let tests: Vec<VMTestCase> = vec![VMTestCase {
            expected_top: Some(Object::Number(-10.0)),
            input: "-10".to_string(),
            expected_cycles: 4,
        }];

        run_vm_test(tests);
    }

    #[test]
    fn test_bang() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "!1".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "!0".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "a<-1;!a".to_string(),
                expected_cycles: 9,
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(0.0),
                    Object::Number(1.0),
                    Object::Number(0.0),
                ])),
                input: "![-10, 0, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }
    #[test]
    fn test_not_equal_operator() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10.5!=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10!=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10!=10.5".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "(0.1+0.2)!=0.3".to_string(),
                expected_cycles: 11,
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(0.0),
                    Object::Number(1.0),
                    Object::Number(1.0),
                ])),
                input: "[-10, 1, 4]!=[-10, 4, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_equal_operator() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10.5=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10=10.5".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "(0.1+0.2)=0.3".to_string(),
                expected_cycles: 11,
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(1.0),
                    Object::Number(0.0),
                    Object::Number(0.0),
                ])),
                input: "[-10, 1, 4]=[-10, 4, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_lessthan_operator() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10.5<10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10<10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10<10.5".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "(0.1+0.2)<0.3".to_string(),
                expected_cycles: 11,
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(0.0),
                    Object::Number(1.0),
                    Object::Number(0.0),
                ])),
                input: "[-10, 1, 4]<[-10, 4, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_lessthanequal_operator() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10.5<=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10<=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10<=10.5".to_string(),
                expected_cycles: 7,
            },
            // TODO: Fix somehow
            // VMTestCase {
            //     expected_top: Some(Object::Number(1.0)),
            //     input: "(0.1+0.2)≤0.3".to_string(),
            // },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(1.0),
                    Object::Number(1.0),
                    Object::Number(0.0),
                ])),
                input: "[-10, 1, 4]<=[-10, 4, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_greaterthan_operator() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10.5>10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10>10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10>10.5".to_string(),
                expected_cycles: 7,
            },
            // TODO: gotta fix somehow
            // VMTestCase {
            //     expected_top: Some(Object::Number(0.0)),
            //     input: "(0.1+0.2)>0.3".to_string(),
            // },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(0.0),
                    Object::Number(0.0),
                    Object::Number(1.0),
                ])),
                input: "[-10, 1, 4]>[-10, 4, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_greaterthanequal_operator() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10.5>=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "10>=10".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(0.0)),
                input: "10>=10.5".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(1.0)),
                input: "(0.1+0.2)>=0.3".to_string(),
                expected_cycles: 11,
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Number(1.0),
                    Object::Number(0.0),
                    Object::Number(1.0),
                ])),
                input: "[-10, 1, 4]>=[-10, 4, 1]".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_let() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(10.0)),
                input: "a <- 10; a".to_string(),
                expected_cycles: 7,
            },
            VMTestCase {
                expected_top: Some(Object::Number(11.0)),
                input: "a <- 10; a <- a + 1; a;".to_string(),
                expected_cycles: 25,
            },
        ];

        run_vm_test(tests);
    }

    // #[test]
    // fn test_reassignment() {
    //     let tests: Vec<VMTestCase> = vec![VMTestCase {
    //         expected_top: Some(Object::Number(2000.0)),
    //         input: r#"a <- [10,20,30,40]\+;
    //         a <- a * 20"#
    //             .to_string(),
    //         expected_cycles: 30,
    //     }];

    //     run_vm_test(tests);
    // }

    #[test]
    fn test_array() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Number(10.0)),
                input: "a <- [10]; a[1-1]".to_string(),
                expected_cycles: 15,
            },
            VMTestCase {
                expected_top: Some(Object::Number(10.0)),
                input: "a <- [[[10]]]; a[1-1][0][100-100]".to_string(),
                expected_cycles: 27,
            },
        ];

        run_vm_test(tests);
    }

    fn parse(input: String) -> parser::ast::Program {
        let l = lexer::Lexer::new(input);
        let mut p = parser::Parser::new(l);
        p.build_ast()
    }

    fn run_vm_test(tests: Vec<VMTestCase>) {
        for test in tests {
            println!("--- testing: {}", test.input);
            let ast = parse(test.input);
            let mut c = Compiler::new();
            let compile_result = c.read(ast);
            assert!(compile_result.is_ok());

            let mut vmm = VM::flash(c.bytecode(), 0, 0, 0, false);
            let mut cycles = 0;
            while cycles < test.expected_cycles {
                let result = vmm.execute();
                assert!(
                    result.is_ok(),
                    "got error on cycle {}: {:?}",
                    cycles,
                    result.unwrap_err()
                );
                cycles += result.expect("expected value") as i32;
            }
            if test.expected_top.clone().is_some() {
                let stack_elem = vmm.last_popped();
                assert_eq!(stack_elem, test.expected_top);
            }
        }
    }
}
