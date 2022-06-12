mod frame;
mod op;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::compiler::Bytecode;

use self::{frame::Frame, op::Opcode};

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum VMError {
    Reason(String),
}

pub struct Palette {
    colors: [[u8; 4]; 8],
}

pub struct VM {
    screen: Vec<u8>,
    frames: Vec<Frame>,
    frames_index: usize,
    stack: Vec<u8>,
    constants: Vec<u8>,
    sp: usize,

    accumulator: u16,
    globals: Vec<u8>,
    palettes: [Palette; 3],
    debug: bool,
    time_per_op: HashMap<Opcode, Duration>,
    op_counts: HashMap<Opcode, u64>,
}

impl VM {
    pub fn flash(bytecode: Bytecode, screen_size: usize, debug: bool) -> Self {
        let frames = vec![Frame::new(bytecode.instructions.clone(), 0, 0, 0)];
        let globals = vec![0; 9999];

        // let frames = vec![Frame::new(bytecode.instructions.clone(), 0, 0, vec![], 0)];
        Self {
            op_counts: HashMap::new(),
            time_per_op: HashMap::new(),
            screen: vec![0; screen_size],
            frames,
            frames_index: 1,
            stack: Vec::with_capacity(9999),
            constants: bytecode.constants,
            sp: 0,
            accumulator: 0,
            globals,
            debug,
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
        return self
            .frames
            .get((self.frames_index - 1) as usize)
            .as_ref()
            .unwrap();
    }

    fn set_ip(&mut self, new_ip: i64) {
        self.frames[(self.frames_index - 1) as usize].ip = new_ip;
    }

    pub fn execute(&mut self) -> Result<u8, VMError> {
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
            return Err(VMError::Reason("Reached end".to_string()));
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
            Opcode::NoOp => 1,
            Opcode::Pop => {
                self.pop();
                1
            }
            Opcode::RelJump => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];
                let rel_index = i16::from_be_bytes(buff);
                // println!("rel index: {}", rel_index);
                self.set_ip(ip as i64 + rel_index as i64 - 1);
                4
            }
            Opcode::Clear => {
                self.accumulator = 0;
                1
            }
            Opcode::Increment => {
                // println!("accumulate!");
                self.accumulator = self.accumulator.wrapping_add(1);
                self.accumulator %= self.screen.capacity() as u16;
                1
            }
            Opcode::Decrement => {
                self.accumulator = self.accumulator.wrapping_sub(1);
                if self.accumulator >= self.screen.capacity() as u16 {
                    self.accumulator = self.screen.capacity() as u16 - 1
                }
                1
            }
            Opcode::PushAccumulator => {
                // println!("push acc");
                self.push2(self.accumulator)?;
                2
            }
            Opcode::LoadAccumulator => {
                let buff = [self.pop(), self.pop()];

                let accumulator_val = u16::from_be_bytes(buff);
                self.accumulator = accumulator_val;
                2
            }
            Opcode::ApplyScreen => {
                let pixel_value = self.pop();
                let buff = [self.pop(), self.pop()];

                let screen_pixel_index = u16::from_be_bytes(buff);

                // std::mem::replace(&mut self.screen[screenPixelIndex as usize], *constVal);
                self.screen[screen_pixel_index as usize] = pixel_value;
                // println!(
                //     "index to value {}:{}",
                //     screen_pixel_index as usize, pixel_value
                // );
                3
            }
            Opcode::Constant => {
                let buff = [
                    *cur_instructions.data.get(ip + 1).expect("expected byte"),
                    *cur_instructions.data.get(ip + 2).expect("expected byte"),
                ];

                let const_index = u16::from_be_bytes(buff);
                let new_ip = self.current_frame().ip + op.operand_width() as i64;
                self.set_ip(new_ip);
                let val = self.constants[const_index as usize];
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
                let val: u8 = self.globals[global_index as usize];
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
                self.globals[global_index as usize] = pop;
                2
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

    fn push(&mut self, obj: u8) -> Result<(), VMError> {
        if self.sp >= self.stack.capacity() {
            return Err(VMError::Reason("Stack overflow".to_string()));
        }

        self.stack.insert(self.sp, obj);
        self.sp += 1;
        Ok(())
    }

    fn push2(&mut self, obj: u16) -> Result<(), VMError> {
        if self.sp >= self.stack.capacity() {
            return Err(VMError::Reason("Stack overflow".to_string()));
        }

        let [high, low] = obj.to_be_bytes();

        self.stack.insert(self.sp, low);
        self.sp += 1;
        self.stack.insert(self.sp, high);
        self.sp += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> u8 {
        let val = self.stack.remove(self.sp - 1);
        self.sp -= 1;
        val
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
}
