use crate::compiler::{Instructions, Object};

#[derive(Debug, Clone)]
pub struct Frame {
    pub instr: Instructions,
    pub num_locals: i32,
    pub ip: i64,
    pub base_pointer: i64,
    pub free: Vec<Object>,
    pub num_args: i64,
}

impl Frame {
    pub fn new(
        instr: Instructions,
        num_locals: i32,
        base_pointer: i64,
        free: Vec<Object>,
        num_args: i64,
    ) -> Self {
        Self {
            instr,
            num_locals,
            ip: -1,
            base_pointer,
            free,
            num_args,
        }
    }

    pub fn instructions(&self) -> Instructions {
        self.instr.clone()
    }
}
