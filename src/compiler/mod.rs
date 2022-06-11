#[derive(Clone, Debug, PartialEq)]
pub struct Instructions {
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Vec<u8>,
}
