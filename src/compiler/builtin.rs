use super::Object;

#[derive(Clone, Debug, PartialEq)]
pub struct BuiltInFunction {
    pub name: String,
    pub func: Box<Object>,
}
