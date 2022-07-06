use super::Object;

#[derive(Clone, Debug, PartialEq)]
pub struct BuiltInFunction {
    pub name: String,
    pub func: Object,
}

pub fn new_builtins() -> Vec<BuiltInFunction> {
    vec![
        BuiltInFunction {
            name: String::from("len"),
            func: Object::Builtin(1, b_len),
        },
        BuiltInFunction {
            name: String::from("first"),
            func: Object::Builtin(1, b_first),
        },
        BuiltInFunction {
            name: String::from("last"),
            func: Object::Builtin(1, b_last),
        },
        BuiltInFunction {
            name: String::from("rest"),
            func: Object::Builtin(1, b_rest),
        },
        BuiltInFunction {
            name: String::from("push"),
            func: Object::Builtin(2, b_push),
        },
        BuiltInFunction {
            name: String::from("puts"),
            func: Object::Builtin(-1, |args| {
                for arg in args {
                    println!("{}", arg);
                }
                Object::Null
            }),
        },
    ]
}

fn b_len(args: Vec<Object>) -> Object {
    match &args[0] {
        //Object::String(s) => Object::Number(s.len() as f64),
        Object::Array(o) => Object::Number(o.len() as f64),
        o => Object::Error(format!("argument to `len` not supported, got {}", o)),
    }
}

fn b_first(args: Vec<Object>) -> Object {
    match &args[0] {
        Object::Array(o) => {
            if let Some(ao) = o.first() {
                ao.clone()
            } else {
                Object::Null
            }
        }
        o => Object::Error(format!("argument to `first` must be array, got {}", o)),
    }
}

fn b_last(args: Vec<Object>) -> Object {
    match &args[0] {
        Object::Array(o) => {
            if let Some(ao) = o.last() {
                ao.clone()
            } else {
                Object::Null
            }
        }
        o => Object::Error(format!("argument to `last` must be array, got {}", o)),
    }
}

fn b_rest(args: Vec<Object>) -> Object {
    match &args[0] {
        Object::Array(o) => {
            if !o.is_empty() {
                Object::Array(o[1..].to_vec())
            } else {
                Object::Null
            }
        }
        o => Object::Error(format!("argument to `rest` must be array, got {}", o)),
    }
}

fn b_push(args: Vec<Object>) -> Object {
    match &args[0] {
        Object::Array(o) => {
            let mut arr = o.clone();
            arr.push(args[1].clone());
            Object::Array(arr)
        }
        o => Object::Error(format!("argument to `push` must be array, got {}", o)),
    }
}
