use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::Object;

#[derive(PartialEq, Clone, Debug)]
pub struct Envir {
    store: HashMap<String, Object>,
    outer: Option<Rc<RefCell<Envir>>>,
}

impl Envir {
    pub fn new() -> Self {
        Envir {
            store: HashMap::new(),
            outer: None,
        }
    }

    pub fn from(store: HashMap<String, Object>) -> Self {
        Envir { store, outer: None }
    }

    pub fn new_with_outer(outer: Rc<RefCell<Envir>>) -> Self {
        Envir {
            store: HashMap::new(),
            outer: Some(outer),
        }
    }

    pub fn get(&mut self, name: String) -> Option<Object> {
        match self.store.get(&name) {
            Some(value) => Some(value.clone()),
            None => match self.outer {
                Some(ref outer) => outer.borrow_mut().get(name),
                None => None,
            },
        }
    }

    pub fn set(&mut self, name: String, value: &Object) {
        self.store.insert(name, value.clone());
    }
}
