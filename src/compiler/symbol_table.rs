// use crate::builtins::new_builtins;
use std::cell::RefCell;
use std::collections::hash_map::{Iter, IterMut};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
    // BuiltIn,
    Free,
    // Function,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub scope: SymbolScope,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolTable {
    store: HashMap<String, Symbol>,
    pub num_definitions: usize,
    pub outer: Option<Rc<RefCell<SymbolTable>>>,
    pub free_symbols: Vec<Symbol>,
}

// We're going to use this to determine whether or not we should
// be allowed to access the last element of SymbolTable.store
#[derive(Clone)]
pub struct StoreNotEmpty(PhantomData<SymbolTable>);
impl StoreNotEmpty {
    pub fn new(tbl: &SymbolTable) -> Option<Self> {
        if tbl.store_len() > 0 {
            Some(StoreNotEmpty(PhantomData))
        } else {
            None
        }
    }
}

impl From<Rc<SymbolTable>> for SymbolTable {
    fn from(s: Rc<SymbolTable>) -> Self {
        SymbolTable {
            store: s.store.clone(),
            num_definitions: s.num_definitions,
            outer: s.outer.clone(),
            free_symbols: s.free_symbols.clone(),
        }
    }
}

impl SymbolTable {
    // pub fn new_with_builtins() -> Self {
    //     let mut result = Self {
    //         store: HashMap::new(),
    //         num_definitions: 0,
    //         outer: None,
    //         free_symbols: vec![],
    //     };
    //     for (ind, b) in new_builtins().iter().enumerate() {
    //         result.define_builtin(ind, b.name.clone());
    //     }

    //     result
    // }
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            num_definitions: 0,
            outer: None,
            free_symbols: vec![],
        }
    }

    // pub fn new_with_outer(outer: Rc<RefCell<SymbolTable>>) -> Self {
    //     Self {
    //         store: HashMap::new(),
    //         num_definitions: 0,
    //         outer: Some(outer),
    //         free_symbols: vec![],
    //     }
    // }

    pub fn last_symbol(&self, _proof: StoreNotEmpty) -> (usize, String) {
        let symbol = self.store.values().last().unwrap().clone();
        (symbol.index, symbol.name)
    }

    pub fn store_len(&self) -> usize {
        self.store.len()
    }

    pub fn store_iter<'a>(&'a self) -> Iter<'a, String, Symbol> {
        self.store.iter()
    }

    pub fn store_iter_mut<'a>(&'a mut self) -> IterMut<'a, String, Symbol> {
        self.store.iter_mut()
    }

    pub fn get_from_store(&self, name: &str) -> Option<&Symbol> {
        self.store.get(name)
    }

    // pub fn define_builtin(&mut self, index: usize, name: String) -> Symbol {
    //     let result = Symbol {
    //         name: String::from(name),
    //         index,
    //         scope: SymbolScope::BuiltIn,
    //     };
    //     self.store.insert(result.name.clone(), result.clone());
    //     result
    // }

    // pub fn define_function(&mut self, name: String) -> Symbol {
    //     let result = Symbol {
    //         name: String::from(name),
    //         index: 0,
    //         scope: SymbolScope::Function,
    //     };
    //     self.store.insert(result.name.clone(), result.clone());
    //     result
    // }

    pub fn define_free(&mut self, original: Symbol) -> Symbol {
        self.free_symbols.push(original.clone());
        let result = Symbol {
            name: original.name,
            index: self.free_symbols.len() - 1,
            scope: SymbolScope::Free,
        };
        self.store.insert(result.name.clone(), result.clone());
        result
    }

    pub fn define(&mut self, name: &str) -> Symbol {
        let scope = match self.outer {
            None => SymbolScope::Global,
            Some(_) => SymbolScope::Local,
        };

        let result = Symbol {
            name: String::from(name),
            index: self.num_definitions,
            scope,
        };
        self.store.insert(result.name.clone(), result.clone());
        self.num_definitions += 1;
        result
    }

    pub fn resolve(&mut self, name: String) -> Option<Symbol> {
        let res = match self.store.get(&name) {
            Some(value) => return Some(value.clone()),
            None => match self.outer {
                Some(ref outer) => outer.borrow_mut().resolve(name),
                None => None,
            },
        };

        match res {
            Some(a) => match a.scope {
                // SymbolScope::BuiltIn | SymbolScope::Global => Some(a),
                SymbolScope::Global => Some(a),
                _ => Some(self.define_free(a)),
            },
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    use crate::compiler::symbol_table::*;

    #[test]
    fn test_define() {
        let mut expected: HashMap<String, Symbol> = HashMap::new();
        expected.insert(
            "a".to_string(),
            Symbol {
                index: 0,
                name: String::from("a"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "b".to_string(),
            Symbol {
                index: 1,
                name: String::from("b"),
                scope: SymbolScope::Global,
            },
        );

        let mut global = SymbolTable::new();
        let a = global.define("a");
        assert_eq!(a, expected["a"]);
        let b = global.define("b");
        assert_eq!(b, expected["b"]);
    }

    #[test]
    fn test_resolve() {
        let mut expected: HashMap<String, Symbol> = HashMap::new();
        expected.insert(
            "a".to_string(),
            Symbol {
                index: 0,
                name: String::from("a"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "b".to_string(),
            Symbol {
                index: 1,
                name: String::from("b"),
                scope: SymbolScope::Global,
            },
        );

        let mut global = SymbolTable::new();
        global.define("a");
        global.define("b");

        for (k, v) in expected.iter() {
            assert_eq!(
                (global
                    .resolve(k.to_string())
                    .expect("expected to get a value")),
                *v
            );
        }
    }

    // #[test]
    // fn test_resolve_function() {
    //     let mut expected: HashMap<String, Symbol> = HashMap::new();
    //     expected.insert(
    //         "a".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("a"),
    //             scope: SymbolScope::Function,
    //         },
    //     );

    //     let mut global = SymbolTable::new();
    //     global.define_function("a".to_string());

    //     for (k, v) in expected.iter() {
    //         assert_eq!(
    //             (global
    //                 .resolve(k.to_string())
    //                 .expect("expected to get a value")),
    //             *v
    //         );
    //     }
    // }

    // #[test]
    // fn test_local_define() {
    //     let mut expected: HashMap<String, Symbol> = HashMap::new();
    //     expected.insert(
    //         "a".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("a"),
    //             scope: SymbolScope::Global,
    //         },
    //     );
    //     expected.insert(
    //         "b".to_string(),
    //         Symbol {
    //             index: 1,
    //             name: String::from("b"),
    //             scope: SymbolScope::Global,
    //         },
    //     );
    //     expected.insert(
    //         "c".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("c"),
    //             scope: SymbolScope::Local,
    //         },
    //     );
    //     expected.insert(
    //         "d".to_string(),
    //         Symbol {
    //             index: 1,
    //             name: String::from("d"),
    //             scope: SymbolScope::Local,
    //         },
    //     );

    //     let mut global = SymbolTable::new();
    //     let a = global.define("a");
    //     assert_eq!(a, expected["a"]);
    //     let b = global.define("b");
    //     assert_eq!(b, expected["b"]);

    //     let mut local = SymbolTable::new_with_outer(Rc::new(RefCell::new(global)));

    //     // let a2 = local.define("a");
    //     // assert_eq!(a2, expected["a"]);
    //     // let b2 = local.define("b");
    //     // assert_eq!(b2, expected["b"]);

    //     let c = local.define("c");
    //     assert_eq!(c, expected["c"]);
    //     local.define("d");
    //     // assert_eq!(d, expected["d"]);
    //     assert_eq!(local.resolve(String::from("a")).unwrap(), expected["a"]);
    //     assert_eq!(local.resolve(String::from("d")).unwrap(), expected["d"]);
    // }

    // #[test]
    // fn test_local_free() {
    //     let mut expected: Vec<Symbol> = vec![];
    //     let mut expected_outer: Vec<Symbol> = vec![];
    //     let mut freescope: Vec<Symbol> = vec![];
    //     freescope.push(Symbol {
    //         index: 0,
    //         name: String::from("c"),
    //         scope: SymbolScope::Local,
    //     });
    //     freescope.push(Symbol {
    //         index: 1,
    //         name: String::from("d"),
    //         scope: SymbolScope::Local,
    //     });

    //     expected.push(Symbol {
    //         index: 0,
    //         name: String::from("a"),
    //         scope: SymbolScope::Global,
    //     });
    //     expected.push(Symbol {
    //         index: 1,
    //         name: String::from("b"),
    //         scope: SymbolScope::Global,
    //     });
    //     expected.push(Symbol {
    //         index: 0,
    //         name: String::from("c"),
    //         scope: SymbolScope::Local,
    //     });
    //     expected.push(Symbol {
    //         index: 1,
    //         name: String::from("d"),
    //         scope: SymbolScope::Local,
    //     });

    //     expected_outer.push(Symbol {
    //         index: 0,
    //         name: String::from("a"),
    //         scope: SymbolScope::Global,
    //     });
    //     expected_outer.push(Symbol {
    //         index: 1,
    //         name: String::from("b"),
    //         scope: SymbolScope::Global,
    //     });
    //     expected_outer.push(Symbol {
    //         index: 0,
    //         name: String::from("c"),
    //         scope: SymbolScope::Free,
    //     });
    //     expected_outer.push(Symbol {
    //         index: 1,
    //         name: String::from("d"),
    //         scope: SymbolScope::Free,
    //     });

    //     expected_outer.push(Symbol {
    //         index: 0,
    //         name: String::from("e"),
    //         scope: SymbolScope::Local,
    //     });
    //     expected_outer.push(Symbol {
    //         index: 1,
    //         name: String::from("f"),
    //         scope: SymbolScope::Local,
    //     });

    //     let mut global = SymbolTable::new();
    //     global.define("a");
    //     global.define("b");

    //     let mut local = SymbolTable::new_with_outer(Rc::new(RefCell::new(global)));
    //     local.define("c");
    //     local.define("d");
    //     for v in expected.iter() {
    //         assert_eq!(local.resolve(String::from(v.name.clone())).unwrap(), *v);
    //     }

    //     let mut local2: SymbolTable = SymbolTable::new_with_outer(Rc::new(RefCell::new(local)));
    //     local2.define("e");
    //     local2.define("f");
    //     for v in expected_outer.iter() {
    //         assert_eq!(local2.resolve(String::from(v.name.clone())).unwrap(), *v);
    //     }
    //     for (i, v) in freescope.iter().enumerate() {
    //         assert_eq!(local2.free_symbols.get(i).unwrap().clone(), *v);
    //     }
    // }

    // #[test]
    // fn test_sublocal_define() {
    //     let mut expected: HashMap<String, Symbol> = HashMap::new();
    //     expected.insert(
    //         "a".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("a"),
    //             scope: SymbolScope::Global,
    //         },
    //     );
    //     expected.insert(
    //         "b".to_string(),
    //         Symbol {
    //             index: 1,
    //             name: String::from("b"),
    //             scope: SymbolScope::Global,
    //         },
    //     );
    //     expected.insert(
    //         "c".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("c"),
    //             scope: SymbolScope::Free,
    //         },
    //     );
    //     expected.insert(
    //         "d".to_string(),
    //         Symbol {
    //             index: 1,
    //             name: String::from("d"),
    //             scope: SymbolScope::Free,
    //         },
    //     );
    //     expected.insert(
    //         "e".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("e"),
    //             scope: SymbolScope::Local,
    //         },
    //     );
    //     expected.insert(
    //         "f".to_string(),
    //         Symbol {
    //             index: 1,
    //             name: String::from("f"),
    //             scope: SymbolScope::Local,
    //         },
    //     );

    //     let mut global = SymbolTable::new();
    //     global.define("a");
    //     global.define("b");
    //     // assert_eq!(a, expected["a"]);
    //     // assert_eq!(b, expected["b"]);

    //     let mut local = SymbolTable::new_with_outer(Rc::new(RefCell::new(global)));
    //     local.define("c");
    //     local.define("d");

    //     let mut local2 = SymbolTable::new_with_outer(Rc::new(RefCell::new(local)));
    //     local2.define("e");
    //     local2.define("f");
    //     // let a2 = local.define("a");
    //     // assert_eq!(a2, expected["a"]);
    //     // let b2 = local.define("b");
    //     // assert_eq!(b2, expected["b"]);

    //     // assert_eq!(c, expected["c"]);
    //     // assert_eq!(d, expected["d"]);
    //     assert_eq!(local2.resolve(String::from("a")).unwrap(), expected["a"]);
    //     assert_eq!(local2.resolve(String::from("b")).unwrap(), expected["b"]);
    //     assert_eq!(local2.resolve(String::from("c")).unwrap(), expected["c"]);
    //     assert_eq!(local2.resolve(String::from("d")).unwrap(), expected["d"]);
    //     assert_eq!(local2.resolve(String::from("e")).unwrap(), expected["e"]);
    //     assert_eq!(local2.resolve(String::from("f")).unwrap(), expected["f"]);
    // }

    // #[test]
    // fn test_builtins() {
    //     let mut expected: HashMap<String, Symbol> = HashMap::new();
    //     expected.insert(
    //         "a".to_string(),
    //         Symbol {
    //             index: 0,
    //             name: String::from("a"),
    //             scope: SymbolScope::BuiltIn,
    //         },
    //     );
    //     // expected.insert(
    //     //     "b".to_string(),
    //     //     Symbol {
    //     //         index: 1,
    //     //         name: String::from("b"),
    //     //         scope: SymbolScope::BuiltIn,
    //     //     },
    //     // );
    //     expected.insert(
    //         "c".to_string(),
    //         Symbol {
    //             index: 1,
    //             name: String::from("c"),
    //             scope: SymbolScope::BuiltIn,
    //         },
    //     );
    //     // expected.insert(
    //     //     "d".to_string(),
    //     //     Symbol {
    //     //         index: 1,
    //     //         name: String::from("d"),
    //     //         scope: SymbolScope::BuiltIn,
    //     //     },
    //     // );
    //     expected.insert(
    //         "e".to_string(),
    //         Symbol {
    //             index: 2,
    //             name: String::from("e"),
    //             scope: SymbolScope::BuiltIn,
    //         },
    //     );
    //     expected.insert(
    //         "f".to_string(),
    //         Symbol {
    //             index: 3,
    //             name: String::from("f"),
    //             scope: SymbolScope::BuiltIn,
    //         },
    //     );

    //     let global = Rc::new(RefCell::new(SymbolTable::new()));
    //     // global.define("a");
    //     // global.define("b");
    //     // assert_eq!(a, expected["a"]);
    //     // assert_eq!(b, expected["b"]);
    //     for (_ind, i) in expected.clone().iter().enumerate() {
    //         global.borrow_mut().define_builtin(i.1.index, i.0.clone());
    //     }

    //     let local = Rc::new(RefCell::new(SymbolTable::new_with_outer(global.clone())));
    //     // local.define("c");
    //     // local.define("d");

    //     let local2 = Rc::new(RefCell::new(SymbolTable::new_with_outer(local.clone())));

    //     // local2.define("e");
    //     // local2.define("f");
    //     // let a2 = local.define("a");
    //     // assert_eq!(a2, expected["a"]);
    //     // let b2 = local.define("b");
    //     // assert_eq!(b2, expected["b"]);

    //     // assert_eq!(c, expected["c"]);
    //     // assert_eq!(d, expected["d"]);
    //     for i in expected {
    //         assert_eq!(
    //             global.borrow_mut().resolve(i.0.clone()).unwrap(),
    //             i.1.clone()
    //         );
    //         assert_eq!(
    //             local.borrow_mut().resolve(i.0.clone()).unwrap(),
    //             i.1.clone()
    //         );
    //         assert_eq!(
    //             local2.borrow_mut().resolve(i.0.clone()).unwrap(),
    //             i.1.clone()
    //         );
    //     }
    //     // assert_eq!(local2.resolve(String::from("a")).unwrap(), expected["a"]);
    //     // assert_eq!(local2.resolve(String::from("b")).unwrap(), expected["b"]);
    //     // assert_eq!(local2.resolve(String::from("c")).unwrap(), expected["c"]);
    //     // assert_eq!(local2.resolve(String::from("d")).unwrap(), expected["d"]);
    //     // assert_eq!(local2.resolve(String::from("e")).unwrap(), expected["e"]);
    //     // assert_eq!(local2.resolve(String::from("f")).unwrap(), expected["f"]);
    // }
}
