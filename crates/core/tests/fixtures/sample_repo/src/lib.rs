// Sample fixture file for integration tests.
mod old;

use old::helper;

/// Add two integers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Double a value using the helper from old.rs.
pub fn double(x: i32) -> i32 {
    helper(x) + helper(x)
}

pub struct Registry {
    entries: Vec<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn register(&mut self, name: &str) {
        self.entries.push(name.to_string());
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}
