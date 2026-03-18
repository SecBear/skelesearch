// Sample fixture file for rename/delete reconciliation tests.

/// Helper used by lib.rs.
pub fn helper(x: i32) -> i32 {
    x * 2
}

/// Another exported function.
pub fn describe() -> &'static str {
    "old module"
}

pub struct OldStruct {
    pub value: i32,
}

impl OldStruct {
    pub fn new(value: i32) -> Self {
        Self { value }
    }

    pub fn doubled(&self) -> i32 {
        self.value * 2
    }
}
