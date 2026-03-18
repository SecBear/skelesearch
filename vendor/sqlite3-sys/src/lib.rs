//! Shim for `sqlite3-sys` that exposes the same FFI surface as the published
//! crate but does NOT declare `links = "sqlite3"`.
//!
//! `cozo` depends on `sqlite` → `sqlite3-sys` (this shim).  We want to add
//! `rusqlite` to the workspace, which pulls in `libsqlite3-sys` (bundled).
//! Two crates declaring `links = "sqlite3"` conflicts in Cargo.  This shim
//! removes the second declaration; `libsqlite3-sys` owns the native library
//! and provides the sqlite3.a that satisfies these extern "C" declarations at
//! final link time.

#![allow(improper_ctypes, non_camel_case_types)]
#![no_std]

#[rustfmt::skip]
mod constants;
mod functions;
mod types;

pub use constants::*;
pub use functions::*;
pub use types::*;
