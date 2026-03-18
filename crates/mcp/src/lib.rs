// skelesearch-mcp library target.
//
// Exporting modules here allows integration tests in tests/ to use the server
// and tool types directly without going through the binary.
pub mod server;
pub mod tools;
