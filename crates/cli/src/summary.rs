// Re-export the OpenAI summary provider from core so CLI code can use
// `crate::summary::OpenAISummaryProvider` without importing from the core crate directly.
pub use skelesearch_core::OpenAISummaryProvider;
