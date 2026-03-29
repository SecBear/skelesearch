use std::sync::{Mutex, OnceLock};

use skelesearch_core::preferred_index_provider_name;

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn preferred_index_provider_prefers_voyage_when_present() {
    let _guard = env_lock().lock().expect("env lock");
    std::env::remove_var("OPENAI_API_KEY");
    std::env::set_var("VOYAGE_API_KEY", "test-key");

    assert_eq!(preferred_index_provider_name(), "voyage");

    std::env::remove_var("VOYAGE_API_KEY");
}

#[test]
fn preferred_index_provider_falls_back_to_openai() {
    let _guard = env_lock().lock().expect("env lock");
    std::env::remove_var("VOYAGE_API_KEY");
    std::env::set_var("OPENAI_API_KEY", "test-key");

    assert_eq!(preferred_index_provider_name(), "openai");

    std::env::remove_var("OPENAI_API_KEY");
}

#[test]
fn preferred_index_provider_falls_back_to_fastembed_without_api_keys() {
    let _guard = env_lock().lock().expect("env lock");
    std::env::remove_var("VOYAGE_API_KEY");
    std::env::remove_var("OPENAI_API_KEY");

    assert_eq!(preferred_index_provider_name(), "fastembed");
}
