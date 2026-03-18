use skelesearch_core::router::{classify_query, QueryStrategy};

#[test]
fn identifier_routes_to_grep() {
    assert_eq!(classify_query("StorageBackend"), QueryStrategy::Grep);
    assert_eq!(classify_query("ERR_INVALID_HANDLE"), QueryStrategy::Grep);
    assert_eq!(classify_query("parse_config"), QueryStrategy::Grep);
}

#[test]
fn regex_routes_to_grep() {
    assert_eq!(classify_query(r"fn \w+_test"), QueryStrategy::Grep);
    assert_eq!(classify_query("[A-Z]+"), QueryStrategy::Grep);
}

#[test]
fn file_path_routes_to_grep() {
    assert_eq!(classify_query("src/main.rs"), QueryStrategy::Grep);
}

#[test]
fn natural_language_routes_to_semantic() {
    assert_eq!(classify_query("how does authentication work"), QueryStrategy::Semantic);
    assert_eq!(classify_query("where is the database connection"), QueryStrategy::Semantic);
}

#[test]
fn short_phrase_routes_to_grep() {
    assert_eq!(classify_query("main function"), QueryStrategy::Grep);
}
