use skelesearch_core::Config;

#[test]
fn parse_minimal_config() {
    let config = Config::from_str("").unwrap();
    assert_eq!(config.index.batch_size, 64);
    assert_eq!(config.index.provider, "fastembed");
    assert_eq!(config.search.default_top_k, 5);
}

#[test]
fn parse_full_config() {
    let config = Config::from_str(
        r#"
[index]
provider = "fastembed"
batch_size = 128
exclude = ["vendor/"]

[search]
default_top_k = 10
"#,
    )
    .unwrap();
    assert_eq!(config.index.batch_size, 128);
    assert_eq!(config.search.default_top_k, 10);
    assert_eq!(config.index.exclude, vec!["vendor/"]);
}

#[test]
fn config_returns_defaults_when_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::load(dir.path()).unwrap();
    assert_eq!(config.index.batch_size, 64);
    assert_eq!(config.search.default_top_k, 5);
}

#[test]
fn invalid_toml_returns_error() {
    let result = Config::from_str("not valid toml [[[");
    assert!(result.is_err(), "expected parse error for invalid TOML");
}

#[test]
fn unknown_keys_are_ignored() {
    // serde(default) means unrecognised keys must not cause a parse error
    let config = Config::from_str(
        r#"
[index]
unknown_future_key = "ignored"
batch_size = 32
"#,
    )
    .unwrap();
    assert_eq!(config.index.batch_size, 32);
}
