use skelesearch_embed_fastembed::FastEmbedProvider;
use skelesearch_core::EmbedProvider;

/// Verifies: one vector per input, correct dimensionality, input order preserved.
///
/// The jina-embeddings-v2-base-code model is ~90 MB and is downloaded on first
/// run. Subsequent runs use the local cache. Mark ignored so CI without network
/// access skips cleanly; run explicitly with `cargo test -- --ignored` when
/// network / cache is available.
#[tokio::test]
#[ignore = "requires model download / network or cache"]
async fn fastembed_provider_returns_one_vector_per_input_in_order() -> anyhow::Result<()> {
    // If model initialisation fails here (no network, no cache), let the error
    // propagate — a test run with --ignored must not silently pass.
    let provider = FastEmbedProvider::default()?;

    let ab = provider
        .embed_batch(vec!["fn alpha() {}".into(), "fn beta() {}".into()])
        .await?;
    let ba = provider
        .embed_batch(vec!["fn beta() {}".into(), "fn alpha() {}".into()])
        .await?;

    assert_eq!(ab.len(), 2, "should return one vector per input");
    assert_eq!(ba.len(), 2, "should return one vector per input");
    assert_eq!(
        ab[0].len(),
        provider.dim(),
        "vector dimensionality must match provider.dim()"
    );
    assert_eq!(
        ab[0], ba[1],
        "ab[0] (alpha) must equal ba[1] (alpha) — order preserved"
    );
    assert_eq!(
        ab[1], ba[0],
        "ab[1] (beta) must equal ba[0] (beta) — order preserved"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "requires model download / network or cache"]
async fn fastembed_provider_empty_batch_returns_empty() -> anyhow::Result<()> {
    let provider = FastEmbedProvider::default()?;
    let result = provider.embed_batch(vec![]).await?;
    assert!(result.is_empty(), "empty input must yield empty output");
    Ok(())
}
