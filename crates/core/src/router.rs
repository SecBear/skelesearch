// router.rs — classify a search query as grep-appropriate or semantic-appropriate.
//
// Classification is heuristic: it looks for surface signals (regex metacharacters,
// camelCase/snake_case identifiers, file extensions) to decide when an exact-match
// search beats a vector search, and vice versa.

/// Strategy for handling a search query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStrategy {
    Grep,
    Semantic,
}

impl std::fmt::Display for QueryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grep => write!(f, "grep"),
            Self::Semantic => write!(f, "semantic"),
        }
    }
}

/// Classify a query as grep-appropriate or semantic-appropriate.
///
/// Rules applied in order (first match wins):
/// 1. Empty / whitespace-only → Semantic (returns no results anyway)
/// 2. Regex metacharacters (`\`, `[`, `^`, `$`, `+`, `|`) → Grep
/// 3. File-path pattern (`/` separator or known extension) → Grep
/// 4. Single token that looks like an identifier (camelCase, snake_case, SCREAMING) → Grep
/// 5. 3+ words containing a natural-language stop word → Semantic
/// 6. 3+ words without stop words → Semantic
/// 7. 1-2 plain words → Grep
pub fn classify_query(query: &str) -> QueryStrategy {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryStrategy::Semantic;
    }

    // Regex special chars indicate the caller wants pattern matching.
    if trimmed.contains('\\')
        || trimmed.contains('[')
        || trimmed.contains('^')
        || trimmed.contains('$')
        || trimmed.contains('+')
        || trimmed.contains('|')
    {
        return QueryStrategy::Grep;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();

    // Long query with an embedded file path → route Semantic: the natural-language
    // context matters for embedding and the path provides BM25 anchor signal.
    // Must check before the short-query file-path→Grep rule below.
    if words.len() >= 3 && trimmed.contains('/') && trimmed.contains('.') {
        return QueryStrategy::Semantic;
    }

    // File path pattern: slash-separated or common code extensions.
    if trimmed.contains('/')
        || trimmed.ends_with(".rs")
        || trimmed.ends_with(".py")
        || trimmed.ends_with(".ts")
        || trimmed.ends_with(".js")
        || trimmed.ends_with(".go")
        || trimmed.ends_with(".java")
    {
        return QueryStrategy::Grep;
    }

    // Single-token identifier: camelCase, snake_case, or SCREAMING_CASE.
    if words.len() == 1 {
        let w = words[0];
        // snake_case
        if w.contains('_') {
            return QueryStrategy::Grep;
        }
        // camelCase or PascalCase: has both upper and lower chars
        if w.chars().any(|c| c.is_uppercase()) && w.chars().any(|c| c.is_lowercase()) {
            return QueryStrategy::Grep;
        }
        // SCREAMING: all uppercase letters (or underscores, already handled above)
        if w.len() > 2 && w.chars().all(|c| c.is_uppercase() || c == '_') {
            return QueryStrategy::Grep;
        }
    }

    // Two-word identifier queries (e.g. "ConnectionPool retry", "AuthMiddleware validate") —
    // conceptual keyword searches best served by embedding, not exact match.
    if words.len() == 2 {
        // Route Semantic when at least one word is a named identifier (camelCase or
        // snake_case). Two plain lowercase words ("hello world") stay as Grep.
        let has_identifier = words.iter().any(|w| {
            w.contains('_')
                || (w.chars().any(|c| c.is_uppercase()) && w.chars().any(|c| c.is_lowercase()))
        });
        if has_identifier {
            return QueryStrategy::Semantic;
        }
    }

    // Natural-language indicators: 3+ words where at least one is a stop word.
    let nl = [
        "how", "what", "where", "why", "when", "does", "is", "are", "the", "this", "that",
    ];
    if words.len() >= 3 && words.iter().any(|w| nl.contains(&w.to_lowercase().as_str())) {
        return QueryStrategy::Semantic;
    }

    // 3+ words without stop words are still treated as natural-language queries.
    if words.len() >= 3 {
        return QueryStrategy::Semantic;
    }

    // 1-2 plain words → treat as identifier/keyword → exact match.
    QueryStrategy::Grep
}


#[cfg(test)]
mod tests {
    use super::{classify_query, QueryStrategy};

    #[test]
    fn two_identifier_words_route_semantic() {
        // Both words carry identifier markers — conceptual keyword search, not grep.
        assert_eq!(classify_query("ConnectionPool retry"), QueryStrategy::Semantic);
        assert_eq!(classify_query("AuthMiddleware validate"), QueryStrategy::Semantic);
    }

    #[test]
    fn two_plain_words_route_grep() {
        // Two plain lowercase words with no identifier markers → exact match.
        assert_eq!(classify_query("hello world"), QueryStrategy::Grep);
    }

    #[test]
    fn long_query_with_filepath_routes_semantic() {
        // Natural-language query embedding a file path → Semantic.
        assert_eq!(
            classify_query("error in src/auth.rs when handling tokens"),
            QueryStrategy::Semantic,
        );
    }

    #[test]
    fn single_camel_case_still_routes_grep() {
        // Existing invariant: single identifier token → Grep.
        assert_eq!(classify_query("EmbedProvider"), QueryStrategy::Grep);
    }

    #[test]
    fn single_filepath_routes_grep() {
        // A bare file path (no surrounding prose) → Grep.
        assert_eq!(classify_query("src/auth.rs"), QueryStrategy::Grep);
    }
}