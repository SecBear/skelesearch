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

    let words: Vec<&str> = trimmed.split_whitespace().collect();

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
