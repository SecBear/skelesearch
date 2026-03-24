use clap::{Parser, Subcommand};

/// skelesearch — code skeleton search: index, search, context, status, clear, watch
#[derive(Parser)]
#[command(name = "skelesearch", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Increase log verbosity (-v info, -vv debug, -vvv trace).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Index a path into the local skelesearch store.
    Index {
        /// Directory or file to index.
        path: std::path::PathBuf,

        /// Embedding provider (default: fastembed).
        #[arg(long, default_value = "fastembed")]
        provider: String,
    },

    /// Search the index for a query.
    Search {
        /// Query string.
        query: String,

        /// Maximum number of results to return.
        #[arg(long, default_value_t = 5)]
        top_k: u32,

        /// Include one-hop import-graph neighbours in results.
        #[arg(long)]
        graph: bool,

        /// Output results as a JSON array.
        #[arg(long)]
        json: bool,

        /// Diversity factor for MMR re-ranking (0.0-1.0). Higher values reduce
        /// redundancy in results. 0 disables MMR.
        #[arg(long, default_value_t = 0.0)]
        diversity: f32,

        /// Embedding provider (must match the provider used during indexing).
        #[arg(long, default_value = "fastembed")]
        provider: String,

        /// Maximum token budget for results. Results are included greedily by
        /// score until the budget is exhausted. Approximate: 1 token ~ 4 chars.
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Scope results to files changed on the current branch (vs main/master).
        #[arg(long)]
        branch: bool,

        /// Print pipeline timing breakdown to stderr after results.
        #[arg(long)]
        timings: bool,
    },

    /// Show all indexed chunks plus import graph for a file.
    Context {
        /// File path (relative to project root).
        file: std::path::PathBuf,
    },

    /// Show index statistics.
    Status {
        /// Project root to query (defaults to current directory).
        path: Option<std::path::PathBuf>,

        /// Output as JSON (hook-facing contract).
        #[arg(long)]
        json: bool,
    },

    /// Remove the local index for a project.
    Clear {
        /// Project root to clear (defaults to current directory).
        path: Option<std::path::PathBuf>,
    },

    /// Watch a directory for changes and re-index automatically (2s debounce).
    Watch {
        /// Directory to watch.
        path: std::path::PathBuf,

        /// Embedding provider (default: fastembed).
        #[arg(long, default_value = "fastembed")]
        provider: String,
    },

    /// Search files for a regex or literal pattern.
    Grep {
        /// Regex pattern to search for.
        pattern: String,

        /// Directory to search (defaults to current directory).
        path: Option<std::path::PathBuf>,

        /// Maximum number of results.
        #[arg(long, default_value_t = 50)]
        max_results: usize,

        /// Case-insensitive matching.
        #[arg(short, long)]
        ignore_case: bool,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Remove index entries for deleted files.
    Gc {
        /// Project root (defaults to current directory).
        path: Option<std::path::PathBuf>,
    },

    /// Look up named symbol definitions in the index.
    Symbol {
        /// Symbol name to search for (exact match).
        name: String,

        /// Optional kind filter (function, struct, class, method, trait, enum, type).
        #[arg(long)]
        kind: Option<String>,
    },

    /// Evaluate search quality against a labeled eval set.
    Eval {
        /// Path to eval set JSON file.
        eval_set: std::path::PathBuf,

        /// Embedding provider (must match the provider used during indexing).
        #[arg(long, default_value = "fastembed")]
        provider: String,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,
    },
}