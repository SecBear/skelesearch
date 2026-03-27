use std::path::Path;

use chrono::{DateTime, Utc};

use crate::ManifestStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessState {
    Fresh,
    Stale,
    Refreshing,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct FreshnessSnapshot {
    pub state: FreshnessState,
    pub estimated_stale: usize,
    pub freshness_checked_at: Option<DateTime<Utc>>,
    pub freshness_error: Option<String>,
}

impl FreshnessSnapshot {
    pub fn from_manifest(manifest: &ManifestStore, project_root: &Path) -> Self {
        Self::from_stale_count_result(manifest.count_stale(project_root))
    }

    pub fn from_stale_count_result(result: anyhow::Result<usize>) -> Self {
        match result {
            Ok(estimated_stale) => Self {
                state: if estimated_stale == 0 {
                    FreshnessState::Fresh
                } else {
                    FreshnessState::Stale
                },
                estimated_stale,
                freshness_checked_at: Some(Utc::now()),
                freshness_error: None,
            },
            Err(err) => Self {
                state: FreshnessState::Unknown,
                estimated_stale: 0,
                freshness_checked_at: None,
                freshness_error: Some(err.to_string()),
            },
        }
    }

    pub fn with_refreshing(mut self, refreshing: bool) -> Self {
        if refreshing {
            self.state = FreshnessState::Refreshing;
        }
        self
    }
}
