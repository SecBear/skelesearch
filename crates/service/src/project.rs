use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Component, Path, PathBuf};

/// Canonical project identifier used by the daemon protocol.
///
/// `canonical_root` is a normalized absolute path string.
/// `logical_id` is optional and reserved for hosted/multi-tenant routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectKey {
    pub canonical_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_id: Option<String>,
}

impl ProjectKey {
    /// Build a key from an absolute project root path.
    ///
    /// Best effort canonicalization:
    /// 1) `std::fs::canonicalize` when possible.
    /// 2) Lexical normalization for existing absolute paths that cannot be canonicalized.
    ///
    /// Relative paths are rejected to avoid implicit current-working-directory assumptions.
    pub fn from_root_path(path: impl AsRef<Path>) -> Result<Self, ProjectKeyError> {
        Self::from_root_path_with_logical_id(path, None)
    }

    /// Same as [`Self::from_root_path`], but allows assigning a logical ID.
    pub fn from_root_path_with_logical_id(
        path: impl AsRef<Path>,
        logical_id: Option<String>,
    ) -> Result<Self, ProjectKeyError> {
        let canonical = canonicalize_best_effort(path.as_ref())?;
        let logical_id = normalize_logical_id(logical_id)?;
        Ok(Self {
            canonical_root: path_to_wire_string(&canonical),
            logical_id,
        })
    }

    /// Build a key from a potentially-relative path with an explicit base directory.
    ///
    /// This is useful for clients that want to route relative to a known workspace
    /// root without relying on daemon cwd.
    pub fn from_root_path_with_base(
        path: impl AsRef<Path>,
        base: impl AsRef<Path>,
        logical_id: Option<String>,
    ) -> Result<Self, ProjectKeyError> {
        let path = path.as_ref();
        let base = canonicalize_best_effort(base.as_ref())?;
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            base.join(path)
        };
        let canonical = canonicalize_best_effort(&absolute)?;
        let logical_id = normalize_logical_id(logical_id)?;
        Ok(Self {
            canonical_root: path_to_wire_string(&canonical),
            logical_id,
        })
    }

    /// Stable normalized string form suitable for logs/protocol tracing.
    pub fn stable_string(&self) -> String {
        match &self.logical_id {
            Some(logical_id) => format!("logical_id={logical_id};root={}", self.canonical_root),
            None => format!("root={}", self.canonical_root),
        }
    }
}

impl fmt::Display for ProjectKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.stable_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProjectKeyError {
    #[error("project path must be absolute or resolved with an explicit base: {0}")]
    RelativePathWithoutBase(String),
    #[error("logical_id cannot be empty")]
    EmptyLogicalId,
}

fn canonicalize_best_effort(path: &Path) -> Result<PathBuf, ProjectKeyError> {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return Ok(canonical);
    }

    if path.is_absolute() {
        return Ok(normalize_absolute_path(path));
    }

    Err(ProjectKeyError::RelativePathWithoutBase(
        path.to_string_lossy().into_owned(),
    ))
}

fn normalize_logical_id(logical_id: Option<String>) -> Result<Option<String>, ProjectKeyError> {
    match logical_id {
        None => Ok(None),
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Err(ProjectKeyError::EmptyLogicalId);
            }
            Ok(Some(trimmed.to_string()))
        }
    }
}

fn normalize_absolute_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(seg) => normalized.push(seg),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }
    normalized
}

fn path_to_wire_string(path: &Path) -> String {
    #[cfg(windows)]
    {
        let mut value = path.to_string_lossy().replace('\\', "/");
        if value.len() >= 2 && value.as_bytes()[1] == b':' {
            let lower = value[0..1].to_ascii_lowercase();
            value.replace_range(0..1, &lower);
        }
        value
    }

    #[cfg(not(windows))]
    {
        path.to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skelesearch-service-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn project_key_canonicalizes_existing_path() {
        let root = unique_temp_path("canonical");
        std::fs::create_dir_all(&root).expect("create temp dir");

        let noisy = root.join(".");
        let key = ProjectKey::from_root_path(noisy).expect("build key");
        let expected = path_to_wire_string(&std::fs::canonicalize(&root).expect("canonicalize"));

        assert_eq!(key.canonical_root, expected);
        assert_eq!(key.logical_id, None);
    }

    #[test]
    fn project_key_normalizes_with_explicit_base_without_cwd_assumptions() {
        let base = unique_temp_path("base");
        std::fs::create_dir_all(&base).expect("create base dir");

        let key = ProjectKey::from_root_path_with_base(
            Path::new("repo/../repo"),
            &base,
            Some("tenant/project-a".to_string()),
        )
        .expect("build key with base");

        let canonical_base = std::fs::canonicalize(&base).expect("canonicalize base");
        let expected = path_to_wire_string(&canonical_base.join("repo"));
        assert_eq!(key.canonical_root, expected);
        assert_eq!(key.logical_id.as_deref(), Some("tenant/project-a"));
        assert_eq!(
            key.stable_string(),
            format!("logical_id=tenant/project-a;root={expected}")
        );
    }

    #[test]
    fn relative_path_without_base_is_rejected() {
        let err =
            ProjectKey::from_root_path("relative/path").expect_err("must reject relative path");
        assert!(matches!(err, ProjectKeyError::RelativePathWithoutBase(_)));
    }
}
