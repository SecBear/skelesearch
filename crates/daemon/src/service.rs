use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use anyhow::Context as _;
use skelesearch_service::{ProjectKey, ProjectKeyError, ProjectTarget};
use tokio::sync::{Mutex, RwLock};

const STORAGE_DIR_NAME: &str = ".skelesearch";
const BACKEND_DB_FILE: &str = "index.db";
const MANIFEST_DB_FILE: &str = "manifest.db";
const INDEX_LOCK_FILE: &str = ".skelesearch.lock";
const INDEX_STATUS_FILE: &str = "indexing-status.json";

#[derive(Debug, Clone, Default)]
pub struct DaemonState {
    projects: Arc<RwLock<HashMap<ProjectKey, Arc<ProjectState>>>>,
}

impl DaemonState {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn resolve_or_open_project(
        &self,
        target: ProjectLookup,
    ) -> anyhow::Result<Arc<ProjectState>> {
        let project_key = canonical_project_key(target)?;

        {
            let projects = self.projects.read().await;
            if let Some(project) = projects.get(&project_key) {
                project.touch_last_access().await;
                return Ok(Arc::clone(project));
            }
        }

        let mut projects = self.projects.write().await;
        if let Some(project) = projects.get(&project_key) {
            project.touch_last_access().await;
            return Ok(Arc::clone(project));
        }

        let project = Arc::new(ProjectState::open(project_key.clone())?);
        projects.insert(project_key, Arc::clone(&project));
        Ok(project)
    }

    pub async fn project_count(&self) -> usize {
        self.projects.read().await.len()
    }
}

#[derive(Debug, Clone)]
pub enum ProjectLookup {
    ProjectKey(ProjectKey),
    RootPath {
        root_path: PathBuf,
        logical_id: Option<String>,
    },
}

impl ProjectLookup {
    pub fn from_root_path(root_path: impl Into<PathBuf>) -> Self {
        Self::RootPath {
            root_path: root_path.into(),
            logical_id: None,
        }
    }
}

impl From<ProjectTarget> for ProjectLookup {
    fn from(target: ProjectTarget) -> Self {
        match target {
            ProjectTarget::ProjectKey { project_key } => Self::ProjectKey(project_key),
            ProjectTarget::RootPath {
                root_path,
                logical_id,
            } => Self::RootPath {
                root_path: PathBuf::from(root_path),
                logical_id,
            },
        }
    }
}

#[derive(Debug)]
pub struct ProjectState {
    pub project_key: ProjectKey,
    pub canonical_root: PathBuf,
    pub storage_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub backend: Arc<BackendHandle>,
    pub cached_searcher: Arc<RwLock<Option<Arc<CachedSearcherPlaceholder>>>>,
    pub provider_identity: Arc<RwLock<Option<ProviderIdentityPlaceholder>>>,
    pub index_progress: Arc<RwLock<Option<IndexProgressPlaceholder>>>,
    pub coordination: CoordinationStatePlaceholder,
    manifest: Arc<ManifestHandle>,
    last_access: Arc<Mutex<SystemTime>>,
}

impl ProjectState {
    fn open(project_key: ProjectKey) -> anyhow::Result<Self> {
        let canonical_root = PathBuf::from(&project_key.canonical_root);
        let storage_dir = storage_dir_for_root(&canonical_root);
        std::fs::create_dir_all(&storage_dir).with_context(|| {
            format!(
                "create daemon project storage directory '{}'",
                storage_dir.display()
            )
        })?;

        let manifest_path = storage_dir.join(MANIFEST_DB_FILE);
        let backend = Arc::new(BackendHandle::open(storage_dir.join(BACKEND_DB_FILE))?);
        let manifest = Arc::new(ManifestHandle::open(manifest_path.clone())?);

        Ok(Self {
            project_key,
            canonical_root,
            storage_dir: storage_dir.clone(),
            manifest_path,
            backend,
            cached_searcher: Arc::new(RwLock::new(None)),
            provider_identity: Arc::new(RwLock::new(None)),
            index_progress: Arc::new(RwLock::new(None)),
            coordination: CoordinationStatePlaceholder::for_storage_dir(&storage_dir),
            manifest,
            last_access: Arc::new(Mutex::new(SystemTime::now())),
        })
    }

    pub async fn touch_last_access(&self) {
        let mut guard = self.last_access.lock().await;
        *guard = SystemTime::now();
    }

    pub async fn last_accessed_at(&self) -> SystemTime {
        *self.last_access.lock().await
    }

    pub fn manifest_handle(&self) -> Arc<ManifestHandle> {
        Arc::clone(&self.manifest)
    }
}

#[derive(Debug)]
pub struct BackendHandle {
    pub index_db_path: PathBuf,
    _index_db_file: File,
}

impl BackendHandle {
    fn open(index_db_path: PathBuf) -> anyhow::Result<Self> {
        let file = open_or_create_rw(&index_db_path)
            .with_context(|| format!("open backend file '{}'", index_db_path.display()))?;
        Ok(Self {
            index_db_path,
            _index_db_file: file,
        })
    }
}

#[derive(Debug)]
pub struct ManifestHandle {
    pub manifest_db_path: PathBuf,
    _manifest_db_file: File,
}

impl ManifestHandle {
    fn open(manifest_db_path: PathBuf) -> anyhow::Result<Self> {
        let file = open_or_create_rw(&manifest_db_path)
            .with_context(|| format!("open manifest file '{}'", manifest_db_path.display()))?;
        Ok(Self {
            manifest_db_path,
            _manifest_db_file: file,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderIdentityPlaceholder {
    pub provider_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexProgressPlaceholder {
    pub phase: String,
    pub files_done: usize,
    pub files_total: usize,
}

#[derive(Debug)]
pub struct CachedSearcherPlaceholder;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinationStatePlaceholder {
    pub lock_path: PathBuf,
    pub status_path: PathBuf,
}

impl CoordinationStatePlaceholder {
    fn for_storage_dir(storage_dir: &Path) -> Self {
        Self {
            lock_path: storage_dir.join(INDEX_LOCK_FILE),
            status_path: storage_dir.join(INDEX_STATUS_FILE),
        }
    }
}

fn canonical_project_key(target: ProjectLookup) -> Result<ProjectKey, ProjectKeyError> {
    match target {
        ProjectLookup::ProjectKey(project_key) => ProjectKey::from_root_path_with_logical_id(
            PathBuf::from(project_key.canonical_root),
            project_key.logical_id,
        ),
        ProjectLookup::RootPath {
            root_path,
            logical_id,
        } => ProjectKey::from_root_path_with_logical_id(root_path, logical_id),
    }
}

fn storage_dir_for_root(root: &Path) -> PathBuf {
    root.join(STORAGE_DIR_NAME)
}

fn open_or_create_rw(path: &Path) -> anyhow::Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open project state file '{}'", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn repeated_lookup_reuses_same_registry_entry() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let first = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("first project lookup");
        let second = state
            .resolve_or_open_project(ProjectLookup::from_root_path(root.join(".")))
            .await
            .expect("second project lookup");

        assert_eq!(first.project_key, second.project_key);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(state.project_count().await, 1);
    }

    #[tokio::test]
    async fn distinct_roots_create_distinct_registry_entries() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root_a = temp.path().join("repo-a");
        let root_b = temp.path().join("repo-b");
        std::fs::create_dir_all(&root_a).expect("create root_a");
        std::fs::create_dir_all(&root_b).expect("create root_b");

        let state = DaemonState::new();
        let project_a = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root_a))
            .await
            .expect("resolve project_a");
        let project_b = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root_b))
            .await
            .expect("resolve project_b");

        assert_ne!(project_a.project_key, project_b.project_key);
        assert_eq!(state.project_count().await, 2);
    }
}
