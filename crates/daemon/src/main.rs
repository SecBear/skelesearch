use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use skelesearch_daemon::{
    DaemonPaths, DaemonService, DaemonSingleton, DaemonState, SingletonError,
};
use skelesearch_telemetry::TracingOptions;

#[derive(Debug, Parser)]
#[command(name = "skelesearchd", version)]
struct Args {
    /// Override daemon Unix socket path.
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,

    /// Enable managed helper mode with idle auto-shutdown.
    #[arg(long, default_value_t = false)]
    managed: bool,

    /// Idle grace period before a managed daemon exits when no clients remain.
    #[arg(long, value_name = "SECONDS", default_value_t = 300)]
    idle_timeout_seconds: u64,
}

#[tokio::main]
async fn main() {
    if let Err(err) = async_main().await {
        eprintln!("skelesearchd: fatal: {err:#}");
        std::process::exit(1);
    }
}

async fn async_main() -> anyhow::Result<()> {
    let args = Args::parse();
    let paths = DaemonPaths::resolve(args.socket)?;

    let telemetry_options = TracingOptions::new("skelesearch-daemon", "info")
        .with_file_log_path(paths.log_path.clone())
        .with_json_file(false);
    let _telemetry = skelesearch_telemetry::init_tracing_with_options(telemetry_options);

    let singleton_guard = match DaemonSingleton::acquire(&paths) {
        Ok(guard) => guard,
        Err(err @ SingletonError::AlreadyRunning { .. })
        | Err(err @ SingletonError::StartupInProgress { .. }) => {
            eprintln!("{}", err.user_message());
            tracing::info!(message = %err.user_message(), "daemon startup skipped");
            return Ok(());
        }
        Err(SingletonError::Other(err)) => return Err(err).context("acquire daemon singleton"),
    };

    let endpoint = skelesearch_daemon::transport::ListenerEndpoint::unix(paths.socket_path.clone());
    let state = DaemonState::new();
    let service = DaemonService::new(state.clone());
    let listener = skelesearch_daemon::transport::bind(&endpoint, service)
        .await
        .with_context(|| format!("bind daemon listener at {}", paths.socket_path.display()))?;

    tracing::info!(
        pid = singleton_guard.pid(),
        socket_path = %paths.socket_path.display(),
        lock_path = %singleton_guard.lock_path().display(),
        version = env!("CARGO_PKG_VERSION"),
        managed = args.managed,
        idle_timeout_seconds = args.idle_timeout_seconds,
        "skelesearch daemon started"
    );

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut serve_task = tokio::spawn(async move { listener.run(shutdown_rx).await });

    let idle_timeout = std::time::Duration::from_secs(args.idle_timeout_seconds.max(1));
    let shutdown_tx_for_reaper = shutdown_tx.clone();
    let mut reaper_task = if args.managed {
        Some(tokio::spawn(async move {
            let mut idle_since: Option<std::time::Instant> = None;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                let expired = state.reap_expired_sessions().await;
                if expired > 0 {
                    tracing::info!(
                        expired_sessions = expired,
                        "expired stale daemon client sessions"
                    );
                }

                let has_sessions = state.live_session_count().await > 0;
                let indexing_running = state.any_indexing_running().await;
                if has_sessions || indexing_running {
                    idle_since = None;
                    continue;
                }

                match idle_since {
                    Some(since) if since.elapsed() >= idle_timeout => {
                        tracing::info!(
                            idle_timeout_seconds = args.idle_timeout_seconds,
                            "managed daemon idle timeout reached; shutting down"
                        );
                        let _ = shutdown_tx_for_reaper.send(true);
                        break;
                    }
                    Some(_) => {}
                    None => idle_since = Some(std::time::Instant::now()),
                }
            }
        }))
    } else {
        None
    };

    tokio::select! {
        result = &mut serve_task => {
            result.context("daemon listener task join")??;
            tracing::warn!("daemon listener exited");
        }
        signal = tokio::signal::ctrl_c() => {
            signal.context("wait for ctrl-c")?;
            tracing::info!("received shutdown signal");
            let _ = shutdown_tx.send(true);
            serve_task.await.context("daemon listener shutdown join")??;
        }
    }

    if let Some(task) = reaper_task.take() {
        task.abort();
    }

    tracing::info!(pid = singleton_guard.pid(), "skelesearch daemon stopping");
    Ok(())
}
