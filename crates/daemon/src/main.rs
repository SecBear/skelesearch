use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use skelesearch_daemon::{DaemonPaths, DaemonService, DaemonSingleton, DaemonState, SingletonError};
use skelesearch_telemetry::TracingOptions;

#[derive(Debug, Parser)]
#[command(name = "skelesearchd", version)]
struct Args {
    /// Override daemon Unix socket path.
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,
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
    let service = DaemonService::new(DaemonState::new());
    let listener = skelesearch_daemon::transport::bind(&endpoint, service)
        .await
        .with_context(|| format!("bind daemon listener at {}", paths.socket_path.display()))?;

    tracing::info!(
        pid = singleton_guard.pid(),
        socket_path = %paths.socket_path.display(),
        lock_path = %singleton_guard.lock_path().display(),
        version = env!("CARGO_PKG_VERSION"),
        "skelesearch daemon started"
    );

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut serve_task = tokio::spawn(async move { listener.run(shutdown_rx).await });

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

    tracing::info!(pid = singleton_guard.pid(), "skelesearch daemon stopping");
    Ok(())
}
