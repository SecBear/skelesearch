//! Tracing and telemetry setup for skelesearch binaries.
//!
//! Provides:
//! - stderr logging with env-filter for interactive use
//! - optional rotating file logging for server deployments
//! - optional OTLP export when the `otlp` feature is enabled

use std::path::PathBuf;

use anyhow::Context as _;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Debug, Clone)]
pub struct TracingOptions {
    pub service_name: String,
    pub default_level: String,
    pub file_log_path: Option<PathBuf>,
    pub json_file: bool,
}

impl TracingOptions {
    pub fn new(service_name: impl Into<String>, default_level: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            default_level: default_level.into(),
            file_log_path: None,
            json_file: true,
        }
    }

    pub fn with_file_log_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.file_log_path = Some(path.into());
        self
    }

    pub fn with_json_file(mut self, json_file: bool) -> Self {
        self.json_file = json_file;
        self
    }
}

pub struct TelemetryGuard {
    _stderr_guard: tracing_appender::non_blocking::WorkerGuard,
    _file_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    #[cfg(feature = "otlp")]
    _provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl TelemetryGuard {
    fn new(
        stderr_guard: tracing_appender::non_blocking::WorkerGuard,
        file_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
        #[cfg(feature = "otlp")] provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
    ) -> Self {
        Self {
            _stderr_guard: stderr_guard,
            _file_guard: file_guard,
            #[cfg(feature = "otlp")]
            _provider: provider,
        }
    }
}

pub fn init_tracing(service_name: &str, default_level: &str) -> TelemetryGuard {
    init_tracing_with_options(TracingOptions::new(service_name, default_level))
}

pub fn init_tracing_with_options(options: TracingOptions) -> TelemetryGuard {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(options.default_level.clone()));

    let (stderr_writer, stderr_guard) = tracing_appender::non_blocking(std::io::stderr());
    let stderr_layer = fmt::layer()
        .with_writer(stderr_writer)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_target(true)
        .with_ansi(false);

    let file_sink = options
        .file_log_path
        .as_ref()
        .map(build_file_writer)
        .transpose();

    let (file_writer, file_guard) = match file_sink {
        Ok(Some(v)) => (Some(v.0), Some(v.1)),
        Ok(None) => (None, None),
        Err(err) => {
            eprintln!(
                "skelesearch: failed to initialize file logging ({err:#}), continuing with stderr only"
            );
            (None, None)
        }
    };

    #[cfg(feature = "otlp")]
    {
        let (otel_layer, provider) = if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok() {
            match init_otlp(&options.service_name) {
                Ok((layer, prov)) => (Some(layer), Some(prov)),
                Err(err) => {
                    eprintln!("skelesearch: OTLP init failed ({err:#}), falling back to stderr/file only");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        match (otel_layer, file_writer, options.json_file) {
            (Some(otel_layer), Some(writer), true) => {
                let file_layer = fmt::layer()
                    .json()
                    .with_writer(writer)
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_current_span(false)
                    .with_span_list(false)
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(otel_layer)
                    .with(stderr_layer)
                    .with(file_layer)
                    .with(env_filter)
                    .init();
                tracing::info!(service = %options.service_name, "OpenTelemetry OTLP exporter enabled");
                return TelemetryGuard::new(stderr_guard, file_guard, provider);
            }
            (Some(otel_layer), Some(writer), false) => {
                let file_layer = fmt::layer()
                    .with_writer(writer)
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(otel_layer)
                    .with(stderr_layer)
                    .with(file_layer)
                    .with(env_filter)
                    .init();
                tracing::info!(service = %options.service_name, "OpenTelemetry OTLP exporter enabled");
                return TelemetryGuard::new(stderr_guard, file_guard, provider);
            }
            (Some(otel_layer), None, _) => {
                tracing_subscriber::registry()
                    .with(otel_layer)
                    .with(stderr_layer)
                    .with(env_filter)
                    .init();
                tracing::info!(service = %options.service_name, "OpenTelemetry OTLP exporter enabled");
                return TelemetryGuard::new(stderr_guard, file_guard, provider);
            }
            (None, Some(writer), true) => {
                let file_layer = fmt::layer()
                    .json()
                    .with_writer(writer)
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_current_span(false)
                    .with_span_list(false)
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(stderr_layer)
                    .with(file_layer)
                    .with(env_filter)
                    .init();
                return TelemetryGuard::new(stderr_guard, file_guard, provider);
            }
            (None, Some(writer), false) => {
                let file_layer = fmt::layer()
                    .with_writer(writer)
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(stderr_layer)
                    .with(file_layer)
                    .with(env_filter)
                    .init();
                return TelemetryGuard::new(stderr_guard, file_guard, provider);
            }
            (None, None, _) => {
                tracing_subscriber::registry()
                    .with(stderr_layer)
                    .with(env_filter)
                    .init();
                return TelemetryGuard::new(stderr_guard, file_guard, provider);
            }
        }
    }

    #[cfg(not(feature = "otlp"))]
    {
        match (file_writer, options.json_file) {
            (Some(writer), true) => {
                let file_layer = fmt::layer()
                    .json()
                    .with_writer(writer)
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_current_span(false)
                    .with_span_list(false)
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(stderr_layer)
                    .with(file_layer)
                    .with(env_filter)
                    .init();
            }
            (Some(writer), false) => {
                let file_layer = fmt::layer()
                    .with_writer(writer)
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(stderr_layer)
                    .with(file_layer)
                    .with(env_filter)
                    .init();
            }
            (None, _) => {
                tracing_subscriber::registry()
                    .with(stderr_layer)
                    .with(env_filter)
                    .init();
            }
        }
        TelemetryGuard::new(stderr_guard, file_guard)
    }
}

fn build_file_writer(
    path: &PathBuf,
) -> anyhow::Result<(
    tracing_appender::non_blocking::NonBlocking,
    tracing_appender::non_blocking::WorkerGuard,
)> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("file log path '{}' has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create telemetry log directory '{}'", parent.display()))?;
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid telemetry log filename '{}'", path.display()))?;
    let appender = tracing_appender::rolling::daily(parent, filename);
    Ok(tracing_appender::non_blocking(appender))
}

#[cfg(feature = "otlp")]
fn init_otlp(
    service_name: &str,
) -> anyhow::Result<(
    tracing_opentelemetry::OpenTelemetryLayer<
        tracing_subscriber::Registry,
        opentelemetry_sdk::trace::SdkTracer,
    >,
    opentelemetry_sdk::trace::SdkTracerProvider,
)> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;
    use opentelemetry_sdk::{
        trace::{Sampler, SdkTracerProvider},
        Resource,
    };

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()?;

    let resource = Resource::builder()
        .with_attributes([
            KeyValue::new("service.name", service_name.to_string()),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION").to_string()),
        ])
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_sampler(Sampler::AlwaysOn)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer(service_name.to_string());
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);

    Ok((layer, provider))
}
