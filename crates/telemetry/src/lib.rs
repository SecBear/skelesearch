//! Tracing and telemetry setup for skelesearch binaries.
//!
//! Provides [`init_tracing`] which configures:
//! - **Always:** `tracing-subscriber` fmt layer to stderr with env-filter
//! - **When `otlp` feature + `OTEL_EXPORTER_OTLP_ENDPOINT` env var:** OpenTelemetry
//!   OTLP exporter layer that sends spans to any OTLP-compatible collector
//!   (Grafana, Datadog, Jaeger, Axiom, etc).
//!
//! # Usage
//!
//! ```rust,ignore
//! let _guard = skelesearch_telemetry::init_tracing("skelesearch-mcp", "info");
//! ```
//!
//! # Environment variables
//!
//! | Variable | Effect |
//! |---|---|
//! | `RUST_LOG` | Controls log level filter (standard env-filter syntax) |
//! | `OTEL_EXPORTER_OTLP_ENDPOINT` | Enables OTLP export (e.g. `http://localhost:4317`) |
//! | `OTEL_SERVICE_NAME` | Overrides service name (default: binary name) |

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Guard that shuts down the telemetry pipeline on drop.
pub struct TelemetryGuard {
    #[cfg(feature = "otlp")]
    _provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

/// Initialize tracing for a skelesearch binary.
pub fn init_tracing(service_name: &str, default_level: &str) -> TelemetryGuard {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    let fmt_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_target(true);

    #[cfg(feature = "otlp")]
    {
        // OpenTelemetryLayer<Registry, _> only implements Layer<Registry>.
        // It must be the first layer added — directly on Registry — before
        // fmt or env_filter wrap the subscriber type.
        let (otel_layer, provider) = if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok() {
            match init_otlp(service_name) {
                Ok((layer, prov)) => (Some(layer), Some(prov)),
                Err(e) => {
                    eprintln!("skelesearch: OTLP init failed ({e:#}), falling back to stderr");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // Layer order: Registry → otel (Option, directly on Registry) → fmt → env_filter
        tracing_subscriber::registry()
            .with(otel_layer)
            .with(fmt_layer)
            .with(env_filter)
            .init();

        if provider.is_some() {
            tracing::info!(service = service_name, "OpenTelemetry OTLP exporter enabled");
        }

        return TelemetryGuard { _provider: provider };
    }

    #[cfg(not(feature = "otlp"))]
    {
        let _ = service_name;
        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(env_filter)
            .init();
        TelemetryGuard {}
    }
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
        trace::{SdkTracerProvider, Sampler},
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
