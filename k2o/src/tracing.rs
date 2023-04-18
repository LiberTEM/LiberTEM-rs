use opentelemetry::global::BoxedTracer;
use opentelemetry::sdk::{trace as sdktrace, Resource};
use opentelemetry::trace::TraceError;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;

pub fn init_tracer() -> Result<(), TraceError> {
    opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint("http://localhost:4317"),
        )
        .with_trace_config(
            sdktrace::config().with_resource(Resource::new(vec![KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                "k2o",
            )])),
        )
        .install_batch(opentelemetry::runtime::Tokio)?;
    Ok(())
}

pub fn get_tracer() -> BoxedTracer {
    global::tracer("k2opy")
}
