use std::sync::{Arc, Barrier};
use std::time::Duration;

use opentelemetry::global::BoxedTracer;
use opentelemetry::trace::{
    self, SpanContext, SpanId, TraceContextExt, TraceError, TraceFlags, TraceId, TraceState, Tracer,
};
use opentelemetry::{global, Context, ContextGuard, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{runtime, trace as sdktrace, Resource};
use pyo3::types::{PyAnyMethods, PyModule};
use pyo3::{PyResult, Python};

fn init_tracer(service_name: String, otlp_endpoint: String) -> Result<(), TraceError> {
    opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(otlp_endpoint),
        )
        .with_trace_config(
            sdktrace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                service_name.to_owned(),
            )])),
        )
        .install_batch(runtime::Tokio)?;
    Ok(())
}

pub fn get_tracer() -> BoxedTracer {
    global::tracer(env!("CARGO_PKG_NAME"))
}

pub fn tracing_from_env(service_name: String) {
    // example OTEL config via environment variables:
    // OTEL_ENABLE=1
    // OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    if std::env::var("OTEL_ENABLE") == Ok("1".to_owned()) {
        let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:4317".to_owned());
        spawn_tracing_thread(service_name, endpoint);
    }
}

pub fn spawn_tracing_thread(service_name: String, otlp_endpoint: String) {
    let thread_builder = std::thread::Builder::new();

    // for waiting until tracing is initialized:
    let barrier = Arc::new(Barrier::new(2));
    let barrier_bg = Arc::clone(&barrier);

    thread_builder
        .name("tracing".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();

            rt.block_on(async {
                init_tracer(service_name, otlp_endpoint).unwrap();
                barrier_bg.wait();

                // do we need to keep this thread alive like this? I think so!
                // otherwise we get:
                // OpenTelemetry trace error occurred. cannot send span to the batch span processor because the channel is closed
                loop {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            });
        })
        .unwrap();

    barrier.wait();
}

pub fn get_py_span_context(py: Python) -> PyResult<SpanContext> {
    let span_context_py = PyModule::import(py, "opentelemetry.trace")?
        .getattr("get_current_span")?
        .call0()?
        .getattr("get_span_context")?
        .call0()?;

    let trace_id_py: u128 = span_context_py.getattr("trace_id")?.extract()?;
    let span_id_py: u64 = span_context_py.getattr("span_id")?.extract()?;
    let trace_flags_py: u8 = span_context_py.getattr("trace_flags")?.extract()?;

    let trace_id = TraceId::from_bytes(trace_id_py.to_be_bytes());
    let span_id = SpanId::from_bytes(span_id_py.to_be_bytes());
    let trace_flags = TraceFlags::new(trace_flags_py);

    // FIXME: Python has a list of something here, wtf is that & do we need it?
    let trace_state = TraceState::default();

    let span_context = SpanContext::new(trace_id, span_id, trace_flags, false, trace_state);

    Ok(span_context)
}

pub fn get_tracing_context(py: Python) -> PyResult<Context> {
    let span_context = get_py_span_context(py)?;
    let context = Context::default().with_remote_span_context(span_context);

    Ok(context)
}

pub fn span_from_py(py: Python, name: &str) -> PyResult<ContextGuard> {
    let tracer = get_tracer();
    let context = get_tracing_context(py)?;
    let span = tracer.start_with_context(name.to_string(), &context);
    Ok(trace::mark_span_as_active(span))
}
