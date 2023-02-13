use pyo3::{create_exception, exceptions};

create_exception!(
    libertem_dectris,
    TimeoutError,
    exceptions::PyException,
    "Timeout while communicating"
);

create_exception!(
    libertem_dectris,
    ConnectionError,
    exceptions::PyException,
    "SHM Connection failed"
);
