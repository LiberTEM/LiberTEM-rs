//! This module should be a wrapper around the generic functionality of the `common`
//! crate, and only contain logic that is specific to the detector. The interface
//! exported to Python should be as uniform as possible compared to other detectors,
//! pending future unification with full compatability between detectors.
use std::time::Duration;

use common::generic_connection::{ConnectionStatus, GenericConnection};

use crate::{
    background_thread::{DectrisBackgroundThread, DectrisDetectorConnConfig, DectrisExtraControl},
    base_types::{
        DConfig, DHeader, DImage, DImageD, DSeriesEnd, DectrisFrameMeta, DectrisPendingAcquisition,
        DetectorConfig, PixelType, TriggerMode,
    },
    exceptions::{DecompressError, TimeoutError},
    sim::DectrisSim,
};

use common::{impl_py_cam_client, impl_py_connection};

use pyo3::{
    exceptions::PyDeprecationWarning,
    prelude::*,
    types::{PyBytes, PyType},
};

use log::trace;
use numpy::PyUntypedArray;

use crate::decoder::DectrisDecoder;

#[pymodule]
fn libertem_dectris(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<DectrisFrameStack>()?;
    m.add_class::<DectrisConnection>()?;
    m.add_class::<PixelType>()?;
    m.add_class::<DectrisSim>()?;
    m.add_class::<DetectorConfig>()?;
    m.add_class::<TriggerMode>()?;
    m.add_class::<CamClient>()?;
    m.add("TimeoutError", py.get_type_bound::<TimeoutError>())?;
    m.add("DecompressError", py.get_type_bound::<DecompressError>())?;

    register_header_module(py, &m)?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_DECTRIS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_DECTRIS_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    Ok(())
}

fn register_header_module(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let headers_module = PyModule::new_bound(py, "headers")?;
    headers_module.add_class::<DHeader>()?;
    headers_module.add_class::<DImage>()?;
    headers_module.add_class::<DImageD>()?;
    headers_module.add_class::<DConfig>()?;
    headers_module.add_class::<DSeriesEnd>()?;
    parent_module.add_submodule(&headers_module)?;
    Ok(())
}

impl_py_connection!(
    _PyDectrisConnection,
    _PyDectrisFrameStack,
    DectrisFrameMeta,
    DectrisBackgroundThread,
    DectrisPendingAcquisition,
    libertem_dectris
);

#[pyclass]
struct DectrisConnection {
    conn: _PyDectrisConnection,
}

#[pymethods]
impl DectrisConnection {
    #[new]
    fn new(
        uri: &str,
        frame_stack_size: usize,
        handle_path: &str,
        num_slots: Option<usize>,
        bytes_per_frame: Option<usize>,
        huge: Option<bool>,
    ) -> PyResult<Self> {
        let num_slots = num_slots.map_or_else(|| 2000, |x| x);
        let bytes_per_frame = bytes_per_frame.map_or_else(|| 512 * 512 * 2, |x| x);
        let config = DectrisDetectorConnConfig::new(
            uri,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            huge.map_or_else(|| false, |x| x),
            handle_path,
        );

        let shm = GenericConnection::<DectrisBackgroundThread, DectrisPendingAcquisition>::shm_from_config(&config).map_err(|e| {
            PyConnectionError::new_err(format!("could not init shm: {}", e))
        })?;
        let bg_thread = DectrisBackgroundThread::spawn(&config, &shm)
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
        let generic_conn =
            GenericConnection::<DectrisBackgroundThread, DectrisPendingAcquisition>::new(
                bg_thread, &shm,
            )
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let conn = _PyDectrisConnection::new(shm, generic_conn);

        Ok(Self { conn })
    }

    /// Wait until the detector is armed, or until the timeout expires (in seconds)
    /// Returns `None` in case of timeout, the detector config otherwise.
    /// This method drops the GIL to allow concurrent Python threads.
    fn wait_for_arm(
        &mut self,
        timeout: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<Option<(DetectorConfig, u64)>> {
        let res = self.conn.wait_for_arm(timeout, py)?;
        Ok(res.map(|config| (config.get_detector_config(), config.get_series())))
    }

    fn get_socket_path(&self) -> PyResult<String> {
        self.conn.get_socket_path()
    }

    fn is_running(&self) -> PyResult<bool> {
        self.conn.is_running()
    }

    fn start(&mut self, series: u64, timeout: Option<f32>) -> PyResult<()> {
        let timeout = timeout.map_or(Duration::from_millis(100), Duration::from_secs_f32);

        self.conn
            .send_specialized(DectrisExtraControl::StartAcquisitionWithSeries { series })?;
        self.conn
            .wait_for_status(ConnectionStatus::Armed, Some(timeout))?;
        Ok(())
    }

    /// Start listening for global acquisition headers on the zeromq socket.
    fn start_passive(&mut self, timeout: Option<f32>, py: Python<'_>) -> PyResult<()> {
        self.conn.start_passive(timeout, py)
    }

    fn close(&mut self) -> PyResult<()> {
        self.conn.close()
    }

    fn log_shm_stats(&self) -> PyResult<()> {
        self.conn.log_shm_stats()
    }

    fn get_next_stack(
        &mut self,
        max_size: usize,
        py: Python<'_>,
    ) -> PyResult<Option<DectrisFrameStack>> {
        let stack_inner = self.conn.get_next_stack(max_size, py)?;
        Ok(stack_inner.map(DectrisFrameStack::new))
    }
}

#[pyclass(name = "FrameStackHandle")]
pub struct DectrisFrameStack {
    inner: _PyDectrisFrameStack,
}

impl DectrisFrameStack {
    pub fn new(inner: _PyDectrisFrameStack) -> Self {
        Self { inner }
    }

    pub fn get_inner(&self) -> &_PyDectrisFrameStack {
        &self.inner
    }

    pub fn get_inner_mut(&mut self) -> &mut _PyDectrisFrameStack {
        &mut self.inner
    }
}

#[pymethods]
impl DectrisFrameStack {
    fn __len__(&self) -> PyResult<usize> {
        self.inner.__len__()
    }

    fn get_dtype_string(&self) -> PyResult<String> {
        self.inner.get_dtype_string()
    }

    /// use `get_dtype_string` instead
    #[deprecated]
    fn get_pixel_type(&self, py: Python<'_>) -> PyResult<String> {
        let meta = self.inner.try_get_inner()?.first_meta();
        PyErr::warn_bound(
            py,
            &py.get_type_bound::<PyDeprecationWarning>(),
            "FrameStackHandle.get_pixel_type is deprecated, use get_dtype_string instead.",
            0,
        )?;

        Ok(match meta.dimaged.type_ {
            PixelType::Uint8 => "uint8",
            PixelType::Uint16 => "uint16",
            PixelType::Uint32 => "uint32",
        }
        .to_owned())
    }

    /// use `get_dtype_string` instead, that includes endianess
    #[deprecated]
    fn get_endianess(&self, py: Python<'_>) -> PyResult<String> {
        PyErr::warn_bound(
            py,
            &py.get_type_bound::<PyDeprecationWarning>(),
            "FrameStackHandle.get_endianess is deprecated, use get_dtype_string instead.",
            0,
        )?;
        Ok(self
            .inner
            .try_get_inner()?
            .first_meta()
            .get_endianess()
            .as_string())
    }

    /// implementation detail that Python shouldn't care about
    #[deprecated]
    fn get_encoding(&self, py: Python<'_>) -> PyResult<String> {
        PyErr::warn_bound(
            py,
            &py.get_type_bound::<PyDeprecationWarning>(),
            "FrameStackHandle.get_encoding is deprecated and will be removed in the future.",
            0,
        )?;
        Ok(self
            .inner
            .try_get_inner()?
            .first_meta()
            .dimaged
            .encoding
            .to_string())
    }

    fn get_shape(&self) -> PyResult<(u64, u64)> {
        self.inner.get_shape()
    }

    fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.inner.serialize(py)
    }

    #[classmethod]
    fn deserialize<'py>(
        _cls: Bound<'py, PyType>,
        serialized: Bound<'py, PyBytes>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: _PyDectrisFrameStack::deserialize_impl(serialized)?,
        })
    }
}

impl_py_cam_client!(
    _PyDectrisCamClient,
    DectrisDecoder,
    _PyDectrisFrameStack,
    DectrisFrameMeta,
    libertem_dectris
);

#[pyclass]
pub struct CamClient {
    inner: _PyDectrisCamClient,
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(handle_path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: _PyDectrisCamClient::new(handle_path)?,
        })
    }

    fn decode_range_into_buffer<'py>(
        &self,
        input: &DectrisFrameStack,
        out: &Bound<'py, PyUntypedArray>,
        start_idx: usize,
        end_idx: usize,
        py: Python<'py>,
    ) -> PyResult<()> {
        self.inner
            .decode_range_into_buffer(input.get_inner(), out, start_idx, end_idx, py)
    }

    #[deprecated]
    fn decompress_frame_stack<'py>(
        &self,
        handle: &DectrisFrameStack,
        out: &Bound<'py, PyUntypedArray>,
        py: Python<'py>,
    ) -> PyResult<()> {
        PyErr::warn_bound(
            py,
            &py.get_type_bound::<PyDeprecationWarning>(),
            "CamClient.decompress_frame_stack is deprecated, use decode_range_into_buffer instead.",
            0,
        )?;
        self.inner.decode_into_buffer(handle.get_inner(), out, py)
    }

    fn done(&mut self, handle: &mut DectrisFrameStack) -> PyResult<()> {
        self.inner.frame_stack_done(handle.get_inner_mut())
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close()
    }
}

impl Drop for CamClient {
    fn drop(&mut self) {
        trace!("CamClient::drop");
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, io::Write, path::PathBuf};

    use common::frame_stack::{FrameStackForWriting, FrameStackHandle};
    use lz4::block::CompressionMode;
    use numpy::{PyArray, PyArrayMethods};
    use tempfile::tempdir;

    use ipc_test::SharedSlabAllocator;
    use pyo3::{prepare_freethreaded_python, Python};
    use zerocopy::AsBytes;

    use crate::{
        base_types::DectrisFrameMeta,
        dectris_py::{CamClient, DectrisFrameStack, _PyDectrisFrameStack},
    };
    use tempfile::TempDir;

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    #[test]
    fn test_cam_client() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, 512);
        let dimage = crate::base_types::DImage {
            htype: "dimage-1.0".to_string().try_into().unwrap(),
            series: 1,
            frame: 1,
            hash: "aaaabbbb".to_string().try_into().unwrap(),
        };
        let dimaged = crate::base_types::DImageD {
            htype: "d-image_d-1.0".to_string().try_into().unwrap(),
            shape: (16, 16),
            type_: crate::base_types::PixelType::Uint16,
            encoding: "bs16-lz4<".to_string().try_into().unwrap(),
        };
        let dconfig = crate::base_types::DConfig {
            htype: "dconfig-1.0".to_string().try_into().unwrap(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };

        // some predictable test data:
        let in_: Vec<u16> = (0..256).map(|i| i % 16).collect();
        let compressed_data = bs_sys::compress_lz4(&in_, None).unwrap();

        // compressed dectris data stream has an (unknown)
        // header in front of the compressed data, which we just cut off,
        // so here we just prepend 12 zero-bytes
        let mut data_with_prefix = vec![0; 12];
        data_with_prefix.extend_from_slice(&compressed_data);
        assert!(data_with_prefix.len() < 512);
        data_with_prefix.iter().take(12).for_each(|&e| {
            assert_eq!(e, 0);
        });
        println!("compressed_data:        {:x?}", &compressed_data);
        println!("data_with_prefix[12..]: {:x?}", &data_with_prefix[12..]);
        assert_eq!(fs.get_cursor(), 0);

        let meta = DectrisFrameMeta {
            dimage,
            dimaged,
            dconfig,
            data_length_bytes: data_with_prefix.len(),
        };

        fs.write_frame(&meta, |mut b| -> Result<(), Infallible> {
            b.write_all(&data_with_prefix).unwrap();
            Ok(())
        })
        .unwrap();

        assert_eq!(fs.get_cursor(), data_with_prefix.len());

        // we have one frame in there:
        assert_eq!(fs.len(), 1);

        let fs_handle = fs.writing_done(&mut shm).unwrap();

        // we still have one frame in there:
        assert_eq!(fs_handle.len(), 1);

        // initialize a Python interpreter so we are able to construct a PyBytes instance:
        prepare_freethreaded_python();

        // roundtrip serialize/deserialize:
        Python::with_gil(|_py| {
            let bytes = fs_handle.serialize().unwrap();
            let new_handle = FrameStackHandle::deserialize_impl(&bytes).unwrap();
            assert_eq!(fs_handle, new_handle);
        });

        let client = CamClient::new(socket_as_path.to_str().unwrap()).unwrap();

        fs_handle.with_slot(&shm, |slot_r| {
            let slice = slot_r.as_slice();
            println!("slice: {:x?}", slice);
        });

        Python::with_gil(|py| {
            let flat: Vec<u16> = (0..256).collect();
            let out = PyArray::from_vec_bound(py, flat)
                .reshape((1, 16, 16))
                .unwrap();

            let out_untyped = out.as_untyped();
            let dfsh = DectrisFrameStack::new(_PyDectrisFrameStack::new(fs_handle));
            client
                .decode_range_into_buffer(&dfsh, out_untyped, 0, dfsh.__len__().unwrap(), py)
                .unwrap();

            out.readonly()
                .as_slice()
                .unwrap()
                .iter()
                .zip(0..)
                .for_each(|(&item, idx)| {
                    assert_eq!(item, in_[idx]);
                    assert_eq!(item, (idx % 16) as u16);
                });
        });
    }

    #[test]
    fn test_cam_client_lz4() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, 512);
        let dimage = crate::base_types::DImage {
            htype: "dimage-1.0".to_string().try_into().unwrap(),
            series: 1,
            frame: 1,
            hash: "aaaabbbb".to_string().try_into().unwrap(),
        };
        let dimaged = crate::base_types::DImageD {
            htype: "dimage_d-1.0".to_string().try_into().unwrap(),
            shape: (16, 16),
            type_: crate::base_types::PixelType::Uint16,
            encoding: "lz4<".to_string().try_into().unwrap(),
        };
        let dconfig = crate::base_types::DConfig {
            htype: "dconfig-1.0".to_string().try_into().unwrap(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };

        // some predictable test data:
        let in_: Vec<u16> = (0..256).map(|i| i % 16).collect();
        let in_bytes = in_.as_bytes();
        let compressed_data =
            lz4::block::compress(in_bytes, Some(CompressionMode::DEFAULT), false).unwrap();

        println!("compressed_data: {:x?}", &compressed_data);
        assert_eq!(fs.get_cursor(), 0);

        let meta = DectrisFrameMeta {
            dimage,
            dimaged,
            dconfig,
            data_length_bytes: compressed_data.len(),
        };

        fs.write_frame(&meta, |buf| {
            buf.copy_from_slice(&compressed_data);
            Ok::<_, Infallible>(())
        })
        .unwrap();

        assert_eq!(fs.get_cursor(), compressed_data.len());

        // we have one frame in there:
        assert_eq!(fs.len(), 1);

        let fs_handle = fs.writing_done(&mut shm).unwrap();

        // we still have one frame in there:
        assert_eq!(fs_handle.len(), 1);

        // initialize a Python interpreter so we are able to construct a PyBytes instance:
        prepare_freethreaded_python();

        // roundtrip serialize/deserialize:
        Python::with_gil(|_py| {
            let bytes = fs_handle.serialize().unwrap();
            let new_handle = FrameStackHandle::deserialize_impl(&bytes).unwrap();
            assert_eq!(fs_handle, new_handle);
        });

        let client = CamClient::new(socket_as_path.to_str().unwrap()).unwrap();

        fs_handle.with_slot(&shm, |slot_r| {
            let slice = slot_r.as_slice();
            let slice_for_frame = fs_handle.get_slice_for_frame(0, slot_r);

            // try decompression directly:
            let out_size = 256 * TryInto::<i32>::try_into(std::mem::size_of::<u16>()).unwrap();
            println!(
                "slice_for_frame.len(): {}, uncompressed_size: {}",
                slice_for_frame.len(),
                out_size
            );
            lz4::block::decompress(slice_for_frame, Some(out_size)).unwrap();

            println!("slice_for_frame: {:x?}", slice_for_frame);
            println!("slice:           {:x?}", slice);
        });

        Python::with_gil(|py| {
            let flat: Vec<u16> = (0..256).collect();
            let out = PyArray::from_vec_bound(py, flat)
                .reshape((1, 16, 16))
                .unwrap();
            let out_untyped = out.as_untyped();
            let dfsh = DectrisFrameStack::new(_PyDectrisFrameStack::new(fs_handle));
            client
                .decode_range_into_buffer(&dfsh, out_untyped, 0, dfsh.__len__().unwrap(), py)
                .unwrap();

            out.readonly()
                .as_slice()
                .unwrap()
                .iter()
                .zip(0..)
                .for_each(|(&item, idx)| {
                    assert_eq!(item, in_[idx]);
                    assert_eq!(item, (idx % 16) as u16);
                });
        });
    }
}
